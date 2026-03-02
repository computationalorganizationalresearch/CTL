#!/usr/bin/env python3
"""Single-file AlphaZero-style trainer for `cuttle.py`.

Highlights
- Uses a policy-value network with GPU-accelerated batched inference/training.
- Runs many games concurrently (vectorized game loop in Python, GPU for NN math).
- Adds AlphaZero-style exploration (Dirichlet noise + temperature schedule).
- Trains from self-play and tracks/exports the best model to ONNX.

Example
    python train_alphazero_cuttle.py \
      --num-games 1000000 \
      --parallel-games 4096 \
      --device cuda \
      --save-best-onnx best_cuttle.onnx

Notes
- `cuttle.py` is a Python environment with rich object actions, so environment stepping stays on CPU.
  This script compensates by batching *all model work* on GPU.
- Reaching very high throughput (e.g., ~1M games / ~3h on A100) depends heavily on
  chosen hyperparameters (`--parallel-games`, `--max-turns`, `--num-workers`, etc.) and
  environment-specific Python overhead.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import math
import os
import random
import time
from typing import Deque, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cuttle import Action, ActionType, CuttleGameState, RANK_ORDER, SUIT_ORDER


MAX_HAND = 8
MAX_FIELD = 16
CARD_DIM = 52
STATE_DIM = CARD_DIM * 6 + 16
ACTION_DIM = 64

RANK_TO_I = {r: i for i, r in enumerate(RANK_ORDER)}
SUIT_TO_I = {s: i for i, s in enumerate(SUIT_ORDER)}
ACTION_TYPE_TO_I = {t: i for i, t in enumerate(ActionType)}


def card_index(card) -> int:
    return RANK_TO_I[card.rank] * 4 + SUIT_TO_I[card.suit]


def encode_cards_multi(cards: Sequence, out: np.ndarray) -> None:
    out.fill(0.0)
    for c in cards:
        out[card_index(c)] += 1.0


def encode_state(game: CuttleGameState) -> np.ndarray:
    """Encodes state from current player's perspective into a fixed vector."""
    me_i = game.turn
    opp_i = 1 - me_i
    me = game.players[me_i]
    opp = game.players[opp_i]

    chunks = [np.zeros(CARD_DIM, dtype=np.float32) for _ in range(6)]
    encode_cards_multi(me.hand, chunks[0])
    encode_cards_multi([fc.card for fc in me.points], chunks[1])
    encode_cards_multi([fc.card for fc in me.royals], chunks[2])
    encode_cards_multi([fc.card for fc in opp.points], chunks[3])
    encode_cards_multi([fc.card for fc in opp.royals], chunks[4])
    encode_cards_multi(game.scrap, chunks[5])

    scalars = np.array(
        [
            game.turn,
            len(game.deck) / 52.0,
            len(game.scrap) / 52.0,
            len(me.hand) / MAX_HAND,
            len(opp.hand) / MAX_HAND,
            me.total_points() / 21.0,
            opp.total_points() / 21.0,
            game.points_goal(me_i) / 21.0,
            game.points_goal(opp_i) / 21.0,
            float(game.pending_one_off is not None),
            float(game.stalemate),
            float(game.consecutive_passes) / 3.0,
            float(me.has_queen_protection()),
            float(opp.has_queen_protection()),
            float(game.winner is not None),
            float((game.winner == me_i) if game.winner is not None else 0.0),
        ],
        dtype=np.float32,
    )
    return np.concatenate([*chunks, scalars], axis=0)


def encode_action(game: CuttleGameState, action: Action) -> np.ndarray:
    """Action feature encoding (relative to current player)."""
    out = np.zeros(ACTION_DIM, dtype=np.float32)
    out[ACTION_TYPE_TO_I[action.type]] = 1.0

    me = game.players[game.turn]
    if action.hand_index is not None and 0 <= action.hand_index < len(me.hand):
        c = me.hand[action.hand_index]
        out[8 + card_index(c)] = 1.0

    base = 8 + CARD_DIM
    out[base + (action.target_player or 0)] = 1.0
    zone = 0 if action.target_zone in (None, "point") else 1
    out[base + 2 + zone] = 1.0

    if action.target_index is not None:
        out[base + 4 + min(action.target_index, 7)] = 1.0
    if action.aux_index is not None:
        out[base + 12 + min(action.aux_index, 7)] = 1.0

    out[-1] = 1.0  # bias feature
    return out


class PolicyValueNet(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM, hidden: int = 512):
        super().__init__()
        self.state_body = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
            nn.Tanh(),
        )
        self.action_embed = nn.Sequential(
            nn.Linear(action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.policy_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, states: torch.Tensor, action_feats: torch.Tensor, action_batch_index: torch.Tensor):
        """
        states: [B, state_dim]
        action_feats: [N, action_dim] for all legal actions across batch
        action_batch_index: [N] maps each action to its state index in [0, B)
        """
        s = self.state_body(states)  # [B, H]
        v = self.value_head(s).squeeze(-1)  # [B]

        a = self.action_embed(action_feats)  # [N, H]
        sp = self.policy_proj(s)  # [B, H]
        logits = (a * sp[action_batch_index]).sum(dim=-1) / math.sqrt(s.shape[-1])
        return logits, v


@dataclasses.dataclass
class StepRecord:
    state: np.ndarray
    policy: np.ndarray
    value_player: int


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.items: Deque[Tuple[np.ndarray, np.ndarray, float]] = collections.deque(maxlen=capacity)

    def add_episode(self, episode: List[StepRecord], winner: Optional[int], stalemate: bool) -> None:
        if stalemate or winner is None:
            z = 0.0
            for rec in episode:
                self.items.append((rec.state, rec.policy, z))
            return
        for rec in episode:
            z = 1.0 if rec.value_player == winner else -1.0
            self.items.append((rec.state, rec.policy, z))

    def __len__(self) -> int:
        return len(self.items)

    def sample(self, batch_size: int):
        batch = random.sample(self.items, batch_size)
        s = torch.from_numpy(np.stack([x[0] for x in batch]))
        p = [x[1] for x in batch]
        z = torch.tensor([x[2] for x in batch], dtype=torch.float32)
        return s, p, z


def masked_policy_from_logits(logits: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 1e-6:
        out = np.zeros_like(logits)
        out[np.argmax(logits)] = 1.0
        return out
    x = logits / temperature
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / (np.sum(ex) + 1e-8)


def choose_actions_batched(
    games: List[CuttleGameState],
    model: PolicyValueNet,
    device: torch.device,
    dirichlet_alpha: float,
    dirichlet_eps: float,
    temperature: float,
) -> Tuple[List[Action], List[np.ndarray], List[np.ndarray], List[int]]:
    """Runs one large GPU inference pass for all active games and picks actions."""
    legal_lists: List[List[Action]] = [g.legal_actions() for g in games]
    states_np = [encode_state(g) for g in games]

    action_feats_np: List[np.ndarray] = []
    action_batch_idx: List[int] = []
    per_game_ranges: List[Tuple[int, int]] = []
    cursor = 0
    for bi, (g, acts) in enumerate(zip(games, legal_lists)):
        start = cursor
        for a in acts:
            action_feats_np.append(encode_action(g, a))
            action_batch_idx.append(bi)
            cursor += 1
        per_game_ranges.append((start, cursor))

    states = torch.from_numpy(np.stack(states_np)).to(device)
    action_feats = torch.from_numpy(np.stack(action_feats_np)).to(device)
    action_batch_index = torch.tensor(action_batch_idx, dtype=torch.long, device=device)

    with torch.no_grad():
        logits_t, values_t = model(states, action_feats, action_batch_index)
    logits = logits_t.detach().cpu().numpy()
    values = values_t.detach().cpu().numpy()

    chosen_actions: List[Action] = []
    target_policies: List[np.ndarray] = []
    value_players: List[int] = []

    for gi, (acts, (s, e), g) in enumerate(zip(legal_lists, per_game_ranges, games)):
        local_logits = logits[s:e]
        policy = masked_policy_from_logits(local_logits, temperature)

        # AlphaZero-style root exploration noise
        if len(policy) > 1 and dirichlet_eps > 0.0:
            noise = np.random.dirichlet([dirichlet_alpha] * len(policy)).astype(np.float32)
            policy = (1.0 - dirichlet_eps) * policy + dirichlet_eps * noise
            policy = policy / (policy.sum() + 1e-8)

        ai = int(np.random.choice(len(acts), p=policy))
        chosen_actions.append(acts[ai])

        dense_policy = np.zeros(256, dtype=np.float32)
        capped = min(len(policy), 256)
        dense_policy[:capped] = policy[:capped]
        target_policies.append(dense_policy)
        value_players.append(g.turn)

    return chosen_actions, states_np, target_policies, value_players


def train_step(
    model: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    batch_size: int,
    device: torch.device,
) -> Optional[Tuple[float, float, float]]:
    if len(replay) < batch_size:
        return None

    states_cpu, policies_list, z = replay.sample(batch_size)
    states = states_cpu.to(device)
    z = z.to(device)

    action_feats_np: List[np.ndarray] = []
    action_batch_idx: List[int] = []
    target_idx = []
    for bi, dense in enumerate(policies_list):
        valid = np.where(dense > 0)[0]
        if len(valid) == 0:
            valid = np.array([0], dtype=np.int64)
        start = len(action_feats_np)
        for vi in valid:
            af = np.zeros(ACTION_DIM, dtype=np.float32)
            af[-1] = 1.0
            af[8 + (vi % CARD_DIM)] = 1.0
            action_feats_np.append(af)
            action_batch_idx.append(bi)
        target_idx.append(start + int(np.argmax(dense[valid])))

    action_feats = torch.from_numpy(np.stack(action_feats_np)).to(device)
    action_batch_index = torch.tensor(action_batch_idx, dtype=torch.long, device=device)
    logits, values = model(states, action_feats, action_batch_index)

    policy_loss = torch.tensor(0.0, device=device)
    for bi in range(batch_size):
        idx = (action_batch_index == bi).nonzero(as_tuple=False).squeeze(-1)
        l = logits[idx]
        tgt = torch.tensor([target_idx[bi] - int(idx[0].item())], device=device)
        policy_loss = policy_loss + F.cross_entropy(l.unsqueeze(0), tgt)
    policy_loss = policy_loss / batch_size

    value_loss = F.mse_loss(values, z)
    loss = policy_loss + value_loss

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return float(loss.item()), float(policy_loss.item()), float(value_loss.item())


def export_best_onnx(model: PolicyValueNet, path: str, device: torch.device) -> None:
    model.eval()
    dummy_states = torch.randn(2, STATE_DIM, device=device)
    dummy_actions = torch.randn(6, ACTION_DIM, device=device)
    dummy_batch_idx = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long, device=device)
    torch.onnx.export(
        model,
        (dummy_states, dummy_actions, dummy_batch_idx),
        path,
        input_names=["states", "action_features", "action_batch_index"],
        output_names=["policy_logits", "state_values"],
        dynamic_axes={
            "states": {0: "batch"},
            "action_features": {0: "num_actions"},
            "action_batch_index": {0: "num_actions"},
            "policy_logits": {0: "num_actions"},
            "state_values": {0: "batch"},
        },
        opset_version=17,
    )


def run(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = PolicyValueNet(hidden=args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    replay = ReplayBuffer(capacity=args.replay_capacity)

    best_score = -1e9
    finished_games = 0
    total_steps = 0
    start_time = time.time()

    games = [CuttleGameState.new_game(seed=args.seed + i) for i in range(args.parallel_games)]
    episodes: List[List[StepRecord]] = [[] for _ in games]

    while finished_games < args.num_games:
        active_idx = [i for i, g in enumerate(games) if g.winner is None and not g.stalemate]
        if not active_idx:
            games = [CuttleGameState.new_game(seed=args.seed + finished_games + i) for i in range(args.parallel_games)]
            episodes = [[] for _ in games]
            continue

        active_games = [games[i] for i in active_idx]
        t = max(0.2, args.temperature_start * (1.0 - finished_games / max(1, args.num_games)))
        chosen_actions, states_np, policies, value_players = choose_actions_batched(
            active_games,
            model,
            device,
            args.dirichlet_alpha,
            args.dirichlet_eps,
            t,
        )

        for local_i, gi in enumerate(active_idx):
            g = games[gi]
            episodes[gi].append(StepRecord(states_np[local_i], policies[local_i], value_players[local_i]))
            g.apply(chosen_actions[local_i])
            total_steps += 1

            if g.winner is not None or g.stalemate or len(episodes[gi]) >= args.max_turns:
                if len(episodes[gi]) >= args.max_turns and g.winner is None:
                    g.stalemate = True
                replay.add_episode(episodes[gi], g.winner, g.stalemate)
                finished_games += 1

                games[gi] = CuttleGameState.new_game(seed=args.seed + finished_games + gi)
                episodes[gi] = []

                if finished_games % args.log_every == 0:
                    elapsed = time.time() - start_time
                    gps = finished_games / max(elapsed, 1e-6)
                    sps = total_steps / max(elapsed, 1e-6)
                    print(
                        f"games={finished_games}/{args.num_games} "
                        f"replay={len(replay)} gps={gps:.2f} steps/s={sps:.2f}",
                        flush=True,
                    )

        for _ in range(args.train_steps_per_iter):
            out = train_step(model, optimizer, replay, args.batch_size, device)
            if out is None:
                break

        if finished_games > 0 and finished_games % args.eval_every == 0:
            score = evaluate_vs_random(model, device, games=args.eval_games, max_turns=args.max_turns)
            print(f"eval_score={score:.4f}", flush=True)
            if score > best_score:
                best_score = score
                torch.save(model.state_dict(), args.save_best_pt)
                export_best_onnx(model, args.save_best_onnx, device)
                print(f"new_best={best_score:.4f} saved to {args.save_best_onnx}", flush=True)

    if not os.path.exists(args.save_best_onnx):
        torch.save(model.state_dict(), args.save_best_pt)
        export_best_onnx(model, args.save_best_onnx, device)

    elapsed = time.time() - start_time
    print(
        f"Done. games={finished_games}, elapsed={elapsed/3600:.2f}h, "
        f"games/hour={finished_games / max(elapsed/3600, 1e-6):.0f}, best_score={best_score:.4f}",
        flush=True,
    )


def evaluate_vs_random(model: PolicyValueNet, device: torch.device, games: int, max_turns: int) -> float:
    wins = 0
    draws = 0
    model.eval()
    with torch.no_grad():
        for seed in range(games):
            g = CuttleGameState.new_game(seed=1000000 + seed)
            turns = 0
            while g.winner is None and not g.stalemate and turns < max_turns:
                legal = g.legal_actions()
                if g.turn == 0:
                    s = torch.from_numpy(encode_state(g)).unsqueeze(0).to(device)
                    af = torch.from_numpy(np.stack([encode_action(g, a) for a in legal])).to(device)
                    bi = torch.zeros(len(legal), dtype=torch.long, device=device)
                    logits, _ = model(s, af, bi)
                    pi = int(torch.argmax(logits).item())
                    g.apply(legal[pi])
                else:
                    g.apply(random.choice(legal))
                turns += 1
            if g.winner == 0:
                wins += 1
            elif g.winner is None:
                draws += 1
    model.train()
    return (wins + 0.5 * draws) / games


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fast AlphaZero-style Cuttle trainer")
    p.add_argument("--num-games", type=int, default=200000)
    p.add_argument("--parallel-games", type=int, default=2048)
    p.add_argument("--max-turns", type=int, default=120)
    p.add_argument("--train-steps-per-iter", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--replay-capacity", type=int, default=2_000_000)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--temperature-start", type=float, default=1.2)
    p.add_argument("--dirichlet-alpha", type=float, default=0.3)
    p.add_argument("--dirichlet-eps", type=float, default=0.25)
    p.add_argument("--eval-every", type=int, default=10000)
    p.add_argument("--eval-games", type=int, default=128)
    p.add_argument("--log-every", type=int, default=2000)
    p.add_argument("--save-best-pt", type=str, default="best_cuttle.pt")
    p.add_argument("--save-best-onnx", type=str, default="best_cuttle.onnx")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
