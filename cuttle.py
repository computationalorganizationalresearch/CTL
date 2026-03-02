"""Single-file Cuttle engine suitable for self-play / AlphaZero environments.

This module intentionally focuses on deterministic, serializable game mechanics.
It implements the 2-player ruleset from cuttle.cards, including:
- draw / pass on empty deck with 3-pass stalemate
- points, scuttle (with suit tiebreak), royals, glasses eight
- one-offs A,2,3,4,5,6,7,9,10
- jack stealing points
- two-as-counter stacks against one-offs
- queen protection against targeted 2/9/J effects
- king-based dynamic goal (21/14/10/5/0)

The API is designed for AI training:
- `CuttleGameState.legal_actions()` returns complete legal actions for current player.
- `CuttleGameState.apply(action)` mutates state by one atomic move.
- `CuttleGameState.clone()` returns a deep copy.
- `CuttleGameState.observation(player)` returns a numeric-friendly dict view.

Notes:
- This file models 2-player Cuttle. 3p/4p variants are not included.
- Deck order is explicit and deterministic once shuffled externally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import random
from typing import Dict, List, Optional, Sequence, Tuple


RANK_ORDER = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
SUIT_ORDER = ["C", "D", "H", "S"]
RANK_VALUE = {r: i + 1 for i, r in enumerate(RANK_ORDER)}
SUIT_VALUE = {s: i for i, s in enumerate(SUIT_ORDER)}  # clubs weakest -> spades strongest

NUMBER_RANKS = {"A", "2", "3", "4", "5", "6", "7", "8", "9", "10"}
ROYAL_RANKS = {"J", "Q", "K"}


@dataclass(frozen=True, order=True)
class Card:
    rank: str
    suit: str

    def __post_init__(self) -> None:
        if self.rank not in RANK_ORDER:
            raise ValueError(f"Invalid rank: {self.rank}")
        if self.suit not in SUIT_ORDER:
            raise ValueError(f"Invalid suit: {self.suit}")

    @property
    def is_number(self) -> bool:
        return self.rank in NUMBER_RANKS

    @property
    def is_royal(self) -> bool:
        return self.rank in ROYAL_RANKS

    @property
    def points_value(self) -> int:
        if self.rank == "A":
            return 1
        if self.rank in {"J", "Q", "K"}:
            return 0
        return int(self.rank)

    def beats_for_scuttle(self, other: "Card") -> bool:
        """True if self can scuttle other based on rank + suit tie-break."""
        if self.points_value != other.points_value:
            return self.points_value > other.points_value
        return SUIT_VALUE[self.suit] > SUIT_VALUE[other.suit]

    def __str__(self) -> str:
        return f"{self.rank}{self.suit}"


@dataclass
class FieldCard:
    card: Card
    owner: int  # original owner for jack-return semantics
    controller: int  # current controller
    kind: str  # "point" | "royal" | "glasses"
    attached_to: Optional[int] = None  # index into board points list for jack attachment


class ActionType(str, Enum):
    DRAW = "draw"
    PASS = "pass"
    PLAY_POINT = "play_point"
    PLAY_ROYAL = "play_royal"  # includes glasses eight
    PLAY_ONE_OFF = "play_one_off"
    SCUTTLE = "scuttle"
    PLAY_JACK_STEAL = "play_jack_steal"
    COUNTER = "counter"


@dataclass(frozen=True)
class Action:
    type: ActionType
    hand_index: Optional[int] = None
    target_player: Optional[int] = None
    target_zone: Optional[str] = None  # "point" | "royal"
    target_index: Optional[int] = None
    aux_index: Optional[int] = None  # used by one-offs for selected discard/scrap indices


@dataclass
class PlayerState:
    hand: List[Card] = field(default_factory=list)
    points: List[FieldCard] = field(default_factory=list)
    royals: List[FieldCard] = field(default_factory=list)
    cannot_play_cards: List[Card] = field(default_factory=list)  # from enemy nine effect, one-turn lock

    def has_queen_protection(self) -> bool:
        return any(fc.card.rank == "Q" for fc in self.royals)

    def total_points(self) -> int:
        return sum(fc.card.points_value for fc in self.points)

    def king_count(self) -> int:
        return sum(1 for fc in self.royals if fc.card.rank == "K")


@dataclass
class PendingOneOff:
    source_player: int
    source_card: Card


@dataclass
class CuttleGameState:
    players: List[PlayerState]
    deck: List[Card]
    scrap: List[Card]
    turn: int
    winner: Optional[int] = None
    stalemate: bool = False
    consecutive_passes: int = 0
    pending_one_off: Optional[PendingOneOff] = None

    @classmethod
    def new_game(cls, seed: Optional[int] = None, dealer: int = 0) -> "CuttleGameState":
        rng = random.Random(seed)
        deck = [Card(rank, suit) for rank in RANK_ORDER for suit in SUIT_ORDER]
        rng.shuffle(deck)
        # dealer gets 6, opponent gets 5, player with 5 starts
        p0 = PlayerState()
        p1 = PlayerState()
        players = [p0, p1]
        dealer_hand = 6
        other_hand = 5
        players[dealer].hand.extend(deck.pop() for _ in range(dealer_hand))
        players[1 - dealer].hand.extend(deck.pop() for _ in range(other_hand))
        turn = 1 - dealer
        return cls(players=players, deck=deck, scrap=[], turn=turn)

    def clone(self) -> "CuttleGameState":
        import copy

        return copy.deepcopy(self)

    def current(self) -> PlayerState:
        return self.players[self.turn]

    def opponent(self) -> PlayerState:
        return self.players[1 - self.turn]

    def points_goal(self, player_idx: int) -> int:
        kings = self.players[player_idx].king_count()
        return [21, 14, 10, 5, 0][min(kings, 4)]

    def check_win(self) -> None:
        for i, p in enumerate(self.players):
            if p.total_points() >= self.points_goal(i):
                self.winner = i
                return

    def legal_actions(self) -> List[Action]:
        if self.winner is not None or self.stalemate:
            return []

        # Counter window: opponent may chain 2s to counter pending one-off
        if self.pending_one_off is not None:
            actor = 1 - self.pending_one_off.source_player
            if actor != self.turn:
                return []
            hand = self.players[actor].hand
            return [
                Action(ActionType.COUNTER, hand_index=i)
                for i, c in enumerate(hand)
                if c.rank == "2"
            ] + [Action(ActionType.PASS)]

        actions: List[Action] = []
        me = self.current()
        opp = self.opponent()

        if self.deck and len(me.hand) < 8:
            actions.append(Action(ActionType.DRAW))
        if not self.deck:
            actions.append(Action(ActionType.PASS))

        for i, c in enumerate(me.hand):
            if c in me.cannot_play_cards:
                continue

            if c.rank in NUMBER_RANKS:
                actions.append(Action(ActionType.PLAY_POINT, hand_index=i))

            if c.rank in {"Q", "K"}:
                actions.append(Action(ActionType.PLAY_ROYAL, hand_index=i))
            if c.rank == "8":
                actions.append(Action(ActionType.PLAY_ROYAL, hand_index=i))  # glasses eight

            if c.rank == "J":
                for ti, tgt in enumerate(opp.points):
                    actions.append(
                        Action(
                            ActionType.PLAY_JACK_STEAL,
                            hand_index=i,
                            target_player=1 - self.turn,
                            target_zone="point",
                            target_index=ti,
                        )
                    )

            for ti, tgt in enumerate(opp.points):
                if c.rank in NUMBER_RANKS and c.beats_for_scuttle(tgt.card):
                    actions.append(
                        Action(
                            ActionType.SCUTTLE,
                            hand_index=i,
                            target_player=1 - self.turn,
                            target_zone="point",
                            target_index=ti,
                        )
                    )

            if c.rank in {"A", "2", "3", "4", "5", "6", "7", "9", "10"}:
                actions.append(Action(ActionType.PLAY_ONE_OFF, hand_index=i))

        return actions

    def _scrap_field_card(self, player_idx: int, zone: str, index: int) -> None:
        p = self.players[player_idx]
        pile = p.points if zone == "point" else p.royals
        fc = pile.pop(index)
        self.scrap.append(fc.card)
        # if jack was attached to this point, scrap attached jack(s)
        if zone == "point":
            attached = [r for r in p.royals if r.card.rank == "J" and r.attached_to == index]
            for j in attached:
                p.royals.remove(j)
                self.scrap.append(j.card)

    def _play_point(self, player_idx: int, card: Card) -> None:
        self.players[player_idx].points.append(
            FieldCard(card=card, owner=player_idx, controller=player_idx, kind="point")
        )

    def _play_royal(self, player_idx: int, card: Card) -> None:
        kind = "glasses" if card.rank == "8" else "royal"
        self.players[player_idx].royals.append(
            FieldCard(card=card, owner=player_idx, controller=player_idx, kind=kind)
        )

    def _resolve_one_off(self, player_idx: int, card: Card, chooser: Optional[int] = None) -> None:
        """Resolve an already-uncountered one-off.

        chooser: optional deterministic choice index for ambiguous effects.
        """
        me = self.players[player_idx]
        opp_idx = 1 - player_idx
        opp = self.players[opp_idx]

        if card.rank == "A":
            for i in [0, 1]:
                while self.players[i].points:
                    self.scrap.append(self.players[i].points.pop().card)
            return

        if card.rank == "2":
            # On-turn use = scrap target royal/glasses (if any legal target exists)
            targets: List[Tuple[int, int]] = []
            for pi in [0, 1]:
                for ri, rfc in enumerate(self.players[pi].royals):
                    if pi == opp_idx and opp.has_queen_protection() and rfc.card.rank != "Q":
                        continue
                    targets.append((pi, ri))
            if not targets:
                return
            pi, ri = targets[0 if chooser is None else chooser % len(targets)]
            self.scrap.append(self.players[pi].royals.pop(ri).card)
            return

        if card.rank == "3":
            candidates = [c for c in self.scrap if c.rank != "3"]
            if not candidates:
                return
            pick = candidates[0 if chooser is None else chooser % len(candidates)]
            self.scrap.remove(pick)
            me.hand.append(pick)
            return

        if card.rank == "4":
            for _ in range(min(2, len(opp.hand))):
                idx = 0 if chooser is None else chooser % len(opp.hand)
                self.scrap.append(opp.hand.pop(idx))
            return

        if card.rank == "5":
            if me.hand:
                idx = 0 if chooser is None else chooser % len(me.hand)
                self.scrap.append(me.hand.pop(idx))
            draws = min(3, max(0, 8 - len(me.hand)), len(self.deck))
            for _ in range(draws):
                me.hand.append(self.deck.pop())
            return

        if card.rank == "6":
            for i in [0, 1]:
                while self.players[i].royals:
                    self.scrap.append(self.players[i].royals.pop().card)
            return

        if card.rank == "7":
            revealed: List[Card] = []
            for _ in range(min(2, len(self.deck))):
                revealed.append(self.deck.pop())
            if not revealed:
                return

            playable = []
            for c in revealed:
                if c.rank == "J" and not opp.points:
                    continue
                playable.append(c)

            if not playable:
                # edge case from FAQ: two unplayable jacks, scrap one keep one on deck
                self.scrap.append(revealed[0])
                if len(revealed) > 1:
                    self.deck.append(revealed[1])
                return

            chosen = playable[0 if chooser is None else chooser % len(playable)]
            other = [c for c in revealed if c is not chosen]
            if other:
                self.deck.extend(other)

            if chosen.rank in NUMBER_RANKS:
                self._play_point(player_idx, chosen)
            elif chosen.rank in {"Q", "K", "8"}:
                self._play_royal(player_idx, chosen)
            elif chosen.rank == "J":
                # choose first enemy point
                target = opp.points[0]
                opp.points.pop(0)
                me.points.append(target)
                me.royals.append(FieldCard(card=chosen, owner=player_idx, controller=player_idx, kind="royal", attached_to=0))
            else:
                # A,2,3,4,5,6,7,9,10 resolved as one-off from reveal
                self.scrap.append(chosen)
                self._resolve_one_off(player_idx, chosen, chooser=chooser)
            return

        if card.rank == "9":
            if not opp.points and not opp.royals:
                return
            # targets opponent field (queens protect non-queen cards)
            target_points = [("point", i) for i, _ in enumerate(opp.points)]
            target_royals = []
            for i, r in enumerate(opp.royals):
                if opp.has_queen_protection() and r.card.rank != "Q":
                    continue
                target_royals.append(("royal", i))
            targets = target_points + target_royals
            if not targets:
                return
            zone, idx = targets[0 if chooser is None else chooser % len(targets)]
            pile = opp.points if zone == "point" else opp.royals
            bounced = pile.pop(idx).card
            opp.hand.append(bounced)
            opp.cannot_play_cards.append(bounced)
            return

        # 10 has no one-off effect

    def apply(self, action: Action, chooser: Optional[int] = None) -> None:
        if action not in self.legal_actions():
            raise ValueError(f"Illegal action: {action}")

        me = self.current()
        opp_idx = 1 - self.turn

        if self.pending_one_off is not None:
            if action.type == ActionType.PASS:
                src = self.pending_one_off
                self.pending_one_off = None
                self._resolve_one_off(src.source_player, src.source_card, chooser=chooser)
                self.turn = opp_idx
                self.check_win()
                return

            # action is counter with 2
            assert action.hand_index is not None
            two = me.hand.pop(action.hand_index)
            self.scrap.append(two)
            self.scrap.append(self.pending_one_off.source_card)
            # new pending one-off = this counter two (stackable)
            self.pending_one_off = PendingOneOff(source_player=self.turn, source_card=two)
            self.turn = opp_idx
            return

        self.consecutive_passes = 0
        me.cannot_play_cards.clear()  # expired at start of owner turn

        if action.type == ActionType.DRAW:
            me.hand.append(self.deck.pop())
            self.turn = opp_idx
        elif action.type == ActionType.PASS:
            self.consecutive_passes += 1
            if self.consecutive_passes >= 3:
                self.stalemate = True
            self.turn = opp_idx
        else:
            assert action.hand_index is not None
            card = me.hand.pop(action.hand_index)

            if action.type == ActionType.PLAY_POINT:
                self._play_point(self.turn, card)
                self.turn = opp_idx
            elif action.type == ActionType.PLAY_ROYAL:
                self._play_royal(self.turn, card)
                self.turn = opp_idx
            elif action.type == ActionType.SCUTTLE:
                assert action.target_index is not None and action.target_player is not None
                self._scrap_field_card(action.target_player, "point", action.target_index)
                self.scrap.append(card)
                self.turn = opp_idx
            elif action.type == ActionType.PLAY_JACK_STEAL:
                assert action.target_index is not None
                if not self.opponent().points:
                    self.scrap.append(card)
                else:
                    stolen = self.opponent().points.pop(action.target_index)
                    me.points.append(stolen)
                    me.royals.append(
                        FieldCard(card=card, owner=self.turn, controller=self.turn, kind="royal", attached_to=len(me.points) - 1)
                    )
                self.turn = opp_idx
            elif action.type == ActionType.PLAY_ONE_OFF:
                self.scrap.append(card)
                self.pending_one_off = PendingOneOff(source_player=self.turn, source_card=card)
                self.turn = opp_idx
            else:
                raise ValueError(f"Unsupported action type {action.type}")

        self.check_win()

    def observation(self, player: int) -> Dict:
        """Partial-information observation for RL agents."""
        me = self.players[player]
        opp = self.players[1 - player]

        return {
            "turn": self.turn,
            "winner": self.winner,
            "stalemate": self.stalemate,
            "deck_count": len(self.deck),
            "scrap_count": len(self.scrap),
            "my_hand": [str(c) for c in me.hand],
            "my_points": [str(fc.card) for fc in me.points],
            "my_royals": [str(fc.card) for fc in me.royals],
            "opp_hand_count": len(opp.hand),
            "opp_points": [str(fc.card) for fc in opp.points],
            "opp_royals": [str(fc.card) for fc in opp.royals],
            "my_goal": self.points_goal(player),
            "opp_goal": self.points_goal(1 - player),
        }


def random_playout(seed: int = 0, max_turns: int = 200) -> Tuple[Optional[int], bool, int]:
    """Utility for smoke-testing environment behavior."""
    rng = random.Random(seed)
    g = CuttleGameState.new_game(seed=seed)
    turns = 0
    while g.winner is None and not g.stalemate and turns < max_turns:
        legal = g.legal_actions()
        if not legal:
            break
        action = rng.choice(legal)
        g.apply(action)
        turns += 1
    return g.winner, g.stalemate, turns


if __name__ == "__main__":
    winner, stalemate, turns = random_playout(seed=42)
    print({"winner": winner, "stalemate": stalemate, "turns": turns})
