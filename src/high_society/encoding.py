"""What a network sees and what it can do (spec §4.2).

`observe()` turns the game, seen by the player to act, into numbers: a vector about the player,
the card and the game, and one row per opponent (4 rows, absent seats masked). Only public
information: the deck order stays hidden. `options()` lists the 17 actions (pass, the smallest
raise, raise to a level) as engine moves, None where illegal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .engine import (
    DECK,
    MAX_PLAYERS,
    MIN_PLAYERS,
    MONEY,
    RED_CARDS_TO_END,
    STARTING_MONEY,
    Action,
    Card,
    Game,
    Kind,
    Pass,
    Player,
    Raise,
    payable,
    score_of,
)

LEVELS: tuple[int, ...] = (2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 70)
ACTIONS: tuple[str, ...] = ("pass", "min", *(f"to {level}" for level in LEVELS))
N_ACTIONS = len(ACTIONS)
OPPONENT_SLOTS = MAX_PLAYERS - 1

# The 14 distinct cards: luxury 1 to 10, prestige, passé, scandale, faux pas.
CARD_TYPES: tuple[Card, ...] = tuple(
    sorted(set(DECK), key=lambda c: (c.kind != Kind.LUXURY, c.kind, c.value))
)
CARD_INDEX = {card: i for i, card in enumerate(CARD_TYPES)}
DECK_COUNT = {card: DECK.count(card) for card in CARD_TYPES}

F32 = NDArray[np.float32]


def _money(value: float) -> float:
    return value / STARTING_MONEY


def _hand(player: Player) -> list[float]:
    """Which of the 11 money cards are still in hand (public: spent money is seen)."""
    return [float(card in player.hand) for card in MONEY]


def _holdings(player: Player) -> list[float]:
    cards = player.cards
    return [
        sum(c.value for c in cards if c.kind is Kind.LUXURY) / 55,
        sum(c.kind is Kind.LUXURY for c in cards) / 10,
        sum(c.kind is Kind.PRESTIGE for c in cards) / 3,
        float(any(c.kind is Kind.PASSE for c in cards)),
        float(any(c.kind is Kind.SCANDALE for c in cards)),
        float(player.owes_luxury),
        max(-1.0, min(score_of(cards) / 40, 2.0)),
    ]


MAIN_FEATURES = 14 + 11 + 6 + 7 + 3 + 1 + 1 + 10 + 4
OPPONENT_FEATURES = 4 + 11 + 5 + 7


@dataclass
class Observation:
    main: F32  # (MAIN_FEATURES,)
    opponents: F32  # (OPPONENT_SLOTS, OPPONENT_FEATURES)
    present: F32  # (OPPONENT_SLOTS,): 1 for a real opponent
    legal: NDArray[np.bool_]  # (N_ACTIONS,)


def observe(game: Game, moves: list[Action | None] | None = None) -> Observation:
    """The game as the player to act sees it."""
    assert game.card is not None, "No decision to make: the game is over"
    me_i = game.turn
    me = game.players[me_i]
    highest = game.highest_bid
    if moves is None:
        moves = options(game)
    poorest_other = min(p.money + p.bid_total for i, p in enumerate(game.players) if i != me_i)
    min_raise = moves[1]

    card = [0.0] * len(CARD_TYPES)
    card[CARD_INDEX[game.card]] = 1.0

    # What is still to come: the face-down cards (the one on auction is out).
    seen = (
        [c for p in game.players for c in p.cards]
        + [e.card for e in game.history if e.kind == "discard" and e.card is not None]
        + [game.card]
    )
    left = dict(DECK_COUNT)
    for c in seen:
        left[c] -= 1
    luxuries_left = [float(left[Card(Kind.LUXURY, v)]) for v in range(1, 11)]

    main = [
        *card,
        *_hand(me),
        _money(me.money),
        _money(me.bid_total),
        _money(highest),
        _money(sum(min_raise.cards)) if isinstance(min_raise, Raise) else 1.0,
        _money(me.money + me.bid_total - poorest_other),
        float(me.money + me.bid_total <= poorest_other),
        *_holdings(me),
        *(float(game.n == n) for n in range(MIN_PLAYERS, MAX_PLAYERS + 1)),
        game.reds / (RED_CARDS_TO_END - 1),
        len(game.deck) / len(DECK),
        *luxuries_left,
        left[Card(Kind.PRESTIGE)] / 3,
        float(left[Card(Kind.PASSE)]),
        float(left[Card(Kind.SCANDALE)]),
        float(left[Card(Kind.FAUX_PAS)]),
    ]

    opponents = np.zeros((OPPONENT_SLOTS, OPPONENT_FEATURES), dtype=np.float32)
    present = np.zeros(OPPONENT_SLOTS, dtype=np.float32)
    for seat in range(1, game.n):
        p = game.players[(me_i + seat) % game.n]
        distance = [0.0] * OPPONENT_SLOTS
        distance[seat - 1] = 1.0  # 1: plays right after me
        opponents[seat - 1] = [
            *distance,
            *_hand(p),
            _money(p.money),
            _money(p.bid_total),
            float(p.in_auction),
            float(p.bid_total == highest and highest > 0),
            float(p.money + p.bid_total < me.money + me.bid_total),
            *_holdings(p),
        ]
        present[seat - 1] = 1.0

    legal = np.array([m is not None for m in moves], dtype=np.bool_)
    return Observation(np.array(main, dtype=np.float32), opponents, present, legal)


def options(game: Game) -> list[Action | None]:
    """The engine move behind each of the 17 actions, None where it isn't possible."""
    me = game.players[game.turn]
    highest = game.highest_bid
    totals = sorted(payable(me.hand).items())

    def raise_to(target: int) -> Raise | None:
        need = max(target, highest + 1) - me.bid_total
        for total, cards in totals:
            if total >= need and total > 0:
                return Raise(tuple(sorted(cards)))
        return None

    moves: list[Action | None] = [Pass(), raise_to(0)]
    for level in LEVELS:
        moves.append(raise_to(level) if level > highest else None)
    return moves
