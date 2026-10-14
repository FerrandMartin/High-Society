"""Simple bots (spec §4.1): opponents for the networks from day one, and a fixed yardstick."""

from __future__ import annotations

import random
from typing import Protocol

from .engine import STARTING_MONEY, Action, Card, Game, Kind, Pass


class Bot(Protocol):
    name: str

    def act(self, game: Game) -> Action: ...


class RandomBot:
    """Passes or makes the smallest raise, at random."""

    name = "random"

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def act(self, game: Game) -> Action:
        raise_ = game.min_raise()
        if raise_ is None or self.rng.random() < 0.5:
            return Pass()
        return raise_


def card_worth(card: Card) -> float:
    """What a card is worth in money (thousands) to an average player: about 2.5 per point
    of luxury, a prestige like a big luxury card, disgraces as what avoiding them is worth."""
    if card.kind is Kind.LUXURY:
        return 2.5 * card.value
    if card.kind is Kind.PRESTIGE:
        return 25
    if card.kind is Kind.PASSE:
        return 10
    if card.kind is Kind.SCANDALE:
        return 20
    return 8  # faux pas


class ValueBot:
    """Bids up to what the card is worth, scaled by how much money it has left."""

    name = "value"

    def __init__(self, greed: float = 1.0) -> None:
        self.greed = greed

    def limit(self, game: Game) -> float:
        assert game.card is not None
        me = game.players[game.turn]
        wealth = (me.money + me.bid_total) / STARTING_MONEY
        return card_worth(game.card) * self.greed * (0.5 + wealth)

    def act(self, game: Game) -> Action:
        raise_ = game.min_raise()
        if raise_ is None:
            return Pass()
        me = game.players[game.turn]
        if me.bid_total + sum(raise_.cards) > self.limit(game):
            return Pass()
        return raise_


class SaverBot(ValueBot):
    """A value bidder that never lets its money fall to the poorest opponent's: the poorest
    player is eliminated at the end."""

    name = "saver"

    def act(self, game: Game) -> Action:
        action = super().act(game)
        if isinstance(action, Pass) or game.card is None:
            return action
        me = game.players[game.turn]
        poorest_other = min(
            p.money + p.bid_total for i, p in enumerate(game.players) if i != game.turn
        )
        left = me.money - sum(action.cards)
        # For a disgrace, money only goes if it's lost: compare with what stays in hand.
        if left + (0 if game.card.disgrace else me.bid_total) < poorest_other + 5:
            return Pass()
        return action


BOTS: dict[str, type[RandomBot] | type[ValueBot]] = {
    "random": RandomBot,
    "value": ValueBot,
    "saver": SaverBot,
}


def make_bot(name: str) -> Bot:
    if name not in BOTS:
        raise ValueError(f"Unknown bot {name!r}: one of {', '.join(BOTS)}")
    return BOTS[name]()


def play_game(bots: list[Bot], seed: int | None = None) -> Game:
    """Plays a whole game between these bots, one per seat."""
    game = Game(len(bots), seed=seed)
    while not game.over:
        game.play(bots[game.turn].act(game))
    return game
