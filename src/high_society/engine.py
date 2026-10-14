"""The rules of High Society (spec §2): the only place they exist.

The engine is pure and deterministic given its random seed: a `Game` holds the whole state,
`legal_actions()` lists what the player to act may do, and `play()` applies an action. Bots,
the trainer and the web server all drive games through this interface.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import combinations

MONEY: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 10, 12, 15, 20, 25)
STARTING_MONEY = sum(MONEY)
MIN_PLAYERS, MAX_PLAYERS = 3, 5
RED_CARDS_TO_END = 4


class Kind(StrEnum):
    LUXURY = "luxury"
    PRESTIGE = "prestige"  # score ×2
    PASSE = "passe"  # −5
    SCANDALE = "scandale"  # score ÷2
    FAUX_PAS = "faux_pas"  # discard a luxury card


@dataclass(frozen=True, order=True)
class Card:
    kind: Kind
    value: int = 0  # luxury cards only: 1 to 10

    @property
    def red(self) -> bool:
        """Red-bordered: the 4th one revealed ends the game."""
        return self.kind in (Kind.PRESTIGE, Kind.SCANDALE)

    @property
    def disgrace(self) -> bool:
        """Auctioned in reverse: players bid to avoid it."""
        return self.kind in (Kind.PASSE, Kind.SCANDALE, Kind.FAUX_PAS)

    def __str__(self) -> str:
        return str(self.value) if self.kind is Kind.LUXURY else self.kind.value


DECK: tuple[Card, ...] = (
    *(Card(Kind.LUXURY, v) for v in range(1, 11)),
    Card(Kind.PRESTIGE),
    Card(Kind.PRESTIGE),
    Card(Kind.PRESTIGE),
    Card(Kind.PASSE),
    Card(Kind.SCANDALE),
    Card(Kind.FAUX_PAS),
)


@dataclass(frozen=True)
class Pass:
    def __str__(self) -> str:
        return "pass"


@dataclass(frozen=True)
class Raise:
    """Adds these money cards to the bid in front of the player."""

    cards: tuple[int, ...]

    def __str__(self) -> str:
        return "+" + "+".join(map(str, self.cards))


Action = Pass | Raise


class IllegalAction(ValueError):
    pass


@dataclass
class Player:
    hand: list[int] = field(default_factory=lambda: list(MONEY))
    bid: list[int] = field(default_factory=list)
    cards: list[Card] = field(default_factory=list)
    in_auction: bool = True
    # A faux pas taken with no luxury card: the next luxury card won is discarded.
    owes_luxury: bool = False

    @property
    def money(self) -> int:
        """Money in hand, not counting the current bid."""
        return sum(self.hand)

    @property
    def bid_total(self) -> int:
        return sum(self.bid)


@dataclass(frozen=True)
class Event:
    """One line of the game's history, for replays and the web UI."""

    kind: str  # "reveal", "raise", "pass", "win", "take", "discard", "end"
    player: int | None = None
    card: Card | None = None
    money: tuple[int, ...] = ()


def score_of(cards: Iterable[Card]) -> float:
    """Luxury values, −5 for passé, ×2 per prestige, ÷2 for scandale (spec §2)."""
    cards = list(cards)
    total: float = sum(c.value for c in cards if c.kind is Kind.LUXURY)
    total -= 5 * sum(c.kind is Kind.PASSE for c in cards)
    total *= 2 ** sum(c.kind is Kind.PRESTIGE for c in cards)
    total /= 2 ** sum(c.kind is Kind.SCANDALE for c in cards)
    return total


def cheapest_cards(hand: Iterable[int], at_least: int) -> tuple[int, ...] | None:
    """The money cards adding up to `at_least` or more at the lowest total (then the fewest
    cards); None when the hand can't reach it. There is no change in this game."""
    hand = sorted(hand)
    if at_least <= 0:
        return ()
    best: tuple[int, ...] | None = None
    for size in range(1, len(hand) + 1):
        for combo in combinations(hand, size):
            total = sum(combo)
            if total >= at_least and (best is None or total < sum(best)):
                best = combo
    return best


class Game:
    def __init__(
        self,
        players: int,
        seed: int | None = None,
        *,
        order: list[Card] | None = None,
        starter: int | None = None,
    ) -> None:
        """`order` (the cards in the order they'll be revealed) and `starter` fix what the
        seed would otherwise draw: for tests and replays."""
        if not MIN_PLAYERS <= players <= MAX_PLAYERS:
            raise ValueError(f"High Society is for {MIN_PLAYERS} to {MAX_PLAYERS} players")
        self.rng = random.Random(seed)
        self.players = [Player() for _ in range(players)]
        if order is None:
            self.deck = list(DECK)
            self.rng.shuffle(self.deck)
        else:
            if sorted(order) != sorted(DECK):
                raise ValueError("The order must hold the 16 cards of the deck")
            self.deck = list(reversed(order))  # the end of the list is the top
        self.card: Card | None = None  # up for auction
        self.turn = self.rng.randrange(players) if starter is None else starter
        self.reds = 0
        self.over = False
        self.history: list[Event] = []
        self._reveal(starter=self.turn)

    # --- reading the state ---

    @property
    def n(self) -> int:
        return len(self.players)

    @property
    def highest_bid(self) -> int:
        return max(p.bid_total for p in self.players)

    def legal_actions(self) -> list[Action]:
        """Pass, and every distinct raise (as sets of cards) that beats the highest bid."""
        if self.over:
            return []
        player = self.players[self.turn]
        actions: list[Action] = [Pass()]
        need = self.highest_bid - player.bid_total + 1
        seen: set[tuple[int, ...]] = set()
        hand = sorted(player.hand)
        for size in range(1, len(hand) + 1):
            for combo in combinations(hand, size):
                if sum(combo) >= need and combo not in seen:
                    seen.add(combo)
                    actions.append(Raise(combo))
        return actions

    def raise_to(self, total: int) -> Raise | None:
        """The cheapest raise making the player's bid at least `total` and above the highest
        bid; None if they can't. Used by bots and the networks' "raise to a level" actions."""
        player = self.players[self.turn]
        target = max(total, self.highest_bid + 1)
        cards = cheapest_cards(player.hand, target - player.bid_total)
        return Raise(cards) if cards else None

    def min_raise(self) -> Raise | None:
        return self.raise_to(0)

    # --- playing ---

    def play(self, action: Action) -> None:
        if self.over:
            raise IllegalAction("The game is over")
        player = self.players[self.turn]
        assert self.card is not None
        if isinstance(action, Raise):
            if not action.cards:
                raise IllegalAction("A raise needs money cards")
            hand = list(player.hand)
            for c in action.cards:
                if c not in hand:
                    raise IllegalAction(f"Card {c} is not in hand")
                hand.remove(c)
            if player.bid_total + sum(action.cards) <= self.highest_bid:
                raise IllegalAction("A raise must beat the highest bid")
            player.hand = hand
            player.bid.extend(action.cards)
            self._log("raise", self.turn, money=action.cards)
            self._next_turn()
            return

        self._log("pass", self.turn)
        if self.card.disgrace:
            # The first to pass takes the card and their bid back; the others pay.
            taker = self.turn
            for i, p in enumerate(self.players):
                if i == taker:
                    p.hand.extend(p.bid)
                elif p.bid:
                    self._log("discard", i, money=tuple(p.bid))
                p.bid = []
            self._give(taker, self.card, "take")
            self._reveal(starter=taker)
            return

        player.hand.extend(player.bid)
        player.bid = []
        player.in_auction = False
        still_in = [i for i, p in enumerate(self.players) if p.in_auction]
        if len(still_in) == 1:
            winner = still_in[0]
            spent = self.players[winner].bid
            if spent:
                self._log("discard", winner, money=tuple(spent))
            self.players[winner].bid = []
            self._give(winner, self.card, "win")
            self._reveal(starter=winner)
        else:
            self._next_turn()

    # --- end of the game ---

    def eliminated(self) -> list[int]:
        """The poorest player(s): all of them on a tie, so everyone if they all have the same
        money (then nobody wins)."""
        money = [p.money for p in self.players]
        return [i for i, m in enumerate(money) if m == min(money)]

    def scores(self) -> list[float]:
        return [score_of(p.cards) for p in self.players]

    def winners(self) -> list[int]:
        """Highest score among those not eliminated; ties broken by money left, then by the
        highest luxury card. Several winners only on a complete tie; none if everyone is
        eliminated."""
        if not self.over:
            raise ValueError("The game is not over")
        out = set(self.eliminated())

        def rank(i: int) -> tuple[float, int, int]:
            p = self.players[i]
            top = max((c.value for c in p.cards if c.kind is Kind.LUXURY), default=0)
            return (score_of(p.cards), p.money, top)

        contenders = [i for i in range(self.n) if i not in out]
        if not contenders:
            return []
        best = max(rank(i) for i in contenders)
        return [i for i in contenders if rank(i) == best]

    # --- internals ---

    def _log(
        self,
        kind: str,
        player: int | None = None,
        card: Card | None = None,
        money: tuple[int, ...] = (),
    ) -> None:
        self.history.append(Event(kind, player, card, tuple(money)))

    def _next_turn(self) -> None:
        i = self.turn
        for _ in range(self.n):
            i = (i + 1) % self.n
            if self.players[i].in_auction:
                self.turn = i
                return

    def _give(self, i: int, card: Card, how: str) -> None:
        player = self.players[i]
        self._log(how, i, card)
        if card.kind is Kind.LUXURY and player.owes_luxury:
            player.owes_luxury = False
            self._log("discard", i, card)
            return
        player.cards.append(card)
        if card.kind is Kind.FAUX_PAS:
            luxuries = [c for c in player.cards if c.kind is Kind.LUXURY]
            if luxuries:
                # Discarding the lowest is always best (score, then the tie-break).
                lowest = min(luxuries)
                player.cards.remove(lowest)
                self._log("discard", i, lowest)
            else:
                player.owes_luxury = True

    def _reveal(self, starter: int) -> None:
        card = self.deck.pop()
        self._log("reveal", card=card)
        if card.red:
            self.reds += 1
        if self.reds >= RED_CARDS_TO_END:
            self.card = None
            self.over = True
            self._log("end")
            return
        self.card = card
        for p in self.players:
            p.in_auction = True
            p.bid = []
        self.turn = starter
