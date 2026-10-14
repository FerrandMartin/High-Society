from __future__ import annotations

import pytest

from high_society.bots import Bot, RandomBot, SaverBot, ValueBot, play_game
from high_society.engine import (
    DECK,
    MONEY,
    STARTING_MONEY,
    Card,
    Game,
    IllegalAction,
    Kind,
    Pass,
    Raise,
    cheapest_cards,
    score_of,
)

LUX = {v: Card(Kind.LUXURY, v) for v in range(1, 11)}
X2 = Card(Kind.PRESTIGE)
PASSE = Card(Kind.PASSE)
SCANDALE = Card(Kind.SCANDALE)
FAUX_PAS = Card(Kind.FAUX_PAS)


def order(*first: Card) -> list[Card]:
    """These cards first, in this order, then the rest of the deck (luxuries, then reds)."""
    rest = list(DECK)
    for c in first:
        rest.remove(c)
    rest.sort(key=lambda c: c.red)
    return [*first, *rest]


def test_deck_and_money() -> None:
    assert len(DECK) == 16
    assert sum(c.red for c in DECK) == 4
    assert sum(c.disgrace for c in DECK) == 3
    assert STARTING_MONEY == 106 and len(MONEY) == 11
    with pytest.raises(ValueError):
        Game(2)
    with pytest.raises(ValueError):
        Game(6)


def test_scores() -> None:
    assert score_of([LUX[3], LUX[7]]) == 10
    assert score_of([LUX[3], LUX[7], PASSE]) == 5
    assert score_of([LUX[3], LUX[7], X2, X2]) == 40
    assert score_of([LUX[3], LUX[7], X2, SCANDALE]) == 10
    assert score_of([LUX[1], PASSE]) == -4
    assert score_of([LUX[1], PASSE, X2]) == -8


def test_cheapest_cards_never_makes_change() -> None:
    assert cheapest_cards(MONEY, 5) == (1, 4)
    assert cheapest_cards(MONEY, 7) == (1, 6)
    assert cheapest_cards([10, 20], 5) == (10,)
    assert cheapest_cards([1, 2], 4) is None
    assert cheapest_cards(MONEY, 0) == ()


def test_luxury_auction_last_bidder_pays_and_others_get_their_money_back() -> None:
    game = Game(3, order=order(LUX[5]), starter=0)
    assert game.card == LUX[5]
    game.play(Raise((3,)))  # p0: 3
    game.play(Raise((4,)))  # p1: 4
    game.play(Raise((1, 6)))  # p2: 7
    game.play(Pass())  # p0 out, takes 3 back
    assert game.players[0].hand == sorted(MONEY) or sorted(game.players[0].hand) == list(MONEY)
    assert game.turn == 1
    game.play(Raise((8,)))  # p1: 12
    game.play(Pass())  # p2 out: p1 wins
    p1 = game.players[1]
    assert p1.cards == [LUX[5]]
    assert p1.money == STARTING_MONEY - 12
    assert game.players[2].money == STARTING_MONEY
    assert game.card is not None and game.turn == 1  # the winner starts the next auction


def test_cards_must_be_in_hand_and_beat_the_highest_bid() -> None:
    game = Game(3, order=order(LUX[5]), starter=0)
    game.play(Raise((10,)))
    with pytest.raises(IllegalAction):
        game.play(Raise((8,)))  # not above 10
    with pytest.raises(IllegalAction):
        game.play(Raise((30,)))  # no such card
    with pytest.raises(IllegalAction):
        game.play(Raise(()))
    assert all(
        isinstance(a, Pass) or game.players[1].bid_total + sum(a.cards) > 10
        for a in game.legal_actions()
    )


def test_disgrace_first_to_pass_takes_it_the_others_pay() -> None:
    game = Game(4, order=order(PASSE), starter=2)
    game.play(Raise((1,)))  # p2
    game.play(Raise((2,)))  # p3
    game.play(Raise((3,)))  # p0
    game.play(Pass())  # p1 passes first, with no bid: takes the card
    assert game.players[1].cards == [PASSE]
    assert game.players[1].money == STARTING_MONEY
    assert [game.players[i].money for i in (2, 3, 0)] == [105, 104, 103]
    assert game.turn == 1  # the taker starts the next auction


def test_disgrace_taker_gets_their_own_bid_back() -> None:
    game = Game(3, order=order(SCANDALE), starter=0)
    game.play(Raise((4,)))
    game.play(Raise((6,)))
    game.play(Pass())  # p2 raised nothing yet... passes: takes it
    game2 = Game(3, order=order(PASSE), starter=0)
    game2.play(Raise((4,)))
    game2.play(Raise((6,)))
    game2.play(Raise((8,)))
    game2.play(Pass())  # p0 takes it and keeps its 4
    assert game2.players[0].cards == [PASSE]
    assert game2.players[0].money == STARTING_MONEY
    assert game2.players[1].money == STARTING_MONEY - 6
    assert game2.players[2].money == STARTING_MONEY - 8
    assert game.players[2].cards == [SCANDALE]


def test_faux_pas_discards_the_lowest_luxury_or_the_next_one() -> None:
    # p0 wins the 4 and the 9, then takes the faux pas: the 4 goes.
    game = Game(3, order=order(LUX[4], LUX[9], FAUX_PAS, LUX[2], LUX[7]), starter=0)
    game.play(Raise((1,)))
    game.play(Pass())
    game.play(Pass())  # p0 wins the 4
    game.play(Raise((2,)))
    game.play(Pass())
    game.play(Pass())  # p0 wins the 9
    game.play(Pass())  # p0 passes first on the faux pas
    assert game.players[0].cards == [LUX[9], FAUX_PAS]
    # Now p0 starts on the 2; p1 wins it with nothing to discard yet.
    game.play(Pass())
    game.play(Raise((1,)))
    game.play(Pass())
    assert game.players[1].cards == [LUX[2]]

    # With no luxury, the faux pas eats the next luxury won.
    game = Game(3, order=order(FAUX_PAS, LUX[6], LUX[3]), starter=1)
    game.play(Pass())  # p1 takes the faux pas
    assert game.players[1].owes_luxury
    game.play(Raise((1,)))  # p1 starts on the 6
    game.play(Pass())
    game.play(Pass())  # p1 wins the 6: discarded
    assert game.players[1].cards == [FAUX_PAS] and not game.players[1].owes_luxury
    game.play(Raise((2,)))
    game.play(Pass())
    game.play(Pass())  # p1 wins the 3: kept
    assert LUX[3] in game.players[1].cards


def test_the_fourth_red_card_ends_the_game_unauctioned() -> None:
    game = Game(3, order=[X2, X2, SCANDALE, X2, *[c for c in DECK if not c.red]], starter=0)
    for bid in (1, 2):  # p0 wins both ×2
        game.play(Raise((bid,)))
        game.play(Pass())
        game.play(Pass())
    game.play(Pass())  # p0 takes the scandale
    assert game.over and game.card is None
    assert game.reds == 4
    assert game.history[-1].kind == "end"
    assert game.legal_actions() == []


def test_the_poorest_is_eliminated_then_the_best_score_wins() -> None:
    game = Game(3, order=order(LUX[10], LUX[1], X2, X2, SCANDALE, X2), starter=0)
    moves: list[Pass | Raise] = [
        Raise((25,)),
        Pass(),
        Pass(),  # p0 pays 25 for the 10
        Pass(),
        Raise((2,)),
        Pass(),  # p1 wins the 1 for 2
        Raise((1,)),
        Pass(),
        Pass(),  # p1 wins a ×2 for 1
        Raise((3,)),
        Pass(),
        Pass(),  # p1 wins a ×2 for 3
        Pass(),  # p1 takes the scandale; the next card is the 4th red
    ]
    for move in moves:
        game.play(move)
    assert game.over
    assert [p.money for p in game.players] == [81, 100, 106]
    assert game.scores() == [10, 2, 0]
    # p0 has the best score but the least money.
    assert game.eliminated() == [0]
    assert game.winners() == [1]


def test_ties_eliminate_everyone_tied_but_never_everyone() -> None:
    game = play_game([ValueBot(), ValueBot(), ValueBot()], seed=1)
    game.players[0].hand = [1]
    game.players[1].hand = [1]
    game.players[2].hand = [5]
    assert game.eliminated() == [0, 1]
    for p in game.players:
        p.hand = [3]
    assert game.eliminated() == []


@pytest.mark.parametrize("players", [3, 4, 5])
def test_random_games_keep_the_money_and_always_end(players: int) -> None:
    for seed in range(300):
        bots: list[Bot] = [
            RandomBot(seed),
            ValueBot(),
            SaverBot(),
            RandomBot(seed + 1),
            ValueBot(0.7),
        ]
        game = play_game(bots[:players], seed=seed)
        assert game.over
        spent = sum(sum(e.money) for e in game.history if e.kind == "discard" and e.money)
        left = sum(p.money for p in game.players)
        assert left + spent == STARTING_MONEY * players
        assert sum(c.red for p in game.players for c in p.cards) == 3
        winners = game.winners()
        assert winners and not set(winners) & set(game.eliminated())
