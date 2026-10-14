from __future__ import annotations

import numpy as np
import pytest
import torch

from high_society.bots import SaverBot, ValueBot
from high_society.encoding import (
    LEVELS,
    MAIN_FEATURES,
    N_ACTIONS,
    OPPONENT_FEATURES,
    OPPONENT_SLOTS,
    observe,
    options,
)
from high_society.engine import Game, Pass, Raise
from high_society.model import ModelConfig, PolicyValueNet
from high_society.play import NetAgent, Step, batch, play_games, reward, think
from high_society.rating import fit_ratings, tournament
from high_society.train import TrainConfig, advantages, update


@pytest.mark.parametrize("players", [3, 4, 5])
def test_observation_shapes_and_absent_seats(players: int) -> None:
    game = Game(players, seed=1)
    obs = observe(game)
    assert obs.main.shape == (MAIN_FEATURES,)
    assert obs.opponents.shape == (OPPONENT_SLOTS, OPPONENT_FEATURES)
    assert obs.present.tolist() == [1.0] * (players - 1) + [0.0] * (5 - players)
    assert not obs.opponents[players - 1 :].any()
    assert obs.legal.shape == (N_ACTIONS,)
    assert np.isfinite(obs.main).all() and np.abs(obs.main).max() <= 2


def test_actions_map_to_the_cheapest_legal_raise() -> None:
    game = Game(3, seed=2, starter=0)
    moves = options(game)
    assert moves[0] == Pass()
    assert moves[1] == Raise((1,))  # the smallest raise
    assert moves[2 + LEVELS.index(5)] in (Raise((1, 4)), Raise((2, 3)))
    assert moves[2 + LEVELS.index(70)] is not None
    game.play(Raise((25,)))
    moves = options(game)
    # Levels at or below the highest bid (25) aren't offered; "min" beats it.
    assert all(m is None for m, level in zip(moves[2:], LEVELS, strict=True) if level <= 25)
    assert isinstance(moves[1], Raise) and sum(moves[1].cards) > 25


def test_observation_hides_the_deck_order() -> None:
    a = Game(4, seed=3, starter=0)
    b = Game(4, seed=3, starter=0)
    b.deck = list(reversed(b.deck))  # same card on auction, different order below
    assert a.card == b.card
    oa, ob = observe(a), observe(b)
    assert np.array_equal(oa.main, ob.main) and np.array_equal(oa.opponents, ob.opponents)


def test_the_network_never_draws_an_illegal_action() -> None:
    torch.manual_seed(0)
    model = PolicyValueNet(ModelConfig(width=32, opponent_width=16))
    tables = [
        [NetAgent(model), ValueBot(), SaverBot(), NetAgent(model)][: 3 + i % 2] for i in range(20)
    ]
    result = play_games(tables, seed=4, record=tables[0][0])  # type: ignore[arg-type]
    assert all(g.over for g in result.games)
    assert result.steps and all(s.obs.legal[s.action] for s in result.steps)
    thought = think(model, Game(3, seed=5))
    assert abs(sum(thought.probabilities.values()) - 1) < 1e-5
    assert 0 <= thought.chance_to_win <= 1


def test_advantages_carry_the_result_back_through_a_players_decisions() -> None:
    obs = observe(Game(3, seed=6))
    steps = [
        Step(0, 1, obs, 0, 0.0, 0.5),
        Step(0, 2, obs, 0, 0.0, 0.5),
        Step(0, 1, obs, 0, 0.0, 0.6),
    ]
    adv, returns = advantages(steps, {(0, 1): 1.0, (0, 2): 0.0}, lam=1.0)
    # With λ = 1 the return of every decision is the final result.
    assert returns.tolist() == pytest.approx([1.0, 0.0, 1.0])
    assert adv.tolist() == pytest.approx([0.5, -0.5, 0.4])


def test_ppo_moves_the_policy_towards_rewarded_actions() -> None:
    torch.manual_seed(0)
    model = PolicyValueNet(ModelConfig(width=32, opponent_width=16))
    obs = observe(Game(3, seed=7, starter=0))
    good = 1  # the smallest raise
    steps = [
        Step(i, 0, obs, a, float(np.log(1 / obs.legal.sum())), 0.5)
        for i, a in enumerate([good, 0] * 32)
    ]
    rewards = {(s.game, 0): 1.0 if s.action == good else 0.0 for s in steps}
    adv, returns = advantages(steps, rewards, 0.95)
    before = torch.softmax(model(*batch([obs]))[0], -1)[0, good].item()
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(5):
        update(model, optimiser, steps, adv, returns, TrainConfig(minibatch=64))
    after = torch.softmax(model(*batch([obs]))[0], -1)[0, good].item()
    assert after > before


def test_ratings_follow_who_beats_whom() -> None:
    wins = {
        ("strong", "weak"): 90.0,
        ("weak", "strong"): 10.0,
        ("strong", "random"): 95.0,
        ("random", "strong"): 5.0,
        ("weak", "random"): 60.0,
        ("random", "weak"): 40.0,
    }
    r = fit_ratings(["random", "weak", "strong"], wins)
    assert r["random"] == pytest.approx(1000)
    assert r["strong"] > r["weak"] > r["random"]


def test_tournament_between_bots() -> None:
    standings = tournament(
        {"random": ValueBot(0.2), "value": ValueBot(), "saver": SaverBot()}, games=300, seed=8
    )
    assert {s.name for s in standings} == {"random", "value", "saver"}
    assert sum(s.games for s in standings) >= 300 * 3
    game = play_games([[ValueBot(), SaverBot(), ValueBot()]], seed=9).games[0]
    assert sum(reward(game, i) for i in range(3)) in (0.0, 1.0)
