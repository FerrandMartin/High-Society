"""Ratings (spec §4.3): each agent plays many games against the others, and a rating like chess
Elo is fitted to who beat whom. In a game, each winner beats each other player at the table."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from .play import Agent, play_games

BASE = 1000.0  # the random bot's rating


@dataclass
class Standing:
    name: str
    rating: float
    win_rate: float  # share of its games won
    games: int


def fit_ratings(
    names: Sequence[str], wins: dict[tuple[str, str], float], iterations: int = 200
) -> dict[str, float]:
    """Bradley-Terry strengths (the classic fixed-point method), as Elo points."""
    strength = dict.fromkeys(names, 1.0)
    total_wins: dict[str, float] = defaultdict(float)
    for (winner, _), w in wins.items():
        total_wins[winner] += w
    pairs: dict[frozenset[str], float] = defaultdict(float)
    for (a, b), w in wins.items():
        pairs[frozenset((a, b))] += w
    for _ in range(iterations):
        new = {}
        for name in names:
            denominator = sum(
                n / (strength[name] + strength[other])
                for pair, n in pairs.items()
                if name in pair
                for other in pair
                if other != name
            )
            # A small prior keeps agents that never (or always) win finite.
            new[name] = (total_wins[name] + 0.5) / (denominator + 1 / strength[name])
        mean = math.exp(sum(math.log(v) for v in new.values()) / len(new))
        strength = {k: v / mean for k, v in new.items()}
    anchor = strength.get("random", 1.0)
    return {k: BASE + 400 * math.log10(v / anchor) for k, v in strength.items()}


def tournament(
    agents: dict[str, Agent], games: int, players: Sequence[int] = (3, 4, 5), seed: int = 0
) -> list[Standing]:
    """Random tables of 3 to 5 agents drawn from the list (repeats allowed if too few)."""
    rng = random.Random(seed)
    names = list(agents)
    tables_names = []
    for _ in range(games):
        n = rng.choice(list(players))
        seats = rng.sample(names, n) if len(names) >= n else [rng.choice(names) for _ in range(n)]
        tables_names.append(seats)
    result = play_games([[agents[s] for s in t] for t in tables_names], seed=seed)

    wins: dict[tuple[str, str], float] = defaultdict(float)
    won: dict[str, float] = defaultdict(float)
    played: dict[str, int] = defaultdict(int)
    for game, seats in zip(result.games, tables_names, strict=True):
        winners = game.winners()
        for i, name in enumerate(seats):
            played[name] += 1
            if i in winners:
                won[name] += 1 / len(winners)
                for j, other in enumerate(seats):
                    if j not in winners and other != name:
                        wins[(name, other)] += 1 / len(winners)
    ratings = fit_ratings(names, wins)
    return sorted(
        (Standing(n, ratings[n], won[n] / max(played[n], 1), played[n]) for n in names),
        key=lambda s: -s.rating,
    )
