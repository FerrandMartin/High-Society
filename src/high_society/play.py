"""Plays many games at once. All the network decisions waiting at a step go through the network
together, which is much faster than one at a time."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import torch

from .bots import Bot
from .encoding import ACTIONS, Observation, observe, options
from .engine import Action, Game
from .model import PolicyValueNet


@dataclass
class NetAgent:
    """A network playing: draws its move from the policy, or takes the likeliest one."""

    model: PolicyValueNet
    name: str = "net"
    greedy: bool = False


Agent = NetAgent | Bot


@dataclass
class Step:
    """One decision of a recorded agent, for training."""

    game: int
    seat: int
    obs: Observation
    action: int
    logp: float
    value: float


@dataclass
class Thought:
    """What a network made of a position: for the website's "watch" mode."""

    probabilities: dict[str, float]
    chance_to_win: float


@dataclass
class Result:
    games: list[Game]
    steps: list[Step] = field(default_factory=list)


def batch(observations: list[Observation]) -> tuple[torch.Tensor, ...]:
    return (
        torch.from_numpy(np.stack([o.main for o in observations])),
        torch.from_numpy(np.stack([o.opponents for o in observations])),
        torch.from_numpy(np.stack([o.present for o in observations])),
        torch.from_numpy(np.stack([o.legal for o in observations])),
    )


def think(model: PolicyValueNet, game: Game) -> Thought:
    with torch.inference_mode():
        logits, value = model(*batch([observe(game)]))
    probs = torch.softmax(logits[0], dim=-1).tolist()
    return Thought(
        {name: p for name, p in zip(ACTIONS, probs, strict=True) if p > 0},
        float(value[0]),
    )


def play_games(
    tables: list[list[Agent]],
    seed: int = 0,
    record: NetAgent | None = None,
) -> Result:
    """Plays one game per table (one agent per seat). Decisions of `record` are kept."""
    rng = random.Random(seed)
    torch_rng = torch.Generator().manual_seed(seed)
    games = [Game(len(t), seed=rng.randrange(2**31)) for t in tables]
    result = Result(games)
    while True:
        waiting: dict[int, list[int]] = defaultdict(list)
        for g, game in enumerate(games):
            if not game.over:
                waiting[id(tables[g][game.turn])].append(g)
        if not waiting:
            return result
        for indices in waiting.values():
            first = games[indices[0]]
            agent = tables[indices[0]][first.turn]
            if not isinstance(agent, NetAgent):
                for g in indices:
                    games[g].play(agent.act(games[g]))
                continue
            moves = [options(games[g]) for g in indices]
            observations = [observe(games[g], m) for g, m in zip(indices, moves, strict=True)]
            with torch.inference_mode():
                logits, values = agent.model(*batch(observations))
            if agent.greedy:
                chosen = logits.argmax(dim=-1)
            else:
                chosen = torch.multinomial(torch.softmax(logits, -1), 1, generator=torch_rng)
                chosen = chosen.squeeze(-1)
            logps = torch.log_softmax(logits, -1).gather(1, chosen.unsqueeze(1)).squeeze(1)
            for k, g in enumerate(indices):
                a = int(chosen[k])
                move: Action | None = moves[k][a]
                assert move is not None, "an illegal action was drawn"
                if agent is record:
                    result.steps.append(
                        Step(
                            g, games[g].turn, observations[k], a, float(logps[k]), float(values[k])
                        )
                    )
                games[g].play(move)


def reward(game: Game, seat: int) -> float:
    """1 for a win, shared on a tie; 0 otherwise."""
    winners = game.winners()
    return 1 / len(winners) if seat in winners else 0.0
