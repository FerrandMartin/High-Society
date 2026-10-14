"""Plays the training games on all processor cores.

The trainer describes each table as seat names ("learner", "value", "gen:<file>"…) and the
games are shared between worker processes; each one rebuilds the agents (keeping the earlier
generations it already loaded), plays its share and sends back the learner's decisions.
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Any

import torch

from .bots import RandomBot, SaverBot, ValueBot
from .model import ModelConfig, PolicyValueNet
from .play import Agent, NetAgent, Step, play_games, reward

Rewards = dict[tuple[int, int], float]
Job = tuple[list[list[str]], int, dict[str, Any], dict[str, torch.Tensor]]

_generations: dict[str, PolicyValueNet] = {}
_learner: PolicyValueNet | None = None


def _init_worker() -> None:
    # One thread each: the cores are shared between the workers.
    torch.set_num_threads(1)


def _agent(spec: str, learner: NetAgent) -> Agent:
    if spec == "learner":
        return learner
    if spec == "value":
        return ValueBot()
    if spec == "saver":
        return SaverBot()
    if spec.startswith("random:"):
        return RandomBot(int(spec.split(":", 1)[1]))
    if spec.startswith("gen:"):
        path = spec.split(":", 1)[1]
        if path not in _generations:
            _generations[path] = PolicyValueNet.load(path)[0]
        return NetAgent(_generations[path], name="past")
    raise ValueError(f"Unknown seat {spec!r}")


def play_job(job: Job) -> tuple[list[Step], Rewards]:
    """Plays some tables and returns the learner's decisions and its results."""
    global _learner
    tables, seed, config, state = job
    if _learner is None or _learner.config != ModelConfig(**config):
        _learner = PolicyValueNet(ModelConfig(**config))
    _learner.load_state_dict(state)
    _learner.eval()
    learner = NetAgent(_learner, name="learner")
    result = play_games([[_agent(s, learner) for s in t] for t in tables], seed, record=learner)
    rewards = {(s.game, s.seat): reward(result.games[s.game], s.seat) for s in result.steps}
    return result.steps, rewards


class Players:
    """A pool of worker processes (or none, for one worker)."""

    def __init__(self, workers: int) -> None:
        self.workers = max(1, workers)
        # "spawn" works the same on Linux, Windows and macOS.
        self.pool = (
            mp.get_context("spawn").Pool(self.workers, initializer=_init_worker)
            if self.workers > 1
            else None
        )

    def play(
        self, tables: list[list[str]], seed: int, model: PolicyValueNet
    ) -> tuple[list[Step], Rewards]:
        from dataclasses import asdict

        state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        config = asdict(model.config)
        size = -(-len(tables) // self.workers)
        chunks = [tables[i : i + size] for i in range(0, len(tables), size)]
        jobs: list[Job] = [(c, seed + k, config, state) for k, c in enumerate(chunks)]
        results = self.pool.map(play_job, jobs) if self.pool else [play_job(j) for j in jobs]
        steps: list[Step] = []
        rewards: Rewards = {}
        offset = 0
        for chunk, (chunk_steps, chunk_rewards) in zip(chunks, results, strict=True):
            for s in chunk_steps:
                s.game += offset
                steps.append(s)
            rewards.update({(g + offset, seat): r for (g, seat), r in chunk_rewards.items()})
            offset += len(chunk)
        return steps, rewards

    def close(self) -> None:
        if self.pool:
            self.pool.close()
            self.pool.join()
