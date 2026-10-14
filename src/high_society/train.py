"""Self-play training with PPO (spec §4.2).

    uv run python -m high_society.train --run main --hours 2

The games are played on all the processor's cores (--workers) and the network learns from
them in between (--threads).
Each iteration plays a batch of games (3 to 5 players at random) where the learner sits next
to copies of itself, earlier generations and the simple bots; then the network is nudged
towards the moves that turned out better than it expected. A generation is saved regularly in
runs/<run>/, with its results against the bots; a stopped run resumes where it was.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .bots import RandomBot, SaverBot, ValueBot
from .model import ModelConfig, PolicyValueNet
from .play import Agent, NetAgent, Step, batch, play_games, reward
from .workers import Players


@dataclass
class TrainConfig:
    games_per_iteration: int = 256
    epochs: int = 4
    minibatch: int = 1024
    learning_rate: float = 3e-4
    clip: float = 0.2
    value_weight: float = 0.5
    entropy_weight: float = 0.01
    gae_lambda: float = 0.95
    self_share: float = 0.5  # other seats: the learner itself, else the pool
    bot_share: float = 0.3  # of the pool seats: simple bots, else earlier generations
    generation_every: int = 20  # iterations
    pool_size: int = 10  # the latest generations kept as opponents
    eval_games: int = 300


def advantages(
    steps: list[Step], game_rewards: dict[tuple[int, int], float], lam: float
) -> tuple[np.ndarray, np.ndarray]:
    """How much better than expected each decision turned out (GAE, no discount: only the end
    of the game counts)."""
    adv = np.zeros(len(steps), dtype=np.float32)
    last: dict[tuple[int, int], tuple[float, float]] = {}  # next value, next advantage
    for i in range(len(steps) - 1, -1, -1):
        s = steps[i]
        key = (s.game, s.seat)
        if key in last:
            next_value, next_adv = last[key]
            delta = next_value - s.value
        else:  # the player's last decision: what follows is the result
            next_adv = 0.0
            delta = game_rewards[key] - s.value
        adv[i] = delta + lam * next_adv
        last[key] = (s.value, float(adv[i]))
    values = np.array([s.value for s in steps], dtype=np.float32)
    return adv, adv + values


def update(
    model: PolicyValueNet,
    optimiser: torch.optim.Optimizer,
    steps: list[Step],
    adv: np.ndarray,
    returns: np.ndarray,
    cfg: TrainConfig,
) -> dict[str, float]:
    main, opponents, present, legal = batch([s.obs for s in steps])
    actions = torch.tensor([s.action for s in steps])
    old_logp = torch.tensor([s.logp for s in steps])
    adv_t = torch.from_numpy(adv)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
    returns_t = torch.from_numpy(returns)
    stats: dict[str, float] = {}
    model.train()
    for _ in range(cfg.epochs):
        order = torch.randperm(len(steps))
        for start in range(0, len(steps), cfg.minibatch):
            idx = order[start : start + cfg.minibatch]
            logits, value = model(main[idx], opponents[idx], present[idx], legal[idx])
            log_probs = torch.log_softmax(logits, -1)
            logp = log_probs.gather(1, actions[idx].unsqueeze(1)).squeeze(1)
            ratio = torch.exp(logp - old_logp[idx])
            # PPO: follow the advantage, but never move the policy far in one update.
            policy_loss = -torch.min(
                ratio * adv_t[idx],
                ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv_t[idx],
            ).mean()
            value_loss = ((value - returns_t[idx]) ** 2).mean()
            probs = torch.exp(log_probs)
            entropy = -(probs * log_probs.masked_fill(~legal[idx], 0)).sum(-1).mean()
            loss = policy_loss + cfg.value_weight * value_loss - cfg.entropy_weight * entropy
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimiser.step()
            stats = {
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
                "entropy": entropy.item(),
            }
    model.eval()
    return stats


def random_bot(rng: random.Random) -> Agent:
    choices: list[Agent] = [RandomBot(rng.randrange(1 << 30)), ValueBot(), SaverBot()]
    return rng.choice(choices)


def against_bots(model: PolicyValueNet, games: int, seed: int) -> dict[str, float]:
    """Win rate of the network (greedy) against the simple bots, per table size. An even
    player would win 1/n of the games."""
    rng = random.Random(seed)
    net = NetAgent(model, greedy=True)
    out = {}
    for n in (3, 4, 5):
        tables: list[list[Agent]] = []
        seats = []
        for _ in range(games // 3):
            others: list[Agent] = [random_bot(rng) for _ in range(n - 1)]
            seat = rng.randrange(n)
            tables.append([*others[:seat], net, *others[seat:]])
            seats.append(seat)
        result = play_games(tables, seed=rng.randrange(1 << 30))
        out[f"win_{n}p"] = sum(
            reward(g, s) for g, s in zip(result.games, seats, strict=True)
        ) / len(seats)
    return out


def train(
    run: str,
    hours: float,
    cfg: TrainConfig,
    model_cfg: ModelConfig,
    seed: int,
    threads: int,
    iterations: int | None = None,
    workers: int = 1,
) -> Path:
    torch.set_num_threads(threads)
    folder = Path("runs") / run
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder / "latest.pt"
    model = PolicyValueNet(model_cfg)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    iteration, games_played = 0, 0
    if checkpoint.exists():
        model, data = PolicyValueNet.load(str(checkpoint))
        optimiser = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
        optimiser.load_state_dict(data["optimiser"])  # type: ignore[arg-type]
        iteration = int(data["iteration"])  # type: ignore[call-overload]
        games_played = int(data["games"])  # type: ignore[call-overload]
        print(f"Resuming {run} at iteration {iteration}")
    generations = sorted(folder.glob("gen_*.pt"))

    rng = random.Random(seed + iteration)
    deadline = time.time() + hours * 3600
    metrics = folder / "metrics.jsonl"
    players = Players(workers)
    try:
        while time.time() < deadline and (iterations is None or iteration < iterations):
            iteration, games_played = _iteration(
                folder,
                model,
                optimiser,
                cfg,
                rng,
                players,
                generations,
                iteration,
                games_played,
                metrics,
            )
    finally:
        players.close()
    return folder


def _seat(cfg: TrainConfig, rng: random.Random, generations: list[Path]) -> str:
    """Who sits next to the learner: itself, an earlier generation or a bot."""
    if rng.random() < cfg.self_share:
        return "learner"
    recent = generations[-cfg.pool_size :]
    if recent and rng.random() >= cfg.bot_share:
        return f"gen:{rng.choice(recent)}"
    return rng.choice([f"random:{rng.randrange(1 << 30)}", "value", "saver"])


def _iteration(
    folder: Path,
    model: PolicyValueNet,
    optimiser: torch.optim.Optimizer,
    cfg: TrainConfig,
    rng: random.Random,
    players: Players,
    generations: list[Path],
    iteration: int,
    games_played: int,
    metrics: Path,
) -> tuple[int, int]:
    started = time.time()
    tables: list[list[str]] = []
    for _ in range(cfg.games_per_iteration):
        n = rng.choice((3, 4, 5))
        table = [_seat(cfg, rng, generations) for _ in range(n - 1)]
        table.insert(rng.randrange(n), "learner")
        tables.append(table)
    steps, rewards = players.play(tables, rng.randrange(1 << 30), model)
    adv, returns = advantages(steps, rewards, cfg.gae_lambda)
    stats = update(model, optimiser, steps, adv, returns, cfg)
    iteration += 1
    games_played += cfg.games_per_iteration
    line: dict[str, object] = {
        "iteration": iteration,
        "games": games_played,
        "steps": len(steps),
        "seconds": round(time.time() - started, 2),
        **stats,
    }
    if iteration % cfg.generation_every == 0:
        number = len(generations) + 1
        path = folder / f"gen_{number:04d}.pt"
        model.save(str(path), iteration=iteration, games=games_played)
        generations.append(path)
        line.update(generation=number, **against_bots(model, cfg.eval_games, iteration))
        print(json.dumps(line), flush=True)
    model.save(
        str(folder / "latest.pt"),
        optimiser=optimiser.state_dict(),
        iteration=iteration,
        games=games_played,
        train_config=asdict(cfg),
    )
    with metrics.open("a") as f:
        f.write(json.dumps(line) + "\n")
    return iteration, games_played


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", default="main")
    parser.add_argument("--hours", type=float, default=1.0)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    cores = os.cpu_count() or 2
    parser.add_argument(
        "--workers", type=int, default=max(1, cores - 1), help="processes playing the games"
    )
    parser.add_argument(
        "--threads", type=int, default=min(cores, 8), help="threads for the learning step"
    )
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--games-per-iteration", type=int, default=256)
    parser.add_argument("--generation-every", type=int, default=20)
    args = parser.parse_args(argv)
    cfg = TrainConfig(
        games_per_iteration=args.games_per_iteration, generation_every=args.generation_every
    )
    train(
        args.run,
        args.hours,
        cfg,
        ModelConfig(width=args.width, layers=args.layers),
        args.seed,
        args.threads,
        args.iterations,
        args.workers,
    )


if __name__ == "__main__":
    main()
