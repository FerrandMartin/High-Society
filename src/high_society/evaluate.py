"""Ratings for bots and saved generations, from a tournament between them.

    uv run python -m high_society.evaluate random value saver runs/main/gen_0010.pt --games 3000

A generation's name is its run and number (main/10). Networks play their likeliest move.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .bots import BOTS
from .model import PolicyValueNet
from .play import Agent, NetAgent
from .rating import tournament


def load_agent(spec: str) -> tuple[str, Agent]:
    if spec in BOTS:
        return spec, BOTS[spec]()
    path = Path(spec)
    model, _ = PolicyValueNet.load(str(path))
    name = (
        f"{path.parent.name}/{int(path.stem.split('_')[-1])}"
        if path.stem.startswith("gen_")
        else path.stem
    )
    return name, NetAgent(model, name=name, greedy=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("agents", nargs="+", help="bot names or generation files")
    parser.add_argument("--games", type=int, default=3000)
    parser.add_argument("--players", default="3,4,5")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    agents = dict(load_agent(s) for s in args.agents)
    players = [int(n) for n in args.players.split(",")]
    print(f"{'agent':<16}{'rating':>8}{'win rate':>10}{'games':>8}")
    for s in tournament(agents, args.games, players, args.seed):
        print(f"{s.name:<16}{s.rating:>8.0f}{s.win_rate:>10.1%}{s.games:>8}")


if __name__ == "__main__":
    main()
