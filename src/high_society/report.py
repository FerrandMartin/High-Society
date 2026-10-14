"""How a training run is going: its generations' results against the bots, and optionally a
tournament between the latest generations and the bots.

    uv run python -m high_society.report --run main
    uv run python -m high_society.report --run main --tournament 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .evaluate import load_agent
from .rating import tournament


def generations_table(folder: Path) -> list[str]:
    lines = [
        f"{'gen':>4}{'games':>10}{'3 players':>11}{'4 players':>11}{'5 players':>11}{'variety':>9}"
    ]
    metrics = folder / "metrics.jsonl"
    if not metrics.exists():
        return [f"No training results in {folder}"]
    for raw in metrics.read_text().splitlines():
        d = json.loads(raw)
        if "generation" in d:
            lines.append(
                f"{d['generation']:>4}{d['games']:>10,}{d['win_3p']:>11.0%}{d['win_4p']:>11.0%}"
                f"{d['win_5p']:>11.0%}{d['entropy']:>9.2f}"
            )
    lines.append("Win rate alone against the bots; an even player wins 33% / 25% / 20%.")
    return lines


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", default="main")
    parser.add_argument(
        "--tournament",
        type=int,
        default=0,
        help="also rate this many of the latest generations against the bots",
    )
    parser.add_argument("--games", type=int, default=3000)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    folder = Path("runs") / args.run
    print("\n".join(generations_table(folder)))
    if args.tournament:
        torch.set_num_threads(args.threads)
        latest = sorted(folder.glob("gen_*.pt"))[-args.tournament :]
        agents = dict(load_agent(s) for s in ["random", "value", "saver", *map(str, latest)])
        print(f"\n{'agent':<16}{'rating':>8}{'win rate':>10}{'games':>8}")
        for s in tournament(agents, args.games):
            print(f"{s.name:<16}{s.rating:>8.0f}{s.win_rate:>10.1%}{s.games:>8}")


if __name__ == "__main__":
    main()
