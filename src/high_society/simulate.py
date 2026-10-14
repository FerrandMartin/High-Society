"""Plays many games between bots and prints how they fare.

uv run python -m high_society.simulate --bots value,saver,random,value --games 10000
"""

from __future__ import annotations

import argparse
from collections import Counter

from .bots import make_bot, play_game


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bots", default="value,saver,random", help="one bot per seat")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    names = args.bots.split(",")
    wins: dict[str, float] = dict.fromkeys(names, 0.0)
    eliminated: Counter[str] = Counter()
    seats = Counter(names)
    lengths = []
    for g in range(args.games):
        # Seats rotate so no bot always starts in the same place.
        order = names[g % len(names) :] + names[: g % len(names)]
        game = play_game([make_bot(n) for n in order], seed=args.seed + g)
        winners = game.winners()
        for i in winners:
            wins[order[i]] += 1 / len(winners)
        for i in game.eliminated():
            eliminated[order[i]] += 1
        lengths.append(sum(e.kind in ("raise", "pass") for e in game.history))

    print(f"{args.games} games, {len(names)} players, {sum(lengths) / len(lengths):.1f} moves")
    print(f"{'bot':<10}{'win rate':>10}{'eliminated':>12}")
    for name in sorted(seats):
        per_seat = args.games * seats[name]
        print(f"{name:<10}{wins[name] / per_seat:>10.1%}{eliminated[name] / per_seat:>12.1%}")


if __name__ == "__main__":
    main()
