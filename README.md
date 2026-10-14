# High Society AI

Play Reiner Knizia's **High Society** against neural-network AIs that learn by playing each
other, or watch them play. A private website on your own server, reachable only through
Tailscale, in French and English.

- **Specification:** [docs/SPEC.md](docs/SPEC.md)
- **Train the AIs on your PC:** [docs/TRAINING.md](docs/TRAINING.md)

## Status

- **Phase 1 (engine and bots), done:** the rules, simple bots and a simulation command.
- **Phase 2 (training), done:** the network, self-play training with PPO, generations and
  ratings.
- Phase 3 (website), phase 4 (deployment): to do.

## Development

Requirements: Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest -q
uv run python -m high_society.simulate --bots value,saver,random --games 2000
uv run python -m high_society.train --run main --hours 2          # generations in runs/main/
uv run python -m high_society.evaluate random value saver runs/main/gen_0010.pt
uv run python -m high_society.report --run main --tournament 4
```
