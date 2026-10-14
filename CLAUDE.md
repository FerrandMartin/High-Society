# High Society AI

A private website to play High Society against self-taught neural-network AIs, or watch them
play. **`docs/SPEC.md` is the source of truth**: read the relevant section first, and keep it
in sync when a decision changes (bump its version, add a changelog row).

## Rules of the code

- **The rules exist once:** `src/high_society/engine.py`. Bots, training and the web server
  drive games through `Game.legal_actions()` / `Game.play()`; nothing else re-implements a
  rule.
- The engine is pure and deterministic for a given seed (and `order`/`starter` in tests).
- Every rule has a test in `tests/test_engine.py`; random-game tests check invariants (money
  is conserved, games end, winners aren't eliminated).

## Commands

```bash
uv sync
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest -q
uv run python -m high_society.simulate --bots value,saver,random --games 2000
```

Run all checks before every push; CI runs the same. Python 3.11+, mypy strict, line length 100.
