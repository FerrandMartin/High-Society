# Training the AIs on your PC

Training plays hundreds of thousands of games; it runs on the processor (the graphics card
doesn't help: the network is tiny and the time goes into playing the games). Every core is
used: on an 8-core processor, about 15 processes play while the network learns in between.

## 1. Install (once)

**Windows**, in PowerShell:

```powershell
winget install --id Git.Git -e
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close and reopen PowerShell, then:

```powershell
git clone https://github.com/FerrandMartin/High-Society.git
cd High-Society
uv sync
```

The clone opens a browser window to sign in to GitHub (the repository is private). `uv sync`
downloads Python and PyTorch (about 1 GB).

**Linux or macOS:** the same, with `curl -LsSf https://astral.sh/uv/install.sh | sh` to install
uv.

## 2. Check that it works (2 minutes)

```powershell
uv run python -m high_society.train --run test --iterations 4 --generation-every 2
```

It prints two lines with `"generation": 1` and `2`. Delete `runs\test` afterwards.

## 3. Train

```powershell
uv run python -m high_society.train --run main --hours 10
```

- A generation is saved every 20 iterations in `runs\main\`, with its win rate against the
  simple bots printed on one line.
- **Stop at any time with Ctrl+C**; the same command resumes where it stopped.
- For a night of training, make sure the PC doesn't go to sleep (Windows: Settings → System →
  Power → Sleep: Never, while plugged in).
- The PC stays usable but slower; `--workers 8` leaves half the cores free.

## 4. See the progress

```powershell
uv run python -m high_society.report --run main
uv run python -m high_society.report --run main --tournament 4
```

The first shows each generation's win rate against the bots; the second also rates the 4
latest generations against the bots in a tournament (a few minutes). Paste the output in a
conversation with Claude to analyse it and tune the training.

## 5. Update the code

```powershell
git pull
uv sync
```

A run started with an older version resumes with the new code, unless the network's design
changed (then start a new run, e.g. `--run main2`).
