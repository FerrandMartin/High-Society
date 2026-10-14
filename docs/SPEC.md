# High Society AI: Specification

|             |                                     |
| ----------- | ----------------------------------- |
| **Version** | 0.4                                 |
| **Date**    | 2026-10-14                          |
| **Status**  | Phase 1 done                        |

A private website to play Reiner Knizia's **High Society** against neural-network AIs, or to
watch them play each other. The AIs learn by playing against each other (self-play). The site
runs on the owner's VPS and is reachable only through Tailscale. French and English.

---

## 1. Goals

- **Play:** one human against 2 to 4 AIs, chosen from the trained generations and simple bots.
- **Watch:** AI-only games, step by step or at speed, showing what each AI "thinks".
- **Learn:** AIs trained by self-play, getting stronger generation after generation, with the
  progress measured and shown.
- **Fun first:** small, fast, understandable. No accounts, no public access.

Out of scope for now: several humans in one game, mobile app, public hosting.

## 2. The game (rules implemented)

**Players:** 3 to 5.

**Money:** each player has the same 11 money cards: 1, 2, 3, 4, 6, 8, 10, 12, 15, 20, 25
(thousands). Money spent is lost; there is no change.

**Status deck (16 cards, shuffled, face down):**

| Kind     | Cards                                   | Effect at the end                    |
| -------- | --------------------------------------- | ------------------------------------ |
| Luxury   | 10 cards worth 1 to 10                  | + their value                        |
| Prestige | 3 × "×2" (red border)                   | score doubled, for each one          |
| Disgrace | Passé                                   | −5                                   |
| Disgrace | Scandale (red border)                   | score halved                         |
| Disgrace | Faux pas                                | discard one of your luxury cards     |

**A round:** the top card is revealed and auctioned. Bidding goes clockwise from the starting
player. On your turn you either **raise** (add money cards in front of you so your total bid is
higher than the highest bid) or **pass**. Cards bid stay in front of you until the auction ends.

- **Luxury and prestige cards:** passing takes your bid back into your hand and you are out of
  this auction. The last player left wins the card and **discards the money bid**.
- **Disgrace cards:** the **first player to pass** takes the card and takes their bid back;
  everyone else **discards their bids**. (Players bid to avoid the card.)
- **Faux pas:** the player discards one luxury card of their choice; with none, they discard
  the next luxury card they win. The engine always discards the lowest one: it is never worse
  (for the score or the tie-break), so it isn't a decision.
- A player who can't raise must pass.
- The winner of an auction (or the taker of a disgrace card) starts the next one. The first
  auction starts with a random player.

**End:** the game ends **immediately when the 4th red-bordered card is revealed**; that card is
not auctioned.

**Winner:**

1. The player(s) with the **least money left in hand** are eliminated, all of them on a tie.
   If every player has the same money, everyone is eliminated and nobody wins.
2. Among the others, score = (sum of luxury values − 5 for Passé) × 2 per prestige ÷ 2 for
   Scandale. Highest score wins.
3. Ties: more money left, then the highest single luxury card.

Hidden information: the order of the deck. Money spent is public, so everyone can work out
everyone's hand.

## 3. Architecture

```
 Phone / laptop (Tailscale) ──HTTPS──► VPS
                                        ├─ web server (FastAPI): pages, game API, websocket
                                        ├─ game engine (Python, pure, tested)
                                        ├─ AIs: neural networks (PyTorch) + simple bots
                                        ├─ trainer: self-play, runs in the background
                                        └─ storage: SQLite (games, replays, ratings), model files
```

- **One engine, in Python**, used by the trainer, the bots and the server, so the rules exist
  only once. The browser shows the state and sends moves; it holds no rules.
- **Web UI:** a small single-page app (TypeScript, Vite) talking to the server over a websocket.
  FR/EN.
- **Access:** the VPS joins the owner's tailnet; the site listens only on the Tailscale
  interface. No public port, no accounts.

## 4. AIs

### 4.1 Simple bots (baselines)

- **Random:** random legal move.
- **Value bidder:** raises while the bid is below a target based on the card's value and the
  money left; for disgrace cards, passes early when cheap.
- **Saver:** like the value bidder but never lets its money drop below the poorest opponent's.

They give the networks opponents from the start and a fixed yardstick.

### 4.2 Neural network

**Input (works for 3 to 5 players):**

- the card up for auction, the current bids, who is still in, whose turn;
- the AI's own hand, money spent, cards won, current score;
- **4 opponent slots**, seated relative to the AI (next, next but one…), each with money left,
  money bid now, cards won, score, still in the auction; empty slots marked absent;
- the number of players, the cards still in the deck, the red cards already revealed.

**Actions (17):** pass; the smallest possible raise; or raise the bid to at least 2, 3, 4,
5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50 or 70. The engine plays the cheapest set of money
cards that reaches it (then the fewest cards). Impossible actions are masked. (Choosing any
subset of 11 cards directly would be 2,048 actions; this keeps it small.)

**Layers:** each opponent row goes through the same small network (2 layers of 64, ReLU);
their maximum and mean are joined to the rest of the inputs; a trunk of 2 layers of 256 (with
LayerNorm and ReLU) feeds a policy head (17 logits) and a value head (a sigmoid: the chance to
win). About 125,000 weights. Inputs are scaled to roughly 0–1 (money ÷ 106).

**Output:** a policy (how likely each action is) and a value (estimated chance to win).

**Training:** reinforcement learning by self-play (PPO).

- Each game picks 3, 4 or 5 players at random. Seats are filled with the current network,
  **earlier generations** and simple bots, so it doesn't overfit to itself.
- Reward: +1 for a win, 0 otherwise (shared on ties); no shaping.
- Each iteration: 256 games; advantages with GAE (λ 0.95, no discount); 4 epochs of PPO
  (clip 0.2, value weight 0.5, entropy bonus 0.01, Adam 3e-4).
- Seats around the learner: itself (50%), else earlier generations or the bots.
- A new **generation** is saved every 20 iterations, with its win rate against the bots.
- `python -m high_society.train --run <name> --hours <h>`; a stopped run resumes.

### 4.3 Rating

Each generation and bot gets an **Elo-style rating** (Bradley-Terry, the random bot at 1000)
from a tournament: in each game, each winner beats each other player at the table.
`python -m high_society.evaluate random value saver runs/<run>/gen_0010.pt`. The website
shows the curve over generations.

## 5. Website

- **Play:** choose 3 to 5 players and the opponents (a generation or a bot); play with buttons
  for each money card and "Pass". The table shows bids, cards won, money spent, scores.
- **Watch:** pick the AIs; play step by step or at speed. For each AI: its estimated chance to
  win and its top 3 considered actions with their probabilities.
- **Progress:** rating curve per generation, win rates against the bots, training status.
- **Replays:** past games (yours and the AIs'), step by step.
- Light and dark themes, mobile-first.

## 6. Phases

1. **Engine and bots:** the rules with exhaustive tests, simple bots, a command to simulate
   thousands of games and print statistics.
2. **Training:** state encoding, network, PPO self-play, generations, ratings.
3. **Website:** play, watch, progress, replays.
4. **Deployment:** Docker on the VPS, Tailscale only, training as a background service.

Each phase ends with tests passing in CI.

## 7. Changelog

| Version | Date       | Change        |
| ------- | ---------- | ------------- |
| 0.1     | 2026-10-14 | First draft.  |
| 0.2     | 2026-10-14 | Phase 1: faux pas discards the lowest luxury; nobody is eliminated when everyone is equally poor. |
| 0.3     | 2026-10-14 | Everyone equally poor: everyone is eliminated, nobody wins. |
| 0.4     | 2026-10-14 | Phase 2: the network, its actions and layers, PPO settings, ratings. |
