# 1. The training data: "history book"

This is the file your model learns from. It is a dataset of historical data:
**one line per coin per day**, going back to 2020. Here are 25 real lines from it:

| id | date | feature_1_lag0 | feature_1_lag15 | feature_2_lag0 | … | target_10d | target_30d |
|---|---|---|---|---|---|---|---|
| BTC | 2026-07-28 | 0.499 | 0.393 | 0.536 | … | 0.441 | 0.659 |
| VVV | 2026-07-28 | 0.566 | 0.363 | 0.575 | … | 0.047 | 0.759 |
| W | 2026-07-28 | 0.502 | 0.518 | 0.481 | … | 0.524 | 0.318 |
| WCT | 2026-07-28 | 0.408 | 0.598 | 0.445 | … | 0.571 | 0.235 |
| WIF | 2026-07-28 | 0.496 | 0.380 | 0.493 | … | 0.271 | 0.876 |
| WLD | 2026-07-28 | 0.402 | 0.398 | 0.412 | … | 0.253 | 0.629 |
| WLFI | 2026-07-28 | 0.466 | 0.631 | 0.479 | … | 0.065 | 0.182 |
| XAI | 2026-07-28 | 0.396 | 0.477 | 0.471 | … | 0.971 | 0.688 |
| XLM | 2026-07-28 | 0.477 | 0.452 | 0.548 | … | 0.147 | 0.194 |
| XMR | 2026-07-28 | 0.569 | 0.582 | 0.595 | … | 0.718 | 0.818 |
| XPL | 2026-07-28 | 0.493 | 0.456 | 0.445 | … | 0.265 | 0.300 |
| XRP | 2026-07-28 | 0.586 | 0.530 | 0.570 | … | 0.176 | 0.771 |
| YGG | 2026-07-28 | 0.489 | 0.471 | 0.513 | … | 0.847 | 0.794 |
| ZEC | 2026-07-28 | 0.284 | 0.626 | 0.503 | … | 0.794 | 0.959 |
| ZEN | 2026-07-28 | 0.537 | 0.509 | 0.552 | … | 0.741 | 0.782 |
| ZETA | 2026-07-28 | 0.362 | 0.499 | 0.444 | … | 0.606 | 0.571 |
| ZK | 2026-07-28 | 0.448 | 0.494 | 0.393 | … | 0.382 | 0.171 |
| ZORA | 2026-07-28 | 0.485 | 0.496 | 0.408 | … | 0.094 | 0.229 |
| ZRO | 2026-07-28 | 0.539 | 0.567 | 0.463 | … | 0.824 | 0.888 |
| kBONK | 2026-07-28 | 0.602 | 0.309 | 0.424 | … | 0.029 | 0.153 |
| kFLOKI | 2026-07-28 | 0.590 | 0.450 | 0.492 | … | 0.594 | 0.706 |
| kLUNC | 2026-07-28 | 0.383 | 0.491 | 0.431 | … | 0.294 | 0.212 |
| kNEIRO | 2026-07-28 | 0.549 | 0.436 | 0.521 | … | 0.924 | 0.935 |
| kPEPE | 2026-07-28 | 0.551 | 0.550 | 0.566 | … | 0.641 | 0.835 |
| kSHIB | 2026-07-28 | 0.572 | 0.491 | 0.515 | … | 0.341 | 0.382 |

## How to read one line

Take the first row. On that day, the coin **BTC** had a bunch of measurements
taken (the `feature_…` columns, think RSI, MACD, OHLCV....), and then, looking 10 and 30 days into the future,
we know how it actually did (the `target_…` columns, how they actually ranked relative to one another).

- **Every number is a score between 0 and 1**, and it always means the same thing:
  *"compared to all the other coins that day, where did this one stand?"*
  0.900 = it beat 90% of its peers. 0.100 = 90% of its peers beat it.
  (The lowest score isn't quite 0. with n coins the last place gets 1/n.)
- **`feature_1_lag0` vs `feature_1_lag15`** is the same measurement, taken today vs
  15 days ago. Every feature comes in 4 flavours: today, 5, 10, and 15 days back.
  There are 45 measurements × 4 = **180 feature columns**, only 3 are shown here
  because they all look the same. Nobody knows what the measurements actually are;
  that mystery is part of the game.
- **`target_10d` / `target_30d`** are the answers: the coin's *rank* among its peers
  over the next 10 / 30 days. 1.000 = the best performer of that day. These are what
  your model tries to predict.

Important rule when you practise: **always test your model on dates it hasn't seen**
(train on the past, check on the most recent days).