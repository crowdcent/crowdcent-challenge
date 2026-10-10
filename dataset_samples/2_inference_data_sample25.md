# 2. The inference data — "today's quiz"

Every day around 14:00 UTC (about 10 a.m. EST), CrowdCent posts a fresh version of this file:
the same ~170 coins, the same 180 measurements as the training data.

| id | date | feature_1_lag0 | feature_1_lag15 | feature_2_lag0 | … |
|---|---|---|---|---|---|
| 0G | 2026-09-01 | 0.559 | 0.588 | 0.496 | … |
| 2Z | 2026-09-01 | 0.474 | 0.398 | 0.480 | … |
| AAVE | 2026-09-01 | 0.519 | 0.519 | 0.556 | … |
| ACE | 2026-09-01 | 0.291 | 0.644 | 0.491 | … |
| ADA | 2026-09-01 | 0.381 | 0.285 | 0.448 | … |
| AERO | 2026-09-01 | 0.473 | 0.444 | 0.478 | … |
| AIXBT | 2026-09-01 | 0.503 | 0.471 | 0.455 | … |
| ALGO | 2026-09-01 | 0.498 | 0.378 | 0.485 | … |
| ALT | 2026-09-01 | 0.474 | 0.606 | 0.465 | … |
| ANIME | 2026-09-01 | 0.491 | 0.513 | 0.507 | … |
| APE | 2026-09-01 | 0.519 | 0.439 | 0.491 | … |
| APEX | 2026-09-01 | 0.470 | 0.399 | 0.494 | … |
| APT | 2026-09-01 | 0.395 | 0.401 | 0.408 | … |
| AR | 2026-09-01 | 0.541 | 0.586 | 0.536 | … |
| ARB | 2026-09-01 | 0.443 | 0.518 | 0.516 | … |
| ASTER | 2026-09-01 | 0.653 | 0.537 | 0.545 | … |
| ATOM | 2026-09-01 | 0.472 | 0.541 | 0.451 | … |
| AVAX | 2026-09-01 | 0.534 | 0.538 | 0.538 | … |
| AVNT | 2026-09-01 | 0.584 | 0.462 | 0.469 | … |
| AXS | 2026-09-01 | 0.449 | 0.462 | 0.461 | … |
| BABY | 2026-09-01 | 0.509 | 0.509 | 0.505 | … |
| BANANA | 2026-09-01 | 0.503 | 0.562 | 0.446 | … |
| BCH | 2026-09-01 | 0.440 | 0.481 | 0.456 | … |
| BERA | 2026-09-01 | 0.484 | 0.504 | 0.568 | … |
| BIGTIME | 2026-09-01 | 0.556 | 0.510 | 0.572 | … |

## Why there are no target columns

Notice what's missing: `target_10d` and `target_30d` aren't here, **and they can't
be.** Those numbers are computed from what prices actually do over the *next* 10 and
30 days, and today, that future hasn't happened yet.

**That's exactly why this file exists.** It's the input you feed into your already-trained
model so it can output predictions:

1. Your model learned from the history book: *these measurements → this ranking*.
2. Today's file gives it the measurements only.
3. Your model fills in the blank, its guess at each coin's future rank, and those
   guesses become the `pred_10d` / `pred_30d` columns of your submission (next file).

Ten days later the real answers finally exist, CrowdCent compares them to your
guesses, and your score updates. Same quiz again tomorrow.
