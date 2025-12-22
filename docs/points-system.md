# CrowdCent Points (CC Points)

The CrowdCent Points (CC Points) system rewards consistent participation and high-quality predictive performance.

## Summary

- **Participation**: Earn a base credit that grows with your daily streak.
- **Performance**: Earn the majority of points based on the **composite percentile** of your daily predictions.
- **Influence**: Your points translate directly to your weight in the Meta-Model.

---

## Daily Base Credit

You earn a guaranteed **0.5 points** just for participating, provided you make at least one valid submission during the daily window.

Maintaining a daily streak increases your base credit:

| Streak Length | Multiplier | Daily Base |
|:---|:---:|:---:|
| Days 1–29 | 1.0x | 0.5 pts |
| Days 30–89 | 1.5x | 0.75 pts |
| Days 90+ | 2.0x | 1.0 pts |

Missing a submission window resets your streak to 0.

## Performance Adjustment (The Core Driver)

Your primary source of points is the quality of your predictions relative to other participants.

At the end of the day, all your valid slots are evaluated against the target metrics. We calculate the **composite percentile** of all your evaluated slots, then apply a **cosine curve** to this percentile.

### The Curve
The curve rewards consistency—moving from "average" (50th) to "good" (70th) yields the steepest point gains. Bad days are penalized, but less severely than a linear model. The curve flattens at the extremes to discourage variance.

![Points Curve](overrides/assets/images/points-curve.png)

**Formula:**

\[
\text{Adjustment} = -\cos\left(\text{AvgPercentile} \times \frac{\pi}{100}\right) \times 10
\]

**Key Benchmarks:**

- **0th percentile**: -10.0 pts
- **50th percentile**: 0.0 pts
- **60th percentile**: +3.1 pts
- **70th percentile**: +5.9 pts
- **80th percentile**: +8.1 pts
- **100th percentile**: +10.0 pts

## Final Daily Score

Your final score for the day cannot be negative.

\[
\text{Daily Points} = \max(0, (\text{Base Credit} \times \text{Streak Multiplier}) + \text{Performance Adjustment})
\]

**Example (Assuming >90 day streak = 1.0 base):**

| Scenario | Avg Percentile | Base (w/ Mult) | Adjustment | Calculation | Final Points |
| :--- | :---: | :---: | :---: | :--- | :---: |
| **Top Tier** | 80th | 1.0 | +8.1 | `1.0 + 8.1` | **9.1** |
| **Consistent** | 70th | 1.0 | +5.9 | `1.0 + 5.9` | **6.9** |
| **Average** | 50th | 1.0 | 0.0 | `1.0 + 0.0` | **1.0** |
| **Poor Day** | 20th | 1.0 | -8.1 | `1.0 - 8.1` | **0.0** |

## Tier System

As you accumulate CC Points, you'll progress through **5 tiers**—each with a unique badge displayed on your profile and in the leaderboard:

| Tier | Points Required | Description |
|:---|:---:|:---|
| **Citizen** | >0 | Welcome to CrowdCent |
| **Challenger** | 100+ | Active participant |
| **Contender** | 500+ | Rising competitor |
| **Centurion** | 1,500+ | Commander of predictions |
| **Sovereign** | 5,000+ | Pinnacle of mastery |

Higher tiers unlock greater recognition in the community and demonstrate your track record of consistent, quality predictions.

## The Meta-Model: Why Points Matter

The ultimate goal of the CrowdCent Challenge is to build a meta-model—an aggregate of all participants that outperforms any single participant on their own.

**Your Points = Your Weight.** The influence of your predictions on the meta-model is proportional to the **Exponential Moving Average (EMA)** of your Total Points history. Recent performance is weighted significantly higher than past performance, so you cannot "rest on your laurels." To maintain high influence, you must perform consistently.

![Meta-Model Weighting](overrides/assets/images/meta-model-weighting.png)

**Formula:**

\[
\text{Weight}_u = \sum_{t=0}^{\text{today}} \text{DailyPoints}_{u,t} \times (0.5)^{\frac{\text{days\_ago}}{7}}
\]

## Program Terms

**CrowdCent reserves the right to modify, suspend, or cancel the points system, scoring rules, and reward structures at any time without prior notice.**

This includes, but is not limited to:
- Adjusting point values, multipliers, or calculation formulas retroactively or proactively.
- Removing points from users found to be in violation of the spirit of the competition.
- Changing the weighting mechanisms for the meta-model.
- Resetting point totals for specific challenges or globally.

Participation in the points program does not create any property rights or contractual obligations. Points have no monetary value and are used solely for ranking and weighting purposes within the CrowdCent ecosystem.
