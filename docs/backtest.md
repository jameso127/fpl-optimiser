# Backtest results

Generated 2026-10-05. Walk-forward: each gameweek is scored by a model trained only on earlier seasons plus earlier gameweeks of the same season. Tested: 2025-26 (gameweeks 3-38), 2026-27 (gameweeks 3-5); 29923 player-rows. Lower MAE/RMSE and higher Spearman are better.

`ep_next` is available for: 2025-26: 9 of 36 tested gameweeks; 2026-27: 0 of 3 tested gameweeks.

## 2025-26: all players

| method | MAE | RMSE | Spearman | Top30 | rows |
|---|---|---|---|---|---|
| model | 0.970 | 1.915 | 0.729 | 4.446 | 27943 |
| model_v1 | 0.973 | 1.937 | 0.713 | 4.395 | 27943 |
| baseline_l5 | 1.052 | 2.124 | 0.720 | 3.755 | 27943 |
| baseline_season | 1.063 | 2.063 | 0.689 | 3.882 | 27943 |

## 2025-26: regular players (>= 60 min/game over last 5)

| method | MAE | RMSE | Spearman | Top30 | rows |
|---|---|---|---|---|---|
| model | 2.348 | 3.122 | 0.291 | 4.418 | 6113 |
| model_v1 | 2.312 | 3.137 | 0.275 | 4.335 | 6113 |
| baseline_l5 | 2.708 | 3.497 | 0.137 | 3.731 | 6113 |
| baseline_season | 2.493 | 3.317 | 0.171 | 3.914 | 6113 |

## 2025-26: head-to-head with ep_next, all players

| method | MAE | RMSE | Spearman | Top30 | rows |
|---|---|---|---|---|---|
| model | 0.978 | 1.882 | 0.738 | 4.519 | 6762 |
| model_v1 | 0.969 | 1.896 | 0.720 | 4.378 | 6762 |
| baseline_l5 | 1.053 | 2.080 | 0.732 | 3.926 | 6762 |
| baseline_season | 1.060 | 2.052 | 0.717 | 3.996 | 6762 |
| ep_next | 1.141 | 2.166 | 0.672 | 3.933 | 6762 |

## 2025-26: head-to-head with ep_next, regular players (>= 60 min/game over last 5)

| method | MAE | RMSE | Spearman | Top30 | rows |
|---|---|---|---|---|---|
| model | 2.290 | 3.032 | 0.297 | 4.459 | 1543 |
| model_v1 | 2.247 | 3.044 | 0.297 | 4.315 | 1543 |
| baseline_l5 | 2.637 | 3.398 | 0.189 | 3.830 | 1543 |
| baseline_season | 2.540 | 3.313 | 0.213 | 3.911 | 1543 |
| ep_next | 2.673 | 3.526 | 0.187 | 3.804 | 1543 |

## 2026-27: all players

| method | MAE | RMSE | Spearman | Top30 | rows |
|---|---|---|---|---|---|
| model | 1.146 | 2.073 | 0.719 | 5.111 | 1980 |
| model_v1 | 1.185 | 2.115 | 0.689 | 5.056 | 1980 |
| baseline_l5 | 1.260 | 2.365 | 0.685 | 4.300 | 1980 |
| baseline_season | 1.260 | 2.365 | 0.685 | 4.300 | 1980 |

## 2026-27: regular players (>= 60 min/game over last 5)

| method | MAE | RMSE | Spearman | Top30 | rows |
|---|---|---|---|---|---|
| model | 2.470 | 3.301 | 0.272 | 5.089 | 551 |
| model_v1 | 2.490 | 3.325 | 0.235 | 5.033 | 551 |
| baseline_l5 | 2.865 | 3.774 | 0.170 | 4.300 | 551 |
| baseline_season | 2.865 | 3.774 | 0.170 | 4.300 | 551 |

## What each feature group is worth (2025-26, every 3rd gameweek from 3)

The full model with one group of features removed at a time. A group is only worth keeping if removing it makes the full model worse.

| variant | MAE (regular) | Spearman (regular) | Top30 (all) | Spearman (all) |
|---|---|---|---|---|
| full model | 2.353 | 0.310 | 4.817 | 0.734 |
| without player_detail | 2.367 | 0.281 | 4.797 | 0.725 |
| without team_form | 2.359 | 0.311 | 4.814 | 0.734 |
| without rest | 2.365 | 0.304 | 4.900 | 0.733 |
| without prior_season | 2.352 | 0.309 | 4.869 | 0.733 |
| without scoring_regime | 2.357 | 0.308 | 4.789 | 0.734 |
| first model's features only | 2.328 | 0.295 | 4.919 | 0.723 |

## Leak audit of stored `ep_next`

Correlation of the change in `ep_next` (g-1 to g) with points scored in g. A pre-deadline value should show about 0 here and a clear positive correlation with points in g-1. `LEAK WARNING` means |corr| >= 0.1: the value contains the outcome and every `ep_next` result in this report is invalid.

| season | rows | corr_with_points_in_g | corr_with_points_in_g-1 | status |
|---|---|---|---|---|
| 2022-23 | 4826 | 0.017 | 0.587 | OK |
| 2023-24 | 5477 | 0.030 | 0.622 | OK |
| 2024-25 | 5057 | 0.062 | 0.591 | OK |
| 2025-26 | 1021 | 0.011 | 0.617 | OK |

## Verdict

- **2025-26: all players**: on MAE `model` beats `model_v1`, 0.970 vs 0.973; on Spearman it beats `baseline_l5`, 0.729 vs 0.720; on top-30 points it beats `model_v1`, 4.446 vs 4.395.
- **2025-26: regular players (>= 60 min/game over last 5)**: on MAE `model` does NOT beat `model_v1`, 2.348 vs 2.312; on Spearman it beats `model_v1`, 0.291 vs 0.275; on top-30 points it beats `model_v1`, 4.418 vs 4.335.
- **2025-26: head-to-head with ep_next, all players**: on MAE `model` beats `ep_next`, 0.978 vs 1.141; on Spearman it beats `ep_next`, 0.738 vs 0.672; on top-30 points it beats `ep_next`, 4.519 vs 3.933.
- **2025-26: head-to-head with ep_next, regular players (>= 60 min/game over last 5)**: on MAE `model` beats `ep_next`, 2.290 vs 2.673; on Spearman it beats `ep_next`, 0.297 vs 0.187; on top-30 points it beats `ep_next`, 4.459 vs 3.804.
- **2026-27: all players**: on MAE `model` beats `model_v1`, 1.146 vs 1.185; on Spearman it beats `model_v1`, 0.719 vs 0.689; on top-30 points it beats `model_v1`, 5.111 vs 5.056.
- **2026-27: regular players (>= 60 min/game over last 5)**: on MAE `model` beats `model_v1`, 2.470 vs 2.490; on Spearman it beats `model_v1`, 0.272 vs 0.235; on top-30 points it beats `model_v1`, 5.089 vs 5.033.

## Caveats

- For past seasons `ep_next` comes from the community archive, not from snapshots we stored before deadlines, and its exact capture time is still undocumented beyond what the leak audit shows. Snapshots stored by our own ingest are the gold standard and will replace it as they accumulate.
- The archive records some gameweeks' `xP` as zero for everyone; those gameweeks are treated as missing, not scored as zero. Single-player values above 20 are also treated as missing (data errors).
- The archive's `xP` on row g was captured after gameweek g (it leaked: its change tracked points scored in g). It is therefore stored against gameweek g+1, as the value known before that deadline; the leak audit above re-checks this on every run. An earlier version of this report used the same-gameweek value and wrongly showed `ep_next` as a very strong predictor.
- Injury news (chance of playing) has no history, so the model cannot learn from it; live it is applied on top as a multiplier on the chance of playing. The backtest therefore cannot measure that adjustment.
- Rest days use the stored fixtures table (league matches only; no cup or European games), so they are a partial congestion signal. Set-piece roles and news text are stored from now on but have no history yet, so they are not features.
- Last season's rates are matched by FPL's stable player `code`; players new to the league have none.
- Fixture difficulty and team strengths use end-of-season values for past gameweeks.
- Position and team come from the latest snapshot of each season.
- FPL scoring rules change between seasons (e.g. defensive-contribution points from 2025-26), so older seasons' targets are not exactly comparable.
- `baseline_l5` and `baseline_season` coincide until more than 5 gameweeks exist.
