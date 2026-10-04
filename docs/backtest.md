# Backtest results

Generated 2026-10-04. Walk-forward: each gameweek is scored by a model trained only on earlier seasons plus earlier gameweeks of the same season. Tested: 2025-26 (gameweeks 3-38), 2026-27 (gameweeks 3-5); 29923 player-rows. Lower MAE/RMSE and higher Spearman are better.

`ep_next` is available for: 2025-26: 9 of 36 tested gameweeks; 2026-27: 0 of 3 tested gameweeks.

## 2025-26: all players

| method | MAE | RMSE | Spearman | rows |
|---|---|---|---|---|
| model | 0.973 | 1.937 | 0.713 | 27943 |
| baseline_l5 | 1.052 | 2.124 | 0.720 | 27943 |
| baseline_season | 1.063 | 2.063 | 0.689 | 27943 |

## 2025-26: regular players (>= 60 min/game over last 5)

| method | MAE | RMSE | Spearman | rows |
|---|---|---|---|---|
| model | 2.312 | 3.137 | 0.275 | 6113 |
| baseline_l5 | 2.708 | 3.497 | 0.137 | 6113 |
| baseline_season | 2.493 | 3.317 | 0.171 | 6113 |

## 2025-26: head-to-head with ep_next, all players

| method | MAE | RMSE | Spearman | rows |
|---|---|---|---|---|
| model | 0.969 | 1.896 | 0.720 | 6762 |
| model_xp | 0.965 | 1.897 | 0.723 | 6762 |
| baseline_l5 | 1.053 | 2.080 | 0.732 | 6762 |
| baseline_season | 1.060 | 2.052 | 0.717 | 6762 |
| ep_next | 1.141 | 2.168 | 0.672 | 6762 |

## 2025-26: head-to-head with ep_next, regular players (>= 60 min/game over last 5)

| method | MAE | RMSE | Spearman | rows |
|---|---|---|---|---|
| model | 2.247 | 3.044 | 0.297 | 1543 |
| model_xp | 2.247 | 3.041 | 0.303 | 1543 |
| baseline_l5 | 2.637 | 3.398 | 0.189 | 1543 |
| baseline_season | 2.540 | 3.313 | 0.213 | 1543 |
| ep_next | 2.677 | 3.534 | 0.187 | 1543 |

## 2026-27: all players

| method | MAE | RMSE | Spearman | rows |
|---|---|---|---|---|
| model | 1.185 | 2.115 | 0.689 | 1980 |
| baseline_l5 | 1.260 | 2.365 | 0.685 | 1980 |
| baseline_season | 1.260 | 2.365 | 0.685 | 1980 |

## 2026-27: regular players (>= 60 min/game over last 5)

| method | MAE | RMSE | Spearman | rows |
|---|---|---|---|---|
| model | 2.490 | 3.325 | 0.235 | 551 |
| baseline_l5 | 2.865 | 3.774 | 0.170 | 551 |
| baseline_season | 2.865 | 3.774 | 0.170 | 551 |

## Leak audit of stored `ep_next`

Correlation of the change in `ep_next` (g-1 to g) with points scored in g. A pre-deadline value should show about 0 here and a clear positive correlation with points in g-1. `LEAK WARNING` means |corr| >= 0.1: the value contains the outcome and every `ep_next` result in this report is invalid.

| season | rows | corr_with_points_in_g | corr_with_points_in_g-1 | status |
|---|---|---|---|---|
| 2022-23 | 4730 | 0.001 | 0.412 | OK |
| 2023-24 | 5439 | 0.025 | 0.483 | OK |
| 2024-25 | 5044 | 0.044 | 0.529 | OK |
| 2025-26 | 1021 | 0.011 | 0.617 | OK |

## Verdict

- **2025-26: all players**: on MAE `model` beats `baseline_l5`, 0.973 vs 1.052; on Spearman it does NOT beat `baseline_l5`, 0.713 vs 0.720.
- **2025-26: regular players (>= 60 min/game over last 5)**: on MAE `model` beats `baseline_season`, 2.312 vs 2.493; on Spearman it beats `baseline_season`, 0.275 vs 0.171.
- **2025-26: head-to-head with ep_next, all players: model_xp vs ep_next**: on MAE `model_xp` beats `ep_next`, 0.965 vs 1.141; on Spearman it beats `ep_next`, 0.723 vs 0.672.
- **2025-26: head-to-head with ep_next, all players: model_xp vs the model without ep_next**: on MAE `model_xp` beats `model`, 0.965 vs 0.969; on Spearman it beats `model`, 0.723 vs 0.720.
- **2025-26: head-to-head with ep_next, all players: model vs ep_next**: on MAE `model` beats `ep_next`, 0.969 vs 1.141; on Spearman it beats `ep_next`, 0.720 vs 0.672.
- **2025-26: head-to-head with ep_next, regular players (>= 60 min/game over last 5): model_xp vs ep_next**: on MAE `model_xp` beats `ep_next`, 2.247 vs 2.677; on Spearman it beats `ep_next`, 0.303 vs 0.187.
- **2025-26: head-to-head with ep_next, regular players (>= 60 min/game over last 5): model_xp vs the model without ep_next**: on MAE `model_xp` does NOT beat `model`, 2.247 vs 2.247; on Spearman it beats `model`, 0.303 vs 0.297.
- **2025-26: head-to-head with ep_next, regular players (>= 60 min/game over last 5): model vs ep_next**: on MAE `model` beats `ep_next`, 2.247 vs 2.677; on Spearman it beats `ep_next`, 0.297 vs 0.187.
- **2026-27: all players**: on MAE `model` beats `baseline_l5`, 1.185 vs 1.260; on Spearman it beats `baseline_l5`, 0.689 vs 0.685.
- **2026-27: regular players (>= 60 min/game over last 5)**: on MAE `model` beats `baseline_l5`, 2.490 vs 2.865; on Spearman it beats `baseline_l5`, 0.235 vs 0.170.

## Caveats

- For past seasons `ep_next` comes from the community archive, not from snapshots we stored before deadlines, and its exact capture time is still undocumented beyond what the leak audit shows. Snapshots stored by our own ingest are the gold standard and will replace it as they accumulate.
- The archive records some gameweeks' `xP` as zero for everyone; those gameweeks are treated as missing, not scored as zero. Single-player values above 20 are also treated as missing (data errors).
- The archive's `xP` on row g was captured after gameweek g (it leaked: its change tracked points scored in g). It is therefore stored against gameweek g+1, as the value known before that deadline; the leak audit above re-checks this on every run. An earlier version of this report used the same-gameweek value and wrongly showed `ep_next` as a very strong predictor.
- `model_xp` uses `ep_next` as a feature. Live, `ep_next` comes from our own snapshot, taken at ingest time while that gameweek is the next one, so it is a genuine pre-deadline value.
- Fixture difficulty and team strengths use end-of-season values for past gameweeks.
- Position and team come from the latest snapshot of each season.
- FPL scoring rules change between seasons (e.g. defensive-contribution points from 2025-26), so older seasons' targets are not exactly comparable.
- `baseline_l5` and `baseline_season` coincide until more than 5 gameweeks exist.
