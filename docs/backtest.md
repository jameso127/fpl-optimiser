# Backtest results

Generated 2026-10-04. Walk-forward: each gameweek is scored by a model trained only on earlier seasons plus earlier gameweeks of the same season. Tested: 2025-26 (gameweeks 3-38), 2026-27 (gameweeks 3-5); 29923 player-rows. Lower MAE/RMSE and higher Spearman are better.

`ep_next` is available for: 2025-26: 9 of 36 tested gameweeks; 2026-27: 0 of 3 tested gameweeks.

## 2025-26: all players

| method | MAE | RMSE | Spearman | rows |
|---|---|---|---|---|
| model | 0.975 | 1.937 | 0.711 | 27943 |
| baseline_l5 | 1.052 | 2.124 | 0.720 | 27943 |
| baseline_season | 1.063 | 2.063 | 0.689 | 27943 |

## 2025-26: regular players (>= 60 min/game over last 5)

| method | MAE | RMSE | Spearman | rows |
|---|---|---|---|---|
| model | 2.312 | 3.136 | 0.277 | 6113 |
| baseline_l5 | 2.708 | 3.497 | 0.137 | 6113 |
| baseline_season | 2.493 | 3.317 | 0.171 | 6113 |

## 2025-26: head-to-head with ep_next, all players

| method | MAE | RMSE | Spearman | rows |
|---|---|---|---|---|
| model | 0.985 | 1.947 | 0.704 | 6898 |
| baseline_l5 | 1.064 | 2.133 | 0.714 | 6898 |
| baseline_season | 1.073 | 2.099 | 0.697 | 6898 |
| ep_next | 0.825 | 1.573 | 0.783 | 6898 |

## 2025-26: head-to-head with ep_next, regular players (>= 60 min/game over last 5)

| method | MAE | RMSE | Spearman | rows |
|---|---|---|---|---|
| model | 2.315 | 3.143 | 0.262 | 1554 |
| baseline_l5 | 2.665 | 3.476 | 0.191 | 1554 |
| baseline_season | 2.572 | 3.399 | 0.190 | 1554 |
| ep_next | 1.852 | 2.530 | 0.629 | 1554 |

## 2026-27: all players

| method | MAE | RMSE | Spearman | rows |
|---|---|---|---|---|
| model | 1.189 | 2.114 | 0.691 | 1980 |
| baseline_l5 | 1.260 | 2.365 | 0.685 | 1980 |
| baseline_season | 1.260 | 2.365 | 0.685 | 1980 |

## 2026-27: regular players (>= 60 min/game over last 5)

| method | MAE | RMSE | Spearman | rows |
|---|---|---|---|---|
| model | 2.487 | 3.321 | 0.240 | 551 |
| baseline_l5 | 2.865 | 3.774 | 0.170 | 551 |
| baseline_season | 2.865 | 3.774 | 0.170 | 551 |

## Verdict

- **2025-26: all players**: on MAE the model beats the best baseline (`baseline_l5`), 0.975 vs 1.052; on Spearman it does NOT beat `baseline_l5`, 0.711 vs 0.720.
- **2025-26: regular players (>= 60 min/game over last 5)**: on MAE the model beats the best baseline (`baseline_season`), 2.312 vs 2.493; on Spearman it beats `baseline_season`, 0.277 vs 0.171.
- **2025-26: head-to-head with ep_next, all players**: on MAE the model does NOT beat the best baseline (`ep_next`), 0.985 vs 0.825; on Spearman it does NOT beat `ep_next`, 0.704 vs 0.783.
- **2025-26: head-to-head with ep_next, regular players (>= 60 min/game over last 5)**: on MAE the model does NOT beat the best baseline (`ep_next`), 2.315 vs 1.852; on Spearman it does NOT beat `ep_next`, 0.262 vs 0.629.
- **2026-27: all players**: on MAE the model beats the best baseline (`baseline_l5`), 1.189 vs 1.260; on Spearman it beats `baseline_l5`, 0.691 vs 0.685.
- **2026-27: regular players (>= 60 min/game over last 5)**: on MAE the model beats the best baseline (`baseline_l5`), 2.487 vs 2.865; on Spearman it beats `baseline_l5`, 0.240 vs 0.170.

## Caveats

- For past seasons `ep_next` is the community archive's `xP`; when it was captured relative to the deadline is not documented, so it may not be a strict pre-deadline benchmark. If it was captured late (e.g. after team news), the head-to-head gap overstates FPL's real advantage. Only snapshots stored by our own ingest before a deadline give a trustworthy comparison.
- The archive records some gameweeks' `xP` as zero for everyone; those gameweeks are treated as missing, not scored as zero.
- Fixture difficulty and team strengths use end-of-season values for past gameweeks.
- Position and team come from the latest snapshot of each season.
- FPL scoring rules change between seasons (e.g. defensive-contribution points from 2025-26), so older seasons' targets are not exactly comparable.
- `baseline_l5` and `baseline_season` coincide until more than 5 gameweeks exist.
