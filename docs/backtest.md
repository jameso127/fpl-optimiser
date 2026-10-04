# Backtest results

Generated 2026-10-04. Walk-forward: each gameweek is scored by a model trained only on earlier gameweeks. Tested gameweeks: 3-5 (3 gameweeks, 1980 player-rows). Lower MAE/RMSE and higher Spearman are better.

## All players

| method | MAE | RMSE | Spearman | rows |
|---|---|---|---|---|
| model | 1.272 | 2.190 | 0.658 | 1980 |
| baseline_l5 | 1.260 | 2.365 | 0.685 | 1980 |
| baseline_season | 1.260 | 2.365 | 0.685 | 1980 |

## Regular players (>= 60 min/game over last 5)

| method | MAE | RMSE | Spearman | rows |
|---|---|---|---|---|
| model | 2.620 | 3.440 | 0.122 | 551 |
| baseline_l5 | 2.865 | 3.774 | 0.170 | 551 |
| baseline_season | 2.865 | 3.774 | 0.170 | 551 |

## Verdict

- **All players**: on MAE the model does NOT beat the best baseline (`baseline_l5`), 1.272 vs 1.260; on Spearman it does NOT beat `baseline_l5`, 0.658 vs 0.685.
- **Regular players (>= 60 min/game over last 5)**: on MAE the model beats the best baseline (`baseline_l5`), 2.620 vs 2.865; on Spearman it does NOT beat `baseline_l5`, 0.122 vs 0.170.
- FPL's `ep_next` benchmark is **not yet available**: it can only be captured before a deadline and no tested gameweek has a stored snapshot. The model has therefore not been shown to beat `ep_next`.

## Caveats

- Tiny sample (current season only, few gameweeks), so differences are noisy.
- `baseline_l5` and `baseline_season` coincide until more than 5 gameweeks exist.
- Fixture difficulty and team strengths use today's values for past gameweeks.
- Position and team come from the latest snapshot.
