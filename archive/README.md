# Archive

`FPL_Optimiser.py` is the original single-file optimiser this project grew out of. It is kept
for reference only and is not part of the build: it needs `pulp` and `requests`, which the
project no longer depends on, and it is excluded from lint and type checks.

What replaced it:

| This script | Now |
|---|---|
| Fetching and flattening FPL data | `ingest/` |
| Expected points taken from FPL's own figures | `ml/`, `train/`, `predict/` (a trained model, backtested against FPL's) |
| The PuLP linear program | `optimise/model.py` (SciPy `milp`, HiGHS) |
| Squad, selling prices and free transfers | `optimise/squad.py` |
