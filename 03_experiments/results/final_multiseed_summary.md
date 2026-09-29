# Final Multi-Seed Comparison

## Experimental Protocol

- Dataset: UCI Spambase
- Missingness mechanism: MCAR
- Missing rate: 20%
- Same missingness mask used for KNN, MICE-style, and GAIN within each seed
- Evaluation: author-code-style normalized RMSE
- Methods:
  - KNN, k=5
  - MICE-style IterativeImputer, max_iter=10
  - Official-code-oriented PyTorch GAIN

- Seeds: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
- GAIN iterations: 10000

## Per-Seed Results

| Seed | Method | RMSE | Runtime (s) | Warning |
|---:|---|---:|---:|---|
| 0 | KNN | 0.054372 | 2.40 | - |
| 0 | MICE-style | 0.052397 | 6.83 | early_stopping_not_reached |
| 0 | GAIN | 0.053801 | 18.32 | - |
| 1 | KNN | 0.053813 | 2.52 | - |
| 1 | MICE-style | 0.053134 | 5.02 | early_stopping_not_reached |
| 1 | GAIN | 0.054280 | 23.24 | - |
| 2 | KNN | 0.053674 | 2.05 | - |
| 2 | MICE-style | 0.051185 | 5.22 | early_stopping_not_reached |
| 2 | GAIN | 0.051353 | 18.36 | - |
| 3 | KNN | 0.052738 | 2.13 | - |
| 3 | MICE-style | 0.051031 | 6.68 | early_stopping_not_reached |
| 3 | GAIN | 0.052035 | 21.79 | - |
| 4 | KNN | 0.054163 | 2.60 | - |
| 4 | MICE-style | 0.052342 | 11.12 | early_stopping_not_reached |
| 4 | GAIN | 0.053327 | 20.17 | - |
| 5 | KNN | 0.052941 | 2.12 | - |
| 5 | MICE-style | 0.051668 | 8.68 | early_stopping_not_reached |
| 5 | GAIN | 0.052915 | 18.32 | - |
| 6 | KNN | 0.053222 | 2.33 | - |
| 6 | MICE-style | 0.050705 | 9.46 | early_stopping_not_reached |
| 6 | GAIN | 0.053242 | 21.02 | - |
| 7 | KNN | 0.052437 | 2.25 | - |
| 7 | MICE-style | 0.050009 | 11.51 | early_stopping_not_reached |
| 7 | GAIN | 0.051567 | 19.08 | - |
| 8 | KNN | 0.053525 | 2.27 | - |
| 8 | MICE-style | 0.051094 | 11.79 | early_stopping_not_reached |
| 8 | GAIN | 0.052629 | 21.51 | - |
| 9 | KNN | 0.052444 | 2.37 | - |
| 9 | MICE-style | 0.048626 | 10.29 | early_stopping_not_reached |
| 9 | GAIN | 0.050904 | 19.91 | - |

## Aggregate Results

| Method | Finite Runs | NaN Runs | Mean RMSE | Std RMSE | Min RMSE | Max RMSE | Mean Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| KNN | 10/10 | 0 | 0.053333 | 0.000688 | 0.052437 | 0.054372 | 2.30 |
| MICE-style | 10/10 | 0 | 0.051219 | 0.001292 | 0.048626 | 0.053134 | 8.66 |
| GAIN | 10/10 | 0 | 0.052605 | 0.001112 | 0.050904 | 0.054280 | 20.17 |

## Interpretation Rule

All three methods receive the same artificially hidden cells for a given seed.

Therefore, differences between methods within the same seed are directly
comparable under this experiment.

This table is intended to serve as the final fair KNN / MICE / GAIN comparison
for the course report.
