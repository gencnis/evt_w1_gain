# Final KNN / MICE / GAIN Comparison

## Protocol

- Dataset: UCI Spambase
- Seeds: 0-9
- Missingness: 20% MCAR
- Same missingness mask for all three methods within each seed
- Evaluation: author-code-style normalized RMSE
- KNN: k=5
- MICE-style: max_iter=10
- GAIN: 10000 iterations, batch=128, hint_rate=0.9, alpha=100

## Results

| Method | Mean RMSE | Std RMSE | Min RMSE | Max RMSE | Mean Runtime (s) |
|---|---:|---:|---:|---:|---:|
| KNN | 0.053333 | 0.000688 | 0.052437 | 0.054372 | 2.30 |
| MICE-style | 0.051219 | 0.001292 | 0.048626 | 0.053134 | 8.66 |
| GAIN | 0.052605 | 0.001112 | 0.050904 | 0.054280 | 20.17 |

## Interpretation

The comparison uses identical seed-specific missingness masks for all three
methods.

MICE-style produced the lowest mean RMSE in this experiment.

GAIN produced a lower mean RMSE than KNN but required the highest runtime.

The MICE max_iter=50 sensitivity test is documented separately and is not used
as the final baseline configuration.
