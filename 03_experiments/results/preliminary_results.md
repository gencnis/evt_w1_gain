# Preliminary Experiment Results

## Experimental Setup

- Dataset: UCI Spambase
- Samples: 4601
- Features: 57
- Missingness mechanism: MCAR
- Target missing rate: 20%
- Actual missing rate: 20.01%
- Artificially hidden cells: 52,468
- Missingness seed: 42
- Evaluation metric: RMSE on artificially hidden cells
- Same missingness mask used for all methods

## Results

| Experiment | Method | Configuration | RMSE | Runtime | Status |
|---|---|---|---:|---:|---|
| E1 | KNN | k=5 | 0.057377 | 2.21 s | Baseline |
| E2 | MICE-style | max_iter=10 | 0.056717 | 6.40 s | Baseline |
| E3 | GAIN | 2000 iterations | 0.058260 | 4.92 s | Initial |
| E4 | GAIN | 10000 iterations | 0.058377 | 23.64 s | Initial |
| E5 | GAIN | 2000 iterations, deterministic evaluation | 0.058253 | 4.81 s | Controlled |
| E6 | GAIN | 10000 iterations, deterministic evaluation | 0.058372 | 21.34 s | Controlled |

## Controlled GAIN Comparison

E5 and E6 use:

- the same dataset
- the same missingness mask
- the same training seed
- the same evaluation seed
- the same architecture
- the same batch size
- the same hint rate
- the same alpha
- the same learning rate

The only intentionally changed variable is the number of training iterations.

| Experiment | Iterations | RMSE | Runtime | Final Reconstruction Loss |
|---|---:|---:|---:|---:|
| E5 | 2000 | 0.058253 | 4.81 s | 0.001905 |
| E6 | 10000 | 0.058372 | 21.34 s | 0.000190 |

## Current Interpretation

Increasing GAIN training from 2000 to 10000 iterations reduced the reconstruction loss substantially but did not improve RMSE on the artificially hidden cells.

This shows that improved reconstruction of observed values does not necessarily correspond to improved imputation accuracy on missing values.

The RMSE difference between E5 and E6 is small and is based on a single training seed. Therefore, no general conclusion about the optimal number of iterations is drawn yet.

E5 and E6 also reproduce identical training losses during their first 2000 iterations, providing a useful reproducibility check.

These experiments are preliminary and are not yet the final multi-seed benchmark.
