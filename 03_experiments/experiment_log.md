# Experiment Log

## Shared Experimental Setup

- Dataset: UCI Spambase
- Samples: 4601
- Features: 57
- Original missing values: 0
- Missingness mechanism: MCAR
- Target missing rate: 20%
- Actual missing rate: 20.01%
- Artificially hidden cells: 52,468
- Random seed: 42
- Evaluation metric: RMSE
- RMSE is calculated only on artificially hidden cells.
- The same missingness mask is used for all compared methods.
- Input features are normalized before imputation.

---

## E1 - KNN Baseline

- Script: `src/01_knn_baseline.py`
- Method: KNNImputer
- Number of neighbors: 5
- Weights: uniform
- RMSE: 0.057377
- Runtime: 2.21 s

### Observation

This experiment provides the first simple baseline for the imputation task.

---

## E2 - MICE-style Baseline

- Script: `src/02_mice_baseline.py`
- Method: scikit-learn IterativeImputer
- max_iter: 10
- initial_strategy: mean
- Random seed: 42
- RMSE: 0.056717
- Runtime: 6.40 s

### Observation

The MICE-style baseline produced a slightly lower RMSE than KNN.

A convergence warning was produced because the early stopping criterion was not reached within 10 iterations.

This implementation is a chained-equation / MICE-style iterative imputation baseline and is not a complete multiple-imputation plus pooling workflow.

---

## E3 - GAIN Baseline

- Script: `src/03_gain_baseline.py`
- Device: CUDA
- GPU: NVIDIA GeForce RTX 3070 Ti Laptop GPU
- Iterations: 2000
- Batch size: 128
- Hint rate: 0.90
- Alpha: 100
- Learning rate: 0.001
- Random seed: 42
- RMSE: 0.058260
- Training time: 4.92 s
- Final reconstruction loss: 0.001905

### Observation

The GAIN implementation trained successfully and produced valid imputations.

At 2000 iterations, its RMSE was slightly higher than both KNN and the MICE-style baseline.

---

## E4 - GAIN Longer Training

- Script: `src/03_gain_baseline.py`
- Same configuration as E3
- Only intentionally changed parameter: number of iterations
- Iterations: 10000
- RMSE: 0.058377
- Training time: 23.64 s
- Final reconstruction loss: 0.000190

### Observation

Increasing the number of iterations substantially reduced the reconstruction loss but did not improve the masked-value RMSE.

This indicates that lower reconstruction loss on observed components does not necessarily imply better imputation accuracy on missing components.

However, the E3 and E4 RMSE values should not yet be treated as a clean comparison of iteration count because the evaluation stage generates a new random noise sample.

A reproducibility fix is required before drawing a conclusion about the effect of training iterations.

---

## E5 - Controlled GAIN Baseline

- Script: `src/03_gain_baseline.py`
- Training seed: 42
- Evaluation seed: 2026
- Iterations: 2000
- Batch size: 128
- Hint rate: 0.90
- Alpha: 100
- Learning rate: 0.001
- Device: CUDA
- RMSE: 0.058253
- Training time: 4.81 s
- Final reconstruction loss: 0.001905

### Observation

This experiment repeats the 2000-iteration GAIN baseline with deterministic evaluation noise.

The RMSE is nearly identical to E3, indicating that the original 2000-iteration result was stable.

---

## E6 - Controlled GAIN Longer Training

- Script: `src/03_gain_baseline.py`
- Training seed: 42
- Evaluation seed: 2026
- Same configuration as E5
- Only intentionally changed parameter: iterations
- Iterations: 10000
- RMSE: 0.058372
- Training time: 21.34 s
- Final reconstruction loss: 0.000190

### Observation

Increasing training from 2000 to 10000 iterations substantially reduced reconstruction loss but did not improve masked-cell RMSE.

The first 2000 iterations reproduce the E5 loss trajectory, confirming deterministic training under the current setup.

The RMSE difference is small and comes from a single training seed, so it is not yet interpreted as evidence that 10000 iterations are generally worse.
