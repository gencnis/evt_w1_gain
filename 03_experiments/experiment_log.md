# Experiment Log

## Shared Experimental Setup for E1-E9

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
- The same missingness mask is used for the local E1-E9 comparisons. Official-repository experiments E10 and later use their separately documented setup.
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

---

## E7 - Official-Code-Oriented Hint and Loss

- Script: `src/06_gain_official_hint_loss.py`
- Purpose: Test the practical effect of the discrepancy between the paper formulation and the authors' released implementation.
- Dataset and missingness mask: Same as E5
- Training seed: 42
- Evaluation seed: 2026
- Iterations: 2000
- Batch size: 128
- Hint rate: 0.90
- Alpha: 100
- Learning rate: 0.001
- Device: CUDA
- RMSE: 0.058247
- Training time: 3.72 s
- Final reconstruction loss: 0.001853

### Difference from E5

E5 follows the paper-oriented hint/loss formulation more closely.

E7 changes the following components to match the authors' released code more closely:

- Hint is computed as `H = M * H_temp`.
- No 0.5-valued uncertain hint entries are used.
- Discriminator classification loss is averaged over all components.
- Generator adversarial loss is applied to missing components and averaged over the full matrix.

All other intended experimental variables are held constant.

### Observation

E5 RMSE: 0.058253

E7 RMSE: 0.058247

The RMSE difference is negligible under the current MCAR setting and fixed seed.

Therefore, although the paper and official code differ in their hint and loss implementation, this discrepancy does not explain the gap between the current implementation and the RMSE reported in the original GAIN paper.

Training loss values between E5 and E7 should not be directly compared because their loss definitions and averaging procedures differ.

---

## E8 - Official Evaluation Audit

- Script: `src/07_gain_official_evaluation.py`
- Variant: Official-code-oriented GAIN
- Training seed: 42
- Evaluation seed: 2026
- Iterations: 2000
- Same training configuration as E7
- Current normalized-space RMSE: 0.058247
- Author-code-style RMSE: 0.054750
- Training time: 4.89 s

### Purpose

E8 isolates the effect of the RMSE evaluation procedure.

The trained model and deterministic imputation procedure are unchanged relative
to E7. The same imputed values are evaluated using two different normalization
and RMSE procedures.

### Observation

The current experiment evaluation gives an RMSE of 0.058247.

When the same imputed values are evaluated using the normalization and RMSE
logic from the authors' released GAIN utility code, the RMSE becomes 0.054750.

Therefore, evaluation and normalization details have a substantial effect on
the numerical RMSE reported for this experiment.

This improvement must not be interpreted as an improvement in the trained
model because the underlying imputations are unchanged.

The remaining difference from the RMSE reported in the original paper must be
investigated separately.

---

## E9 - Official-Code-Oriented 10000 Iterations

- Script: `src/07_gain_official_evaluation.py`
- Variant: Official-code-oriented GAIN
- Training seed: 42
- Evaluation seed: 2026
- Iterations: 10000
- Same configuration as E8 except for iteration count
- Current normalized-space RMSE: 0.058006
- Author-code-style RMSE: 0.054585
- Training time: 23.01 s
- Final reconstruction loss: 0.000185

### Observation

Increasing training from 2000 to 10000 iterations produced only a small
improvement in imputation RMSE.

Author-code-style RMSE changed from 0.054750 in E8 to 0.054585 in E9.

Therefore, the remaining difference from the original paper's reported Spam
result cannot be explained primarily by insufficient training iterations.

The largest numerical change identified so far comes from reproducing the
authors' evaluation normalization procedure rather than from increasing the
training duration.

---

## E10 - Official Authors' Implementation Reproduction Attempt

Official repository commit:

`ed53e6d0be14a8d4ce35eff46449d4047bcb483e`

Environment:

- Python: 3.10
- TensorFlow CPU: 2.15.1
- NumPy: 1.26.4

Official parameters:

- Dataset: authors' bundled Spam dataset
- Missing rate: 20%
- Batch size: 128
- Hint rate: 0.9
- Alpha: 100
- Iterations: 10000

### E10

- RMSE: NaN
- Runtime: 19.53 s

### E10b

Exact unmodified rerun:

- RMSE: NaN
- Runtime: 18.01 s

### Dataset Integrity Check

The official `data/spam.csv` file was inspected directly:

- Shape: 4601 x 57
- NaN values: 0
- Infinite values: 0
- All values finite: yes
- Minimum: 0.0
- Maximum: 15841.0

Therefore, the NaN result is not caused by invalid values in the input dataset.

### Iteration Diagnostic Sweep

The unmodified official implementation was also executed at shorter training
lengths:

| Iterations | RMSE |
|---:|---:|
| 10 | 0.3861 |
| 100 | 0.0598 |
| 500 | 0.0549 |
| 1000 | 0.0529 |
| 2000 | 0.0535 |
| 5000 | 0.0524 |
| 10000 | NaN |
| 10000 rerun | NaN |

### Interpretation

The official implementation is capable of producing finite and competitive
RMSE values in the tested modern TensorFlow compatibility environment.

The two independent 10000-iteration runs both produced NaN, while runs up to
5000 iterations remained finite.

However, the official implementation does not fix NumPy or TensorFlow random
seeds. Therefore, the iteration sweep is confounded by changes in missingness
mask, initialization, minibatch order, and random noise.

A seeded diagnostic is required before concluding that training length itself
causes the numerical instability.

---

## E11 - Seeded Official Implementation Iteration Sweep

Official repository commit:

`ed53e6d0be14a8d4ce35eff46449d4047bcb483e`

Environment:

- Python 3.10
- TensorFlow CPU 2.15.1
- NumPy 1.26.4
- Seed: 42
- oneDNN disabled for the seeded diagnostic runs

Official parameters:

- Dataset: authors' bundled Spam dataset
- Missing rate: 20%
- Batch size: 128
- Hint rate: 0.9
- Alpha: 100

Results:

| Iterations | RMSE |
|---:|---:|
| 1000 | 0.0564 |
| 2000 | 0.0548 |
| 5000 | 0.0541 |
| 10000 | 0.0533 |

### Interpretation

With the random seed fixed, the official implementation remained finite at
10000 iterations and RMSE improved as training duration increased.

Therefore, the two earlier unseeded 10000-iteration NaN runs cannot be
attributed to iteration count alone.

The evidence indicates run-to-run stochastic sensitivity in some unseeded
runs of the official implementation under the tested modern TensorFlow
compatibility environment. The exact numerical cause of the NaN outcomes
has not yet been isolated.

This experiment also demonstrates why uncontrolled random seeds make direct
iteration-count comparisons unreliable.
