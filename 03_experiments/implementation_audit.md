# GAIN Implementation Audit

## Purpose

The preliminary experiments produced an RMSE of approximately 0.0583 for the
current PyTorch implementation of GAIN on UCI Spambase.

The original GAIN paper reports an RMSE of approximately 0.0513 on the Spam
dataset under its reported experimental setting.

Before tuning hyperparameters, the implementation is compared against both:

1. the algorithm described in the GAIN paper, and
2. the authors' official released implementation.

The purpose is to identify methodological differences before interpreting
performance differences.

## Current Implementation

Experiments E5 and E6 use a PyTorch implementation designed primarily from the
algorithmic formulation presented in the paper.

Shared parameters:

- Dataset: UCI Spambase
- Missing rate: 20%
- Missingness mechanism: MCAR
- Batch size: 128
- Hint rate: 0.90
- Alpha: 100
- Learning rate: 0.001
- Training seed: 42
- Evaluation seed: 2026

## Paper vs Official Code vs Current Implementation

| Component | Paper formulation | Authors' official code | Current E5/E6 implementation |
|---|---|---|---|
| Generator input | Data + mask, with noise in missing components | Same | Same |
| Generator architecture | Fully connected network | 2 hidden ReLU layers + sigmoid output | Same high-level architecture |
| Discriminator input | Completed data + hint | Same | Same |
| Hint | Mask information with uncertain entries represented by 0.5 | `H = M * H_temp` | Paper-style 0.5 hint |
| Discriminator loss | Applied to entries selected as uncertain by the hint mechanism | Computed over all components | Applied to uncertain components |
| Generator adversarial loss | Missing components | Missing components using matrix mean | Missing and uncertain components |
| Reconstruction loss | Observed components | Observed components | Observed components |
| Noise | Uniform small random noise | Uniform [0, 0.01] | Uniform [0, 0.01] |
| Batch size | Hyperparameter | 128 in example configuration | 128 |
| Hint rate | Hyperparameter | 0.9 in example configuration | 0.9 |
| Alpha | Hyperparameter | 100 in example configuration | 100 |
| Iterations | Hyperparameter | 10000 in example configuration | 2000 and 10000 tested |
| Weight initialization | Not central to Algorithm 1 | Custom Xavier-style initialization | PyTorch default Linear initialization |
| Evaluation | RMSE on artificially missing values | Official normalization and RMSE utility | RMSE in the experiment's normalized feature space |

## Important Observation

The paper formulation and the authors' released implementation are not
identical, particularly in the implementation of the hint mechanism and the
loss computation.

Therefore, reproducing the exact numerical result reported in the paper should
not be treated as equivalent to merely implementing Algorithm 1 literally.

## Next Experiment

E7 will implement a second GAIN variant that follows the authors' released
code more closely.

The existing E5/E6 implementation will be preserved as the paper-oriented
variant.

The two variants will be compared under the same Spambase data and missingness
mask before any further hyperparameter tuning is attempted.

## Evaluation Protocol Audit

Experiment E8 tested whether the RMSE evaluation procedure contributes to the
difference between the current results and the result reported in the original
GAIN paper.

The same official-code-oriented trained model produced:

- Current experiment RMSE: 0.058247
- Author-code-style RMSE: 0.054750

No change was made to the learned model or its imputed values between these two
measurements.

The difference therefore originates from the evaluation normalization and RMSE
procedure rather than from improved imputation.

This finding demonstrates that reproducing a reported RMSE requires reproducing
the evaluation procedure in addition to the model architecture and training
procedure.

## Reference Links

- GAIN paper:
  https://proceedings.mlr.press/v80/yoon18a.html
- Authors' official implementation:
  https://github.com/jsyoon0823/GAIN
- Discussion of the hint-mechanism discrepancy:
  https://github.com/jsyoon0823/GAIN/issues/2
