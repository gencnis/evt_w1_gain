# Experiments

This directory contains the experimental component of the GAIN course work.

## Objective

The project investigates missing-data imputation using:

- KNN Imputation
- MICE-style Iterative Imputation
- Generative Adversarial Imputation Networks (GAIN)

In addition to the baseline comparison, the experiments examine reproducibility differences between:

- the GAIN paper formulation,
- the authors' released implementation,
- and the local PyTorch implementation.

## Dataset

The main local experiments use UCI Spambase.

- Samples: 4601
- Input features: 57
- Original missing values: 0
- Target variable is excluded from the imputation experiment.

## Local Missingness Setup

Experiments E1-E9 use a shared artificial MCAR mask.

- Target missing rate: 20%
- Missingness seed: 42
- Actual missing rate: 20.01%
- Artificially hidden cells: 52,468

The same mask is reused for the local method comparisons so that the compared
methods are evaluated on the same hidden cells.

Official-repository experiments E10 and later use the authors' bundled Spam
dataset and their separately documented missingness-generation procedure.

## Preprocessing

For the initial local pipeline, feature-wise normalization is estimated using
values that remain observable after masking.

The complete normalized data is retained only as ground truth for evaluating
the artificially hidden cells.

A later audit showed that the authors' official normalization and RMSE
procedure produces a different numerical RMSE for the same imputed values.
Both evaluation procedures are therefore documented separately.

## Evaluation

The primary metric is RMSE on artificially hidden values whose original values
are known.

Lower RMSE indicates more accurate imputation.

Runtime is recorded as a secondary measurement.

Because RMSE depends on the normalization and evaluation procedure, results
from different evaluation definitions are not treated as directly
interchangeable.

## Reproducibility

Controlled local GAIN experiments use:

- training seed: 42
- deterministic evaluation seed: 2026

This makes repeated runs directly comparable when only one intended
experimental variable is changed.

The authors' released implementation does not fix all random sources by
default. Separate seeded diagnostics are therefore used when studying its
iteration behavior.

## Scripts

1. `src/00_prepare_spambase.py`
   - Downloads UCI Spambase.
   - Generates the shared MCAR mask.
   - Normalizes the data.
   - Stores complete data, incomplete data, and mask locally.

2. `src/01_knn_baseline.py`
   - Runs KNN imputation with k=5.
   - Reports masked-cell RMSE and runtime.

3. `src/02_mice_baseline.py`
   - Runs a MICE-style chained-equation baseline using IterativeImputer.
   - Reports masked-cell RMSE and runtime.

4. `src/03_gain_baseline.py`
   - Runs the paper-oriented PyTorch GAIN implementation.
   - Supports deterministic evaluation and configurable training iterations.
   - Stores training history and run summaries.

5. `src/04_plot_preliminary_results.py`
   - Generates figures from the preliminary result table.

6. `src/05_plot_controlled_results.py`
   - Generates controlled RMSE, runtime, and GAIN training-loss figures.

7. `src/06_gain_official_hint_loss.py`
   - Implements a PyTorch variant closer to the authors' released hint and
     loss formulation.

8. `src/07_gain_official_evaluation.py`
   - Audits the effect of the authors' normalization and RMSE procedure.
   - Reports both the local evaluation RMSE and author-code-style RMSE.

## Result Files

Main result files:

- `results/preliminary_results.csv`
- `results/preliminary_results.md`
- `results/evaluation_audit.csv`
- `experiment_log.md`
- `implementation_audit.md`

GAIN run-specific files include:

- `results/E5_gain_loss_history.csv`
- `results/E5_gain_summary.csv`
- `results/E6_gain_loss_history.csv`
- `results/E6_gain_summary.csv`
- `results/E7_gain_loss_history.csv`
- `results/E7_gain_summary.csv`
- `results/E8_gain_loss_history.csv`
- `results/E8_gain_summary.csv`
- `results/E9_gain_loss_history.csv`
- `results/E9_gain_summary.csv`

## Figures

Generated figures are stored in `../04_figures_tables/`.

Current figures include:

- `preliminary_rmse_comparison.png`
- `preliminary_runtime_comparison.png`
- `controlled_rmse_comparison.png`
- `controlled_runtime_comparison.png`
- `gain_reconstruction_loss.png`

## Local Baseline Results

| Experiment | Method | Configuration | RMSE | Runtime |
|---|---|---|---:|---:|
| E1 | KNN | k=5 | 0.057377 | 2.21 s |
| E2 | MICE-style | max_iter=10 | 0.056717 | 6.40 s |
| E5 | GAIN paper-oriented | 2000 iterations | 0.058253 | 4.81 s |
| E6 | GAIN paper-oriented | 10000 iterations | 0.058372 | 21.34 s |
| E7 | GAIN official-code-oriented | 2000 iterations | 0.058247 | 3.72 s |

## Evaluation Audit

The same official-code-oriented model was evaluated using two RMSE procedures.

| Experiment | Iterations | Local RMSE | Author-code RMSE |
|---|---:|---:|---:|
| E8 | 2000 | 0.058247 | 0.054750 |
| E9 | 10000 | 0.058006 | 0.054585 |

The difference demonstrates that reproducing a reported RMSE requires
reproducing the evaluation and normalization procedure, not only the model.

## Paper vs Official Code

The theoretical hint formulation in the paper and the authors' released
implementation are not identical.

The paper uses a hint representation containing 0.5-valued uncertain entries,
while the released implementation uses a hint derived as a masked subset of
observed entries.

A controlled comparison between E5 and E7 produced nearly identical RMSE
values under the current MCAR setting. Therefore, this implementation
difference did not explain the observed performance gap in the local
experiment.

See `implementation_audit.md` for details.

## Official Implementation Reproduction

The authors' official repository was tested at commit:

`ed53e6d0be14a8d4ce35eff46449d4047bcb483e`

Environment:

- Python 3.10
- TensorFlow CPU 2.15.1
- NumPy 1.26.4

Two unseeded 10000-iteration runs produced NaN RMSE.

However, when seed=42 was fixed, the official implementation remained finite:

| Iterations | RMSE |
|---:|---:|
| 1000 | 0.0564 |
| 2000 | 0.0548 |
| 5000 | 0.0541 |
| 10000 | 0.0533 |

Therefore, the earlier NaN outcomes cannot be attributed to iteration count
alone. They indicate run-to-run sensitivity in the unseeded official
implementation under the tested compatibility environment.

The original GAIN paper reports Spam RMSE of 0.0513 ± 0.0016. A multi-seed
official-code reproduction is planned before making a direct comparison with
that reported mean and standard deviation.

## Experimental Rule

When testing the effect of a parameter or implementation choice, only one
intended causal variable should be changed at a time.

Every experimental run should record:

- experiment ID
- implementation variant
- configuration
- random seed
- evaluation procedure
- RMSE
- runtime
- warnings or failures
- interpretation
