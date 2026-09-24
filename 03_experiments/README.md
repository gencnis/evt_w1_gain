# Experiments

This directory contains the experimental component of the GAIN course work.

## Objective

The goal is to compare missing-data imputation performance using:

- KNN Imputation
- MICE-style Iterative Imputation
- Generative Adversarial Imputation Networks (GAIN)

The experiments use the same dataset, missingness mechanism, missingness mask, and evaluation metric whenever methods are compared.

## Dataset

UCI Spambase

- Samples: 4601
- Input features: 57
- Original missing values: 0
- Target variable is excluded from the imputation experiment.

## Missingness Generation

Artificial missing values are generated using an MCAR mechanism.

- Target missing rate: 20%
- Random seed: 42
- Actual missing rate in the current mask: 20.01%
- Artificially hidden cells: 52,468

The same mask is reused for all methods so that every imputation method is evaluated on exactly the same hidden cells.

## Preprocessing

Feature-wise normalization is performed using values that remain observable after masking.

The complete normalized data is retained only as ground truth for evaluating artificially hidden cells.

## Evaluation

The primary metric is RMSE.

RMSE is calculated only on artificially hidden cells whose original values are known.

Lower RMSE indicates more accurate reconstruction of the hidden values.

Runtime is also recorded as a secondary measurement.

## Scripts

1. `src/00_prepare_spambase.py`
   - Downloads Spambase.
   - Creates the MCAR mask.
   - Normalizes the data.
   - Stores the complete data, incomplete data, and mask locally.

2. `src/01_knn_baseline.py`
   - Runs KNN imputation.
   - Uses 5 nearest neighbors.
   - Reports masked-cell RMSE and runtime.

3. `src/02_mice_baseline.py`
   - Runs a MICE-style chained-equation baseline using IterativeImputer.
   - Reports masked-cell RMSE and runtime.

4. `src/03_gain_baseline.py`
   - Runs the PyTorch GAIN implementation.
   - Uses the observation mask, random noise, hint mechanism, generator, and discriminator.
   - Reports masked-cell RMSE and training time.

5. `src/04_plot_preliminary_results.py`
   - Reads the preliminary results CSV.
   - Generates RMSE and runtime comparison figures.

## Results

Numerical results are stored in:

- `results/preliminary_results.csv`
- `results/preliminary_results.md`

Detailed experiment history is stored in:

- `experiment_log.md`

Figures are stored in:

- `../04_figures_tables/preliminary_rmse_comparison.png`
- `../04_figures_tables/preliminary_runtime_comparison.png`

## Preliminary Results

| Experiment | Method | Main Configuration | RMSE | Runtime |
|---|---|---|---:|---:|
| E1 | KNN | k=5 | 0.057377 | 2.21 s |
| E2 | MICE-style | max_iter=10 | 0.056717 | 6.40 s |
| E3 | GAIN | 2000 iterations | 0.058260 | 4.92 s |
| E4 | GAIN | 10000 iterations | 0.058377 | 23.64 s |

These results are preliminary and should not yet be interpreted as the final benchmark.

In particular, the current GAIN evaluation contains stochastic sampling during inference. This will be addressed before final experiments are performed.

## Experimental Rule

When testing the effect of a parameter, only one causal variable should be intentionally changed at a time.

Every experimental run should record:

- experiment ID
- method
- configuration
- random seed
- RMSE
- runtime
- relevant warnings
- interpretation
