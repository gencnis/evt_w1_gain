# Experiments

Experimental comparison of missing-data imputation methods for the EVT GAIN assignment.

## Planned Methods

- GAIN
- KNN Imputation
- MICE / Iterative Imputation

## Initial Dataset

UCI Spambase

## Initial Experimental Setting

- Missingness mechanism: MCAR
- Missing rate: 20%
- Same missingness mask for every method
- Primary metric: RMSE on artificially masked values
- Runtime will also be recorded
- Experiments will later be repeated across multiple random seeds

## Directory Structure

- `data/`: local datasets, not committed to Git
- `src/`: experiment source code
- `results/`: generated numerical results

## Experimental Principles

1. All methods must receive the same train/test data and missingness masks.
2. Evaluation must use only artificially hidden values whose ground truth is known.
3. Random seeds and hyperparameters must be recorded.
4. Data leakage must be prevented.
5. Results must be reproducible from the documented configuration.
