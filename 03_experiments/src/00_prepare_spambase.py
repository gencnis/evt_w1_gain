from pathlib import Path

import numpy as np
from ucimlrepo import fetch_ucirepo


SEED = 42
MISSING_RATE = 0.20

ROOT = Path(__file__).resolve().parents[2]

OUTPUT_PATH = (
    ROOT
    / "03_experiments"
    / "data"
    / f"spambase_mcar20_seed{SEED}.npz"
)


def main():
    # Load the UCI Spambase dataset.
    dataset = fetch_ucirepo(id=94)

    # Use only the input features for the imputation experiment.
    # The spam/non-spam target is intentionally excluded.
    X = dataset.data.features.to_numpy(dtype=np.float64)

    print(f"Original shape: {X.shape}")
    print(f"Original missing values: {np.isnan(X).sum()}")

    # Create a reproducible random number generator.
    rng = np.random.default_rng(SEED)

    # True  = observed
    # False = artificially hidden
    mask = rng.random(X.shape) >= MISSING_RATE

    # Copy the complete data and hide selected cells.
    X_missing = X.copy()
    X_missing[~mask] = np.nan

    # Normalize each feature using only values that remain observed
    # after masking.
    col_min = np.nanmin(X_missing, axis=0)
    col_max = np.nanmax(X_missing, axis=0)

    scale = col_max - col_min
    scale[scale < 1e-8] = 1.0

    # Complete normalized data is kept only as ground truth
    # for evaluation of the artificially hidden cells.
    X_true_norm = (X - col_min) / scale
    X_missing_norm = (X_missing - col_min) / scale

    actual_missing_rate = (~mask).mean()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        OUTPUT_PATH,
        X_true=X_true_norm,
        X_missing=X_missing_norm,
        mask=mask,
        seed=SEED,
        missing_rate=MISSING_RATE,
    )

    print(f"Artificial missing rate: {actual_missing_rate:.4f}")
    print(f"Hidden cells: {(~mask).sum():,}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
