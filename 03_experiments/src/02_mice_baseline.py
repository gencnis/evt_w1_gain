from pathlib import Path
from time import perf_counter

import numpy as np

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer


SEED = 42

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT
    / "03_experiments"
    / "data"
    / f"spambase_mcar20_seed{SEED}.npz"
)


def masked_rmse(X_true, X_imputed, mask):
    missing = ~mask

    squared_error = (
        X_true[missing] - X_imputed[missing]
    ) ** 2

    return np.sqrt(np.mean(squared_error))


def main():
    data = np.load(DATA_PATH)

    X_true = data["X_true"]
    X_missing = data["X_missing"]
    mask = data["mask"]

    print(f"Shape: {X_true.shape}")
    print(f"Missing rate: {(~mask).mean():.4f}")

    imputer = IterativeImputer(
        max_iter=10,
        random_state=SEED,
        initial_strategy="mean",
        skip_complete=True,
    )

    start = perf_counter()

    X_imputed = imputer.fit_transform(X_missing)

    runtime = perf_counter() - start

    rmse = masked_rmse(
        X_true,
        X_imputed,
        mask,
    )

    print(f"MICE-style RMSE: {rmse:.6f}")
    print(f"MICE-style runtime: {runtime:.2f} seconds")


if __name__ == "__main__":
    main()
