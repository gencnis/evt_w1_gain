from pathlib import Path
from time import perf_counter
import warnings

import numpy as np
import pandas as pd

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.exceptions import ConvergenceWarning
from ucimlrepo import fetch_ucirepo


MISSING_RATE = 0.20
MAX_ITER = 50
SEEDS = list(range(10))

ROOT = Path(__file__).resolve().parents[2]

OUTPUT_CSV = (
    ROOT
    / "03_experiments"
    / "results"
    / "mice_convergence_check.csv"
)


def create_mask(shape, seed):
    rng = np.random.default_rng(seed)
    return rng.random(shape) >= MISSING_RATE


def prepare_data(x_raw, mask):
    x_missing_raw = x_raw.copy()
    x_missing_raw[~mask] = np.nan

    col_min = np.nanmin(x_missing_raw, axis=0)
    col_max = np.nanmax(x_missing_raw, axis=0)

    scale = col_max - col_min
    scale[scale < 1e-8] = 1.0

    x_missing_norm = (
        x_missing_raw - col_min
    ) / scale

    return x_missing_norm, col_min, scale


def to_raw(
    x_imputed_norm,
    x_raw,
    mask,
    col_min,
    scale,
):
    x_imputed_raw = (
        x_imputed_norm * scale
        + col_min
    )

    x_imputed_raw[mask] = x_raw[mask]

    return x_imputed_raw


def official_normalization(data, parameters=None):
    norm_data = data.copy().astype(np.float64)

    _, dim = norm_data.shape

    if parameters is None:
        min_val = np.zeros(dim)
        max_val = np.zeros(dim)

        for i in range(dim):
            min_val[i] = np.nanmin(norm_data[:, i])

            norm_data[:, i] -= min_val[i]

            max_val[i] = np.nanmax(norm_data[:, i])

            norm_data[:, i] /= (
                max_val[i] + 1e-6
            )

        parameters = {
            "min_val": min_val,
            "max_val": max_val,
        }

    else:
        min_val = parameters["min_val"]
        max_val = parameters["max_val"]

        for i in range(dim):
            norm_data[:, i] -= min_val[i]

            norm_data[:, i] /= (
                max_val[i] + 1e-6
            )

    return norm_data, parameters


def official_rmse(
    x_raw,
    x_imputed_raw,
    mask,
):
    x_true_norm, params = official_normalization(
        x_raw
    )

    x_imp_norm, _ = official_normalization(
        x_imputed_raw,
        params,
    )

    missing = ~mask

    return np.sqrt(
        np.mean(
            (
                x_true_norm[missing]
                - x_imp_norm[missing]
            ) ** 2
        )
    )


def main():
    dataset = fetch_ucirepo(id=94)

    x_raw = dataset.data.features.to_numpy(
        dtype=np.float64
    )

    rows = []

    for seed in SEEDS:
        mask = create_mask(
            x_raw.shape,
            seed,
        )

        (
            x_missing_norm,
            col_min,
            scale,
        ) = prepare_data(
            x_raw,
            mask,
        )

        imputer = IterativeImputer(
            max_iter=MAX_ITER,
            random_state=seed,
            initial_strategy="mean",
            skip_complete=True,
        )

        start = perf_counter()

        with warnings.catch_warnings(
            record=True
        ) as caught:
            warnings.simplefilter(
                "always",
                ConvergenceWarning,
            )

            x_imputed_norm = imputer.fit_transform(
                x_missing_norm
            )

        runtime = perf_counter() - start

        convergence_warning = any(
            issubclass(
                w.category,
                ConvergenceWarning,
            )
            for w in caught
        )

        x_imputed_raw = to_raw(
            x_imputed_norm,
            x_raw,
            mask,
            col_min,
            scale,
        )

        rmse = official_rmse(
            x_raw,
            x_imputed_raw,
            mask,
        )

        rows.append(
            {
                "seed": seed,
                "max_iter": MAX_ITER,
                "actual_iterations": imputer.n_iter_,
                "convergence_warning":
                    convergence_warning,
                "rmse": rmse,
                "runtime_seconds": runtime,
            }
        )

        print(
            f"seed={seed} "
            f"rmse={rmse:.6f} "
            f"n_iter={imputer.n_iter_} "
            f"warning={convergence_warning} "
            f"runtime={runtime:.2f}s"
        )

    df = pd.DataFrame(rows)

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print()
    print("===== MICE-50 SUMMARY =====")
    print(
        f"Mean RMSE: "
        f"{df['rmse'].mean():.6f}"
    )
    print(
        f"Std RMSE:  "
        f"{df['rmse'].std(ddof=1):.6f}"
    )
    print(
        f"Warnings:  "
        f"{df['convergence_warning'].sum()}/10"
    )
    print(
        f"Mean runtime: "
        f"{df['runtime_seconds'].mean():.2f}s"
    )


if __name__ == "__main__":
    main()

