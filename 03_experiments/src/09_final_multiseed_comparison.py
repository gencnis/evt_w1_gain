from pathlib import Path
from time import perf_counter
import argparse
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.exceptions import ConvergenceWarning

from ucimlrepo import fetch_ucirepo


# ============================================================
# Configuration
# ============================================================

MISSING_RATE = 0.20

KNN_K = 5
MICE_MAX_ITER = 10

GAIN_ITERATIONS = 10000
BATCH_SIZE = 128
HINT_RATE = 0.90
ALPHA = 100.0
LEARNING_RATE = 1e-3

EVAL_SEED_BASE = 2026

ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = (
    ROOT
    / "03_experiments"
    / "results"
)

RESULTS_CSV = (
    RESULTS_DIR
    / "final_multiseed_results.csv"
)

SUMMARY_CSV = (
    RESULTS_DIR
    / "final_multiseed_summary.csv"
)

SUMMARY_MD = (
    RESULTS_DIR
    / "final_multiseed_summary.md"
)


# ============================================================
# GAIN Models
# ============================================================

class Generator(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )

    def forward(self, x_tilde, mask):
        inputs = torch.cat(
            [x_tilde, mask],
            dim=1,
        )

        return self.net(inputs)


class Discriminator(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )

    def forward(self, x_hat, hint):
        inputs = torch.cat(
            [x_hat, hint],
            dim=1,
        )

        return self.net(inputs)


# ============================================================
# Data / Evaluation Utilities
# ============================================================

def create_mcar_mask(shape, seed):
    rng = np.random.default_rng(seed)

    return (
        rng.random(shape)
        >= MISSING_RATE
    )


def prepare_normalized_data(
    x_raw,
    mask,
):
    """
    Normalize using only values that remain observed after masking.
    """

    x_missing_raw = x_raw.copy()
    x_missing_raw[~mask] = np.nan

    col_min = np.nanmin(
        x_missing_raw,
        axis=0,
    )

    col_max = np.nanmax(
        x_missing_raw,
        axis=0,
    )

    scale = col_max - col_min

    scale[
        scale < 1e-8
    ] = 1.0

    x_true_norm = (
        (x_raw - col_min)
        / scale
    )

    x_missing_norm = (
        (x_missing_raw - col_min)
        / scale
    )

    return (
        x_true_norm,
        x_missing_norm,
        col_min,
        scale,
    )


def normalized_to_raw(
    x_imputed_norm,
    x_raw,
    mask,
    col_min,
    scale,
):
    x_imputed_raw = (
        x_imputed_norm
        * scale
        + col_min
    )

    # Preserve observed values exactly.
    x_imputed_raw[mask] = x_raw[mask]

    return x_imputed_raw


def official_normalization(
    data,
    parameters=None,
):
    """
    Replicates the normalization procedure used in
    the authors' official GAIN utils.py.
    """

    norm_data = (
        data
        .copy()
        .astype(np.float64)
    )

    _, dim = norm_data.shape

    if parameters is None:

        min_val = np.zeros(dim)
        max_val = np.zeros(dim)

        for i in range(dim):

            min_val[i] = np.nanmin(
                norm_data[:, i]
            )

            norm_data[:, i] = (
                norm_data[:, i]
                - min_val[i]
            )

            max_val[i] = np.nanmax(
                norm_data[:, i]
            )

            norm_data[:, i] = (
                norm_data[:, i]
                / (
                    max_val[i]
                    + 1e-6
                )
            )

        parameters = {
            "min_val": min_val,
            "max_val": max_val,
        }

    else:

        min_val = parameters["min_val"]
        max_val = parameters["max_val"]

        for i in range(dim):

            norm_data[:, i] = (
                norm_data[:, i]
                - min_val[i]
            )

            norm_data[:, i] = (
                norm_data[:, i]
                / (
                    max_val[i]
                    + 1e-6
                )
            )

    return norm_data, parameters


def author_code_rmse(
    original_raw,
    imputed_raw,
    mask,
):
    """
    RMSE procedure matching the authors' official utility code.
    """

    original_norm, params = (
        official_normalization(
            original_raw
        )
    )

    imputed_norm, _ = (
        official_normalization(
            imputed_raw,
            params,
        )
    )

    missing = ~mask

    error = (
        original_norm[missing]
        - imputed_norm[missing]
    )

    return np.sqrt(
        np.mean(error ** 2)
    )


def sample_batch(
    x,
    mask,
    batch_size,
    rng,
):
    indices = rng.choice(
        len(x),
        size=batch_size,
        replace=False,
    )

    return (
        x[indices],
        mask[indices],
    )


# ============================================================
# Baselines
# ============================================================

def run_knn(
    x_raw,
    x_missing_norm,
    mask,
    col_min,
    scale,
):
    start = perf_counter()

    imputer = KNNImputer(
        n_neighbors=KNN_K,
        weights="uniform",
    )

    x_imputed_norm = (
        imputer.fit_transform(
            x_missing_norm
        )
    )

    runtime = (
        perf_counter()
        - start
    )

    x_imputed_raw = normalized_to_raw(
        x_imputed_norm,
        x_raw,
        mask,
        col_min,
        scale,
    )

    rmse = author_code_rmse(
        x_raw,
        x_imputed_raw,
        mask,
    )

    return rmse, runtime, ""


def run_mice(
    x_raw,
    x_missing_norm,
    mask,
    col_min,
    scale,
    seed,
):
    start = perf_counter()

    imputer = IterativeImputer(
        max_iter=MICE_MAX_ITER,
        random_state=seed,
        initial_strategy="mean",
        skip_complete=True,
    )

    warning_text = ""

    with warnings.catch_warnings(
        record=True
    ) as caught:

        warnings.simplefilter(
            "always",
            ConvergenceWarning,
        )

        x_imputed_norm = (
            imputer.fit_transform(
                x_missing_norm
            )
        )

        convergence_warnings = [
            w
            for w in caught
            if issubclass(
                w.category,
                ConvergenceWarning,
            )
        ]

        if convergence_warnings:
            warning_text = (
                "early_stopping_not_reached"
            )

    runtime = (
        perf_counter()
        - start
    )

    x_imputed_raw = normalized_to_raw(
        x_imputed_norm,
        x_raw,
        mask,
        col_min,
        scale,
    )

    rmse = author_code_rmse(
        x_raw,
        x_imputed_raw,
        mask,
    )

    return (
        rmse,
        runtime,
        warning_text,
    )


# ============================================================
# GAIN
# ============================================================

def run_gain(
    x_raw,
    x_missing_norm,
    mask,
    col_min,
    scale,
    seed,
    device,
    iterations,
):

    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    training_rng = (
        np.random.default_rng(seed)
    )

    eval_seed = (
        EVAL_SEED_BASE
        + seed
    )

    x_observed = np.nan_to_num(
        x_missing_norm,
        nan=0.0,
    ).astype(np.float32)

    dim = x_observed.shape[1]

    generator = (
        Generator(dim)
        .to(device)
    )

    discriminator = (
        Discriminator(dim)
        .to(device)
    )

    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=LEARNING_RATE,
    )

    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=LEARNING_RATE,
    )

    start = perf_counter()

    for iteration in range(
        1,
        iterations + 1,
    ):

        (
            x_batch_np,
            m_batch_np,
        ) = sample_batch(
            x_observed,
            mask,
            BATCH_SIZE,
            training_rng,
        )

        x_batch = torch.tensor(
            x_batch_np,
            dtype=torch.float32,
            device=device,
        )

        m_batch = torch.tensor(
            m_batch_np.astype(
                np.float32
            ),
            dtype=torch.float32,
            device=device,
        )

        # Uniform [0, 0.01] noise.
        z_batch = (
            torch.rand_like(
                x_batch
            )
            * 0.01
        )

        x_tilde = (
            m_batch * x_batch
            + (1.0 - m_batch)
            * z_batch
        )

        # Official-code-oriented hint.
        hint_selector = (
            torch.rand_like(
                m_batch
            )
            < HINT_RATE
        ).float()

        hint = (
            m_batch
            * hint_selector
        )

        # ---------------------------
        # Discriminator
        # ---------------------------

        optimizer_d.zero_grad()

        with torch.no_grad():

            g_sample = generator(
                x_tilde,
                m_batch,
            )

        x_hat = (
            m_batch * x_batch
            + (1.0 - m_batch)
            * g_sample
        )

        d_prob = discriminator(
            x_hat,
            hint,
        )

        eps = 1e-8

        d_loss_matrix = -(
            m_batch
            * torch.log(
                d_prob + eps
            )
            + (1.0 - m_batch)
            * torch.log(
                1.0
                - d_prob
                + eps
            )
        )

        d_loss = (
            d_loss_matrix.mean()
        )

        d_loss.backward()
        optimizer_d.step()

        # ---------------------------
        # Generator
        # ---------------------------

        optimizer_g.zero_grad()

        g_sample = generator(
            x_tilde,
            m_batch,
        )

        x_hat = (
            m_batch * x_batch
            + (1.0 - m_batch)
            * g_sample
        )

        d_prob = discriminator(
            x_hat,
            hint,
        )

        g_adv_loss = -(
            (1.0 - m_batch)
            * torch.log(
                d_prob + eps
            )
        ).mean()

        reconstruction_loss = (
            m_batch
            * (
                x_batch
                - g_sample
            ) ** 2
        ).sum() / (
            m_batch.sum()
            .clamp_min(1.0)
        )

        g_loss = (
            g_adv_loss
            + ALPHA
            * reconstruction_loss
        )

        g_loss.backward()
        optimizer_g.step()

    runtime = (
        perf_counter()
        - start
    )

    # -------------------------------
    # Deterministic inference
    # -------------------------------

    generator.eval()

    eval_generator = (
        torch.Generator(
            device=device.type
        )
    )

    eval_generator.manual_seed(
        eval_seed
    )

    with torch.no_grad():

        x_tensor = torch.tensor(
            x_observed,
            dtype=torch.float32,
            device=device,
        )

        m_tensor = torch.tensor(
            mask.astype(
                np.float32
            ),
            dtype=torch.float32,
            device=device,
        )

        z_tensor = torch.rand(
            x_tensor.shape,
            dtype=x_tensor.dtype,
            device=device,
            generator=eval_generator,
        ) * 0.01

        x_tilde = (
            m_tensor * x_tensor
            + (1.0 - m_tensor)
            * z_tensor
        )

        g_sample = generator(
            x_tilde,
            m_tensor,
        )

        x_imputed_norm = (
            m_tensor * x_tensor
            + (1.0 - m_tensor)
            * g_sample
        )

        x_imputed_norm = (
            x_imputed_norm
            .cpu()
            .numpy()
        )

    x_imputed_raw = normalized_to_raw(
        x_imputed_norm,
        x_raw,
        mask,
        col_min,
        scale,
    )

    rmse = author_code_rmse(
        x_raw,
        x_imputed_raw,
        mask,
    )

    return (
        rmse,
        runtime,
        "",
    )


# ============================================================
# Summary
# ============================================================

def build_summary(df):

    rows = []

    for method, group in df.groupby(
        "method",
        sort=False,
    ):

        finite = group[
            np.isfinite(
                group["rmse"]
            )
        ]

        rows.append(
            {
                "method": method,
                "runs": len(group),
                "finite_runs": len(finite),
                "nan_runs": (
                    len(group)
                    - len(finite)
                ),
                "mean_rmse": (
                    finite["rmse"].mean()
                ),
                "std_rmse": (
                    finite["rmse"].std(
                        ddof=1
                    )
                ),
                "min_rmse": (
                    finite["rmse"].min()
                ),
                "max_rmse": (
                    finite["rmse"].max()
                ),
                "mean_runtime_seconds": (
                    group[
                        "runtime_seconds"
                    ].mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def write_markdown_summary(
    df,
    summary,
    seeds,
    iterations,
):

    text = """# Final Multi-Seed Comparison

## Experimental Protocol

- Dataset: UCI Spambase
- Missingness mechanism: MCAR
- Missing rate: 20%
- Same missingness mask used for KNN, MICE-style, and GAIN within each seed
- Evaluation: author-code-style normalized RMSE
- Methods:
  - KNN, k=5
  - MICE-style IterativeImputer, max_iter=10
  - Official-code-oriented PyTorch GAIN

"""

    text += (
        f"- Seeds: {', '.join(map(str, seeds))}\n"
    )

    text += (
        f"- GAIN iterations: {iterations}\n"
    )

    text += """
## Per-Seed Results

| Seed | Method | RMSE | Runtime (s) | Warning |
|---:|---|---:|---:|---|
"""

    for _, row in df.iterrows():

        rmse_text = (
            "NaN"
            if not np.isfinite(
                row["rmse"]
            )
            else f'{row["rmse"]:.6f}'
        )

        warning_text = (
            row["warning"]
            if row["warning"]
            else "-"
        )

        text += (
            f'| {int(row["seed"])} '
            f'| {row["method"]} '
            f'| {rmse_text} '
            f'| {row["runtime_seconds"]:.2f} '
            f'| {warning_text} |\n'
        )

    text += """
## Aggregate Results

| Method | Finite Runs | NaN Runs | Mean RMSE | Std RMSE | Min RMSE | Max RMSE | Mean Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
"""

    for _, row in summary.iterrows():

        text += (
            f'| {row["method"]} '
            f'| {int(row["finite_runs"])}'
            f'/{int(row["runs"])} '
            f'| {int(row["nan_runs"])} '
            f'| {row["mean_rmse"]:.6f} '
            f'| {row["std_rmse"]:.6f} '
            f'| {row["min_rmse"]:.6f} '
            f'| {row["max_rmse"]:.6f} '
            f'| {row["mean_runtime_seconds"]:.2f} |\n'
        )

    text += """
## Interpretation Rule

All three methods receive the same artificially hidden cells for a given seed.

Therefore, differences between methods within the same seed are directly
comparable under this experiment.

This table is intended to serve as the final fair KNN / MICE / GAIN comparison
for the course report.
"""

    SUMMARY_MD.write_text(text)


# ============================================================
# Main
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(10)),
    )

    parser.add_argument(
        "--gain-iterations",
        type=int,
        default=GAIN_ITERATIONS,
    )

    return parser.parse_args()


def main():

    args = parse_args()

    seeds = args.seeds
    gain_iterations = (
        args.gain_iterations
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset = fetch_ucirepo(
        id=94
    )

    x_raw = (
        dataset.data.features
        .to_numpy(
            dtype=np.float64
        )
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Dataset shape: "
        f"{x_raw.shape}"
    )

    print(
        f"GAIN device: {device}"
    )

    print(
        f"Seeds: {seeds}"
    )

    print(
        f"GAIN iterations: "
        f"{gain_iterations}"
    )

    rows = []

    for seed in seeds:

        print()
        print(
            "=" * 60
        )

        print(
            f"SEED {seed}"
        )

        print(
            "=" * 60
        )

        mask = create_mcar_mask(
            x_raw.shape,
            seed,
        )

        (
            _,
            x_missing_norm,
            col_min,
            scale,
        ) = prepare_normalized_data(
            x_raw,
            mask,
        )

        actual_missing = (
            (~mask).mean()
        )

        print(
            f"Missing rate: "
            f"{actual_missing:.4f}"
        )

        # -------------------------
        # KNN
        # -------------------------

        rmse, runtime, warning = (
            run_knn(
                x_raw,
                x_missing_norm,
                mask,
                col_min,
                scale,
            )
        )

        print(
            f"KNN        "
            f"RMSE={rmse:.6f} "
            f"runtime={runtime:.2f}s"
        )

        rows.append(
            {
                "seed": seed,
                "method": "KNN",
                "rmse": rmse,
                "runtime_seconds":
                    runtime,
                "warning": warning,
                "missing_rate":
                    actual_missing,
            }
        )

        # -------------------------
        # MICE
        # -------------------------

        rmse, runtime, warning = (
            run_mice(
                x_raw,
                x_missing_norm,
                mask,
                col_min,
                scale,
                seed,
            )
        )

        print(
            f"MICE-style "
            f"RMSE={rmse:.6f} "
            f"runtime={runtime:.2f}s "
            f"{warning}"
        )

        rows.append(
            {
                "seed": seed,
                "method": "MICE-style",
                "rmse": rmse,
                "runtime_seconds":
                    runtime,
                "warning": warning,
                "missing_rate":
                    actual_missing,
            }
        )

        # -------------------------
        # GAIN
        # -------------------------

        rmse, runtime, warning = (
            run_gain(
                x_raw,
                x_missing_norm,
                mask,
                col_min,
                scale,
                seed,
                device,
                gain_iterations,
            )
        )

        print(
            f"GAIN       "
            f"RMSE={rmse:.6f} "
            f"runtime={runtime:.2f}s"
        )

        rows.append(
            {
                "seed": seed,
                "method": "GAIN",
                "rmse": rmse,
                "runtime_seconds":
                    runtime,
                "warning": warning,
                "missing_rate":
                    actual_missing,
            }
        )

        # Save after every seed.
        pd.DataFrame(rows).to_csv(
            RESULTS_CSV,
            index=False,
        )

    df = pd.DataFrame(rows)

    summary = build_summary(
        df
    )

    summary.to_csv(
        SUMMARY_CSV,
        index=False,
    )

    write_markdown_summary(
        df,
        summary,
        seeds,
        gain_iterations,
    )

    print()
    print(
        "=" * 60
    )

    print(
        "FINAL SUMMARY"
    )

    print(
        "=" * 60
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        f"Results: {RESULTS_CSV}"
    )

    print(
        f"Summary: {SUMMARY_CSV}"
    )

    print(
        f"Markdown: {SUMMARY_MD}"
    )


if __name__ == "__main__":
    main()
