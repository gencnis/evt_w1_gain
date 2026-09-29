from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

SUMMARY_PATH = (
    ROOT
    / "03_experiments"
    / "results"
    / "final_multiseed_summary.csv"
)

FINAL_CSV = (
    ROOT
    / "03_experiments"
    / "results"
    / "final_comparison.csv"
)

FINAL_MD = (
    ROOT
    / "03_experiments"
    / "results"
    / "final_comparison.md"
)

FIGURE_DIR = (
    ROOT
    / "04_figures_tables"
)


def main():
    df = pd.read_csv(SUMMARY_PATH)

    order = [
        "KNN",
        "MICE-style",
        "GAIN",
    ]

    df["method"] = pd.Categorical(
        df["method"],
        categories=order,
        ordered=True,
    )

    df = (
        df.sort_values("method")
        .reset_index(drop=True)
    )

    df.to_csv(
        FINAL_CSV,
        index=False,
    )

    text = """# Final KNN / MICE / GAIN Comparison

## Protocol

- Dataset: UCI Spambase
- Seeds: 0-9
- Missingness: 20% MCAR
- Same missingness mask for all three methods within each seed
- Evaluation: author-code-style normalized RMSE
- KNN: k=5
- MICE-style: max_iter=10
- GAIN: 10000 iterations, batch=128, hint_rate=0.9, alpha=100

## Results

| Method | Mean RMSE | Std RMSE | Min RMSE | Max RMSE | Mean Runtime (s) |
|---|---:|---:|---:|---:|---:|
"""

    for _, row in df.iterrows():
        text += (
            f'| {row["method"]} '
            f'| {row["mean_rmse"]:.6f} '
            f'| {row["std_rmse"]:.6f} '
            f'| {row["min_rmse"]:.6f} '
            f'| {row["max_rmse"]:.6f} '
            f'| {row["mean_runtime_seconds"]:.2f} |\n'
        )

    text += """
## Interpretation

The comparison uses identical seed-specific missingness masks for all three
methods.

MICE-style produced the lowest mean RMSE in this experiment.

GAIN produced a lower mean RMSE than KNN but required the highest runtime.

The MICE max_iter=50 sensitivity test is documented separately and is not used
as the final baseline configuration.
"""

    FINAL_MD.write_text(text)

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # RMSE figure
    plt.figure(figsize=(7, 5))

    plt.bar(
        df["method"].astype(str),
        df["mean_rmse"],
        yerr=df["std_rmse"],
        capsize=5,
    )

    plt.ylabel("RMSE")
    plt.title(
        "Final Missing-Data Imputation Comparison"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR
        / "final_rmse_comparison.png",
        dpi=300,
    )

    plt.close()

    # Runtime figure
    plt.figure(figsize=(7, 5))

    plt.bar(
        df["method"].astype(str),
        df["mean_runtime_seconds"],
    )

    plt.ylabel("Mean runtime (seconds)")
    plt.title(
        "Final Runtime Comparison"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR
        / "final_runtime_comparison.png",
        dpi=300,
    )

    plt.close()

    print(df.to_string(index=False))
    print()
    print(f"Saved: {FINAL_CSV}")
    print(f"Saved: {FINAL_MD}")
    print("Saved final figures.")


if __name__ == "__main__":
    main()
