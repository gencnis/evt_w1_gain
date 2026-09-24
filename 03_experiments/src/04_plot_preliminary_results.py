from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

RESULTS_PATH = (
    ROOT
    / "03_experiments"
    / "results"
    / "preliminary_results.csv"
)

FIGURE_DIR = ROOT / "04_figures_tables"


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RESULTS_PATH)

    labels = [
        "KNN",
        "MICE-style",
        "GAIN\n2k",
        "GAIN\n10k",
    ]

    # RMSE comparison
    plt.figure(figsize=(7, 5))
    plt.bar(labels, df["rmse"])
    plt.ylabel("RMSE")
    plt.title("Preliminary Imputation RMSE Comparison")
    plt.tight_layout()

    rmse_path = FIGURE_DIR / "preliminary_rmse_comparison.png"
    plt.savefig(rmse_path, dpi=300)
    plt.close()

    # Runtime comparison
    plt.figure(figsize=(7, 5))
    plt.bar(labels, df["runtime_seconds"])
    plt.ylabel("Runtime (seconds)")
    plt.title("Preliminary Runtime Comparison")
    plt.tight_layout()

    runtime_path = FIGURE_DIR / "preliminary_runtime_comparison.png"
    plt.savefig(runtime_path, dpi=300)
    plt.close()

    print(f"Saved: {rmse_path}")
    print(f"Saved: {runtime_path}")


if __name__ == "__main__":
    main()
