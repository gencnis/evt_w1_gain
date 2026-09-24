from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = ROOT / "03_experiments" / "results"
FIGURE_DIR = ROOT / "04_figures_tables"


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    results = pd.read_csv(
        RESULTS_DIR / "preliminary_results.csv"
    )

    controlled = results[
        results["experiment"].isin(["E1", "E2", "E5", "E6"])
    ].copy()

    labels = [
        "KNN",
        "MICE-style",
        "GAIN 2k",
        "GAIN 10k",
    ]

    # RMSE
    plt.figure(figsize=(7, 5))
    plt.bar(labels, controlled["rmse"])
    plt.ylabel("RMSE")
    plt.title("Controlled Imputation RMSE Comparison")
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "controlled_rmse_comparison.png",
        dpi=300,
    )
    plt.close()

    # Runtime
    plt.figure(figsize=(7, 5))
    plt.bar(labels, controlled["runtime_seconds"])
    plt.ylabel("Runtime (seconds)")
    plt.title("Controlled Runtime Comparison")
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "controlled_runtime_comparison.png",
        dpi=300,
    )
    plt.close()

    # GAIN reconstruction loss
    e5 = pd.read_csv(
        RESULTS_DIR / "E5_gain_loss_history.csv"
    )

    e6 = pd.read_csv(
        RESULTS_DIR / "E6_gain_loss_history.csv"
    )

    plt.figure(figsize=(8, 5))
    plt.plot(
        e5["iteration"],
        e5["reconstruction_loss"],
        label="GAIN 2k",
    )
    plt.plot(
        e6["iteration"],
        e6["reconstruction_loss"],
        label="GAIN 10k",
    )
    plt.xlabel("Training iteration")
    plt.ylabel("Reconstruction loss")
    plt.title("GAIN Reconstruction Loss During Training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "gain_reconstruction_loss.png",
        dpi=300,
    )
    plt.close()

    print("Saved controlled comparison figures.")


if __name__ == "__main__":
    main()
