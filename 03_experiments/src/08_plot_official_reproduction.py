from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

RESULTS_PATH = (
    ROOT
    / "03_experiments"
    / "results"
    / "official_10seed_results.csv"
)

FIGURE_PATH = (
    ROOT
    / "04_figures_tables"
    / "official_10seed_reproduction.png"
)

PAPER_MEAN = 0.0513
PAPER_STD = 0.0016


def main():
    df = pd.read_csv(RESULTS_PATH)

    seeds = df["seed"].to_numpy()
    rmse = df["rmse"].to_numpy()

    finite = np.isfinite(rmse)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.scatter(
        seeds[finite],
        rmse[finite],
        s=60,
        label="Official code runs",
    )

    ax.axhline(
        PAPER_MEAN,
        linestyle="--",
        label="Paper mean RMSE",
    )

    ax.axhspan(
        PAPER_MEAN - PAPER_STD,
        PAPER_MEAN + PAPER_STD,
        alpha=0.15,
        label="Paper mean ± std",
    )

    for seed in seeds[~finite]:
        ax.text(
            seed,
            PAPER_MEAN + PAPER_STD + 0.0003,
            "NaN",
            ha="center",
            va="bottom",
        )

    ax.set_xticks(seeds)
    ax.set_xlabel("Random seed")
    ax.set_ylabel("RMSE")
    ax.set_title(
        "Official GAIN Reproduction Across 10 Random Seeds"
    )

    ax.legend()
    fig.tight_layout()

    FIGURE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        FIGURE_PATH,
        dpi=300,
    )

    plt.close(fig)

    print(f"Saved: {FIGURE_PATH}")


if __name__ == "__main__":
    main()
