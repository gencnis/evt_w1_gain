from pathlib import Path
from time import perf_counter
import argparse
import csv

import numpy as np
import torch
import torch.nn as nn


SEED = 42
EVAL_SEED = 2026

BATCH_SIZE = 128
HINT_RATE = 0.90
ALPHA = 100.0
LEARNING_RATE = 1e-3

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT
    / "03_experiments"
    / "data"
    / f"spambase_mcar20_seed{SEED}.npz"
)

RESULTS_DIR = (
    ROOT
    / "03_experiments"
    / "results"
)


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
        inputs = torch.cat([x_tilde, mask], dim=1)
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
        inputs = torch.cat([x_hat, hint], dim=1)
        return self.net(inputs)


def masked_rmse(x_true, x_imputed, mask):
    missing = ~mask

    squared_error = (
        x_true[missing] - x_imputed[missing]
    ) ** 2

    return np.sqrt(np.mean(squared_error))


def sample_batch(x, mask, batch_size, rng):
    indices = rng.choice(
        len(x),
        size=batch_size,
        replace=False,
    )

    return x[indices], mask[indices]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--iterations",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--experiment-id",
        type=str,
        required=True,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    iterations = args.iterations
    experiment_id = args.experiment_id

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Experiment: {experiment_id}")
    print(f"Device: {device}")
    print(f"Training seed: {SEED}")
    print(f"Evaluation seed: {EVAL_SEED}")

    data = np.load(DATA_PATH)

    x_true = data["X_true"].astype(np.float32)
    x_missing = data["X_missing"].astype(np.float32)
    mask = data["mask"].astype(bool)

    print(f"Shape: {x_true.shape}")
    print(f"Missing rate: {(~mask).mean():.4f}")

    x_observed = np.nan_to_num(
        x_missing,
        nan=0.0,
    )

    dim = x_true.shape[1]

    generator = Generator(dim).to(device)
    discriminator = Discriminator(dim).to(device)

    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=LEARNING_RATE,
    )

    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=LEARNING_RATE,
    )

    rng = np.random.default_rng(SEED)

    loss_history = []

    start_time = perf_counter()

    for iteration in range(1, iterations + 1):

        x_batch_np, m_batch_np = sample_batch(
            x_observed,
            mask,
            BATCH_SIZE,
            rng,
        )

        x_batch = torch.tensor(
            x_batch_np,
            dtype=torch.float32,
            device=device,
        )

        m_batch = torch.tensor(
            m_batch_np,
            dtype=torch.float32,
            device=device,
        )

        z_batch = torch.rand_like(x_batch) * 0.01

        x_tilde = (
            m_batch * x_batch
            + (1.0 - m_batch) * z_batch
        )

        b_batch = (
            torch.rand_like(m_batch) < HINT_RATE
        ).float()

        hint = (
            b_batch * m_batch
            + 0.5 * (1.0 - b_batch)
        )

        uncertain = 1.0 - b_batch

        # -------------------------
        # Discriminator
        # -------------------------

        optimizer_d.zero_grad()

        with torch.no_grad():
            g_sample = generator(
                x_tilde,
                m_batch,
            )

        x_hat = (
            m_batch * x_batch
            + (1.0 - m_batch) * g_sample
        )

        d_prob = discriminator(
            x_hat,
            hint,
        )

        eps = 1e-8

        d_loss_matrix = -(
            m_batch * torch.log(d_prob + eps)
            + (1.0 - m_batch)
            * torch.log(1.0 - d_prob + eps)
        )

        d_loss = (
            uncertain * d_loss_matrix
        ).sum() / uncertain.sum().clamp_min(1.0)

        d_loss.backward()
        optimizer_d.step()

        # -------------------------
        # Generator
        # -------------------------

        optimizer_g.zero_grad()

        g_sample = generator(
            x_tilde,
            m_batch,
        )

        x_hat = (
            m_batch * x_batch
            + (1.0 - m_batch) * g_sample
        )

        d_prob = discriminator(
            x_hat,
            hint,
        )

        missing_uncertain = (
            uncertain
            * (1.0 - m_batch)
        )

        g_adv_loss = -(
            missing_uncertain
            * torch.log(d_prob + eps)
        ).sum() / missing_uncertain.sum().clamp_min(1.0)

        reconstruction_loss = (
            m_batch
            * (x_batch - g_sample) ** 2
        ).sum() / m_batch.sum().clamp_min(1.0)

        g_loss = (
            g_adv_loss
            + ALPHA * reconstruction_loss
        )

        g_loss.backward()
        optimizer_g.step()

        if iteration == 1 or iteration % 200 == 0:

            row = {
                "iteration": iteration,
                "d_loss": d_loss.item(),
                "g_adversarial_loss": g_adv_loss.item(),
                "reconstruction_loss": reconstruction_loss.item(),
            }

            loss_history.append(row)

            print(
                f"Iteration {iteration:5d}/{iterations} | "
                f"D loss: {d_loss.item():.4f} | "
                f"G adv: {g_adv_loss.item():.4f} | "
                f"Recon: {reconstruction_loss.item():.6f}"
            )

    training_time = perf_counter() - start_time

    # ------------------------------------------------------
    # Deterministic evaluation
    # ------------------------------------------------------

    generator.eval()

    eval_generator = torch.Generator(
        device=device.type
    )

    eval_generator.manual_seed(EVAL_SEED)

    with torch.no_grad():

        x_tensor = torch.tensor(
            x_observed,
            dtype=torch.float32,
            device=device,
        )

        m_tensor = torch.tensor(
            mask.astype(np.float32),
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
            + (1.0 - m_tensor) * z_tensor
        )

        g_sample = generator(
            x_tilde,
            m_tensor,
        )

        x_imputed = (
            m_tensor * x_tensor
            + (1.0 - m_tensor) * g_sample
        )

        x_imputed = (
            x_imputed
            .cpu()
            .numpy()
        )

    rmse = masked_rmse(
        x_true,
        x_imputed,
        mask,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save training history
    history_path = (
        RESULTS_DIR
        / f"{experiment_id}_gain_loss_history.csv"
    )

    with open(history_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "iteration",
                "d_loss",
                "g_adversarial_loss",
                "reconstruction_loss",
            ],
        )

        writer.writeheader()
        writer.writerows(loss_history)

    # Save run summary
    summary_path = (
        RESULTS_DIR
        / f"{experiment_id}_gain_summary.csv"
    )

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "experiment",
                "iterations",
                "training_seed",
                "evaluation_seed",
                "batch_size",
                "hint_rate",
                "alpha",
                "learning_rate",
                "rmse",
                "training_time_seconds",
                "device",
            ],
        )

        writer.writeheader()

        writer.writerow(
            {
                "experiment": experiment_id,
                "iterations": iterations,
                "training_seed": SEED,
                "evaluation_seed": EVAL_SEED,
                "batch_size": BATCH_SIZE,
                "hint_rate": HINT_RATE,
                "alpha": ALPHA,
                "learning_rate": LEARNING_RATE,
                "rmse": rmse,
                "training_time_seconds": training_time,
                "device": str(device),
            }
        )

    print()
    print("----- GAIN RESULT -----")
    print(f"Experiment: {experiment_id}")
    print(f"GAIN RMSE: {rmse:.6f}")
    print(
        f"GAIN training time: "
        f"{training_time:.2f} seconds"
    )
    print(f"Loss history: {history_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

