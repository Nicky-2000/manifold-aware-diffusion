# scripts/run_swissroll_experiments.py

from __future__ import annotations

import os
import json
from dataclasses import asdict
from typing import List

import numpy as np
import pandas as pd
import torch

from src.generate_data import generate_dataset
from src.manifold_learning import estimate_tangent_normal_knn_pca
from src.diffusion import DiffusionExperiment
from src.metrics import chamfer_distance, swiss_roll_manifold_distance
from src.experiments_swissroll import SwissRollExperimentConfig


def set_global_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_default_configs() -> List[SwissRollExperimentConfig]:
    """
    Build a list of experiment configs:
      - one baseline isotropic DDPM (no manifold info)
      - a sweep over tangent / normal raw weights (tangent_fraction / normal_fraction)
        with internal normalization happening inside DiffusionExperiment.
    """
    configs: List[SwissRollExperimentConfig] = []

    # Baseline: no manifold, standard isotropic noise
    configs.append(
        SwissRollExperimentConfig(
            name="baseline_isotropic",
            use_manifold=False,
            mixed_noise=False,
            tangent_fraction=0.0,
            normal_fraction=1.0,
            num_epochs=5,
            num_eval_samples=2000,
        )
    )

    # Manifold-aware sweep over tangent/normal weights
    # raw tangent_fraction = r, normal_fraction = 1.0
    # DiffusionExperiment will internally normalize so that
    # E||noise||^2 ~ D for all (r, 1.0).
    ratios = np.linspace(0.0, 4.0, 9)  # 0, 0.5, 1, ..., 4

    for r in ratios:
        name = f"tangent_ratio_{r:.2f}"
        configs.append(
            SwissRollExperimentConfig(
                name=name,
                use_manifold=True,
                mixed_noise=True,
                tangent_fraction=float(r),
                normal_fraction=1.0,
                num_epochs=5,
                num_eval_samples=2000,
            )
        )

    return configs


def main(output_dir: str = "outputs/swissroll_experiments") -> None:
    os.makedirs(output_dir, exist_ok=True)

    # -----------------------------
    # 1. Generate swiss roll data
    # -----------------------------
    print("=== Generating swiss roll data ===")
    data = generate_dataset(
        name="swiss_roll",
        n_samples=5000,
        embed_dim=3,
        noise_sigma=0.0,
        random_state=0,
    )
    X = data["X"]        # (N, 3)
    E = data["E"]        # (3, 3) embedding matrix used

    # -----------------------------
    # 2. Estimate local frames
    # -----------------------------
    print("=== Estimating local tangent/normal frames via KNN+PCA ===")
    frames = estimate_tangent_normal_knn_pca(
        X,
        intrinsic_dim=2,
        n_neighbors=32,
        include_self=True,
    )

    # -----------------------------
    # 3. Build experiment configs
    # -----------------------------
    configs = build_default_configs()
    print(f"=== Built {len(configs)} experiment configs ===")

    # -----------------------------
    # 4. Run experiments
    # -----------------------------
    results = []

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Using device: {device_str}")

    for cfg in configs:
        print(f"\n=== Running experiment: {cfg.name} ===")
        set_global_seed(cfg.seed)

        # Decide whether this run uses manifold frames
        local_frames = frames if cfg.use_manifold else None

        # Instantiate diffusion experiment
        exp = DiffusionExperiment(
            X=X,
            local_frames=local_frames,
            tangent_fraction=cfg.tangent_fraction,
            normal_fraction=cfg.normal_fraction,
            mixed_noise=cfg.mixed_noise and cfg.use_manifold,
            num_timesteps=cfg.num_timesteps,
            batch_size=cfg.batch_size,
            lr=cfg.lr,
            device=device_str,
        )

        # Train
        exp.diffusion_train(num_epochs=cfg.num_epochs)

        # Sample
        samples = exp.sample(num_samples=cfg.num_eval_samples)  # (N_gen, D) tensor
        samples_np = samples.detach().cpu().numpy()

        # -----------------------------
        # 5. Metrics
        # -----------------------------
        cd = chamfer_distance(
            X_true=X,
            X_gen=samples_np,
            device=device,
            max_true=cfg.max_true_eval,
            max_gen=cfg.max_gen_eval,
        )

        swiss_dist = swiss_roll_manifold_distance(
            X_gen=samples_np,
            embed_dim=X.shape[1],
            u_range=cfg.u_range,
            v_range=cfg.v_range,
            n_u=cfg.grid_n_u,
            n_v=cfg.grid_n_v,
            E=E,
            device=device,
            max_gen=cfg.max_gen_eval,
        )

        print(
            f" -> chamfer={cd:.4f}, "
            f"swiss_dist={swiss_dist:.4f}"
        )

        # Collect result row
        row = asdict(cfg)
        row["chamfer"] = cd
        row["swiss_dist"] = swiss_dist
        results.append(row)

        # -----------------------------
        # 6. Save per-run artifacts
        # -----------------------------
        run_dir = os.path.join(output_dir, cfg.name)
        os.makedirs(run_dir, exist_ok=True)

        # Save model weights
        model_path = os.path.join(run_dir, "model.pt")
        torch.save(exp.model.state_dict(), model_path)

        # Save generated samples
        samples_path = os.path.join(run_dir, "samples.npy")
        np.save(samples_path, samples_np)

        # Save config as JSON
        config_path = os.path.join(run_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(asdict(cfg), f, indent=2)

    # -----------------------------
    # 7. Save and print summary
    # -----------------------------
    summary_path = os.path.join(output_dir, "summary.csv")
    df = pd.DataFrame(results)
    df.to_csv(summary_path, index=False)

    print("\n=== Summary of experiments ===")
    print(df[["name", "use_manifold", "tangent_fraction", "normal_fraction", "chamfer", "swiss_dist"]])
    print(f"\nSummary written to: {summary_path}")


if __name__ == "__main__":
    main()
