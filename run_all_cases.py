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


def build_default_configs(prefix: str = "") -> List[SwissRollExperimentConfig]:
    """
    Build a list of experiment configs:
      - one baseline isotropic DDPM (no manifold info)
      - a sweep over tangent / normal raw weights (tangent_fraction / normal_fraction)
        with internal normalization happening inside DiffusionExperiment.
    """
    configs: List[SwissRollExperimentConfig] = []
    for num_epochs in range(5, 16, 5):
        # Baseline: no manifold, standard normal noise
        configs.append(
            SwissRollExperimentConfig(
                name=f"{prefix}baseline_full_normal_epochs{num_epochs}",
                use_manifold=True,
                mixed_noise=True,
                tangent_fraction=0.0,
                normal_fraction=1.0,
                num_epochs=num_epochs,
                num_eval_samples=2000,
            )
        )

        # Baseline: no manifold, standard tangent noise
        configs.append(
            SwissRollExperimentConfig(
                name=f"{prefix}baseline_full_tangent_epochs{num_epochs}",
                use_manifold=True,
                mixed_noise=True,
                tangent_fraction=1.0,
                normal_fraction=0.0,
                num_epochs=num_epochs,
                num_eval_samples=2000,
            )
        )

        # Manifold-aware sweep over tangent/normal weights
        # raw tangent_fraction = r, normal_fraction = 1.0
        # DiffusionExperiment will internally normalize so that
        # E||noise||^2 ~ D for all (r, 1.0).
        ratios = np.linspace(1.0, 10.0, 10)  # 0, 0.5, 1, ..., 4

        for r in ratios:
            name = f"{prefix}tangent_ratio_{r/5:.2f}_epochs{num_epochs}"
            configs.append(
                SwissRollExperimentConfig(
                    name=name,
                    use_manifold=True,
                    mixed_noise=True,
                    tangent_fraction=float(r),
                    normal_fraction=5.0,
                    num_epochs=num_epochs,
                    num_eval_samples=2000,
                )
            )

    return configs


def main(output_dir: str = "outputs/full_swissroll_experiments", num_trials = 10) -> None:
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for embed_dim in range(3, 4, 3): # Will only loop once, bc infrastructure for >3 not working
        # -----------------------------
        # 1. Generate swiss roll data
        # -----------------------------
        print("=== Generating swiss roll data ===")
        data = generate_dataset(
            name="swiss_roll",
            n_samples=10000,
            embed_dim=embed_dim,
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
        configs = build_default_configs(prefix=f"embed_dim_{embed_dim}_")
        print(f"=== Built {len(configs)} experiment configs ===")

        # -----------------------------
        # 4. Run experiments
        # -----------------------------

        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_str)
        print(f"Using device: {device_str}")

        for cfg in configs:
            print(f"\n=== Running experiment: {cfg.name} ===")

            chamfer_scores = []
            swiss_scores = []

            for trial_idx in range(num_trials):

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

                chamfer_scores.append(cd)
                swiss_scores.append(swiss_dist)

            # Collect sample means
            chamfer_mean = np.mean(chamfer_scores)
            swiss_mean = np.mean(swiss_scores)

            # Collect sample stds
            chamfer_std = np.std(chamfer_scores)
            swiss_std = np.std(swiss_scores)

            print(
                f" -> chamfer_mean={chamfer_mean:.4f}, "
                f"chamfer_std={chamfer_std:.4f}, "
                f" swiss_dist={swiss_mean:.4f}, "
                f" swiss_dist={swiss_std:.4f}"
            )

            # Collect result row
            row = asdict(cfg)
            row["chamfer_mean"] = chamfer_mean
            row["swiss_dist_mean"] = swiss_mean
            row["chamfer_std"] = chamfer_std
            row["swiss_dist_std"] = swiss_std
            results.append(row)

    # -----------------------------
    # 7. Save and print summary
    # -----------------------------
    summary_path = os.path.join(output_dir, "summary.csv")
    df = pd.DataFrame(results)
    df.to_csv(summary_path, index=False)

    print("\n=== Summary of experiments ===")
    print(df[["name", "use_manifold", "tangent_fraction", "normal_fraction", "chamfer_mean", "swiss_dist_mean"]])
    print(f"\nSummary written to: {summary_path}")


if __name__ == "__main__":
    main()
