# src/experiments_swissroll.py

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional, Iterable, Dict, Any, Tuple

import numpy as np
import torch

from src.diffusion import DiffusionExperiment
from src.metrics import chamfer_distance, swiss_roll_manifold_distance


@dataclass
class SwissRollExperimentConfig:
    """
    Configuration for a single swiss-roll diffusion experiment.
    """
    name: str

    # Manifold-aware vs baseline
    use_manifold: bool = True          # if False, local_frames=None
    mixed_noise: bool = True           # only relevant if use_manifold=True

    # Noise fractions (only matter if use_manifold=True)
    tangent_fraction: float = 1.0
    normal_fraction: float = 1.0

    # Diffusion hyperparams
    num_timesteps: int = 1000
    num_epochs: int = 10
    batch_size: int = 128
    lr: float = 1e-3

    # Evaluation
    num_eval_samples: int = 2000
    max_true_eval: int = 2000
    max_gen_eval: int = 2000

    # Swiss roll manifold params (for known-manifold metric)
    u_range: Tuple[float, float] = (1.5 * np.pi, 4.5 * np.pi)
    v_range: Tuple[float, float] = (0.0, 10.0)
    grid_n_u: int = 200
    grid_n_v: int = 50

    # Reproducibility
    seed: int = 0

    # Device (optional override)
    device: Optional[str] = None  # "cuda", "cpu", or None for auto


def _set_global_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_swiss_roll_experiment(
    X: np.ndarray,
    E: np.ndarray,
    frames,  # LocalFrames or None; we don't import the type here to avoid circular deps
    config: SwissRollExperimentConfig,
) -> Dict[str, Any]:
    """
    Run a single experiment on a fixed swiss-roll dataset.

    Parameters
    ----------
    X : (N, D) np.ndarray
        Training data (e.g., swiss roll points).
    E : (D, 3) np.ndarray
        Embedding matrix used for swiss roll (from generate_dataset).
        Needed for swiss-roll manifold metric.
    frames : LocalFrames or None
        Local tangent/normal frames. Should correspond to X.
        If config.use_manifold is False, this will be ignored.
    config : SwissRollExperimentConfig
        Experiment config.

    Returns
    -------
    result : dict
        Contains:
          - all config fields
          - "chamfer": float
          - "swiss_dist": float
    """
    _set_global_seed(config.seed)

    # Decide device
    if config.device is not None:
        device_str = config.device
    else:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    # Decide whether to pass frames
    local_frames = frames if config.use_manifold else None

    # Instantiate diffusion experiment
    exp = DiffusionExperiment(
        X=X,
        local_frames=local_frames,
        tangent_fraction=config.tangent_fraction,
        normal_fraction=config.normal_fraction,
        mixed_noise=config.mixed_noise and config.use_manifold,
        num_timesteps=config.num_timesteps,
        batch_size=config.batch_size,
        lr=config.lr,
        device=device_str,
    )

    # Train
    exp.diffusion_train(num_epochs=config.num_epochs)

    # Sample
    samples = exp.sample(num_samples=config.num_eval_samples)  # (N_gen, D) tensor
    samples_np = samples.detach().cpu().numpy()

    # Metrics
    cd = chamfer_distance(
        X_true=X,
        X_gen=samples_np,
        device=device,
        max_true=config.max_true_eval,
        max_gen=config.max_gen_eval,
    )

    swiss_dist = swiss_roll_manifold_distance(
        X_gen=samples_np,
        embed_dim=X.shape[1],
        u_range=config.u_range,
        v_range=config.v_range,
        n_u=config.grid_n_u,
        n_v=config.grid_n_v,
        E=E,
        device=device,
        max_gen=config.max_gen_eval,
    )

    # Build result dict: config fields + metrics
    result: Dict[str, Any] = asdict(config)
    result["chamfer"] = cd
    result["swiss_dist"] = swiss_dist

    return result


def run_swiss_roll_experiment_grid(
    X: np.ndarray,
    E: np.ndarray,
    frames,
    configs: Iterable[SwissRollExperimentConfig],
) -> List[Dict[str, Any]]:
    """
    Run a grid of experiments on the same swiss-roll dataset + frames.

    Parameters
    ----------
    X : (N, D) np.ndarray
        Training data.
    E : (D, 3) np.ndarray
        Embedding matrix from generate_dataset.
    frames : LocalFrames or None
        Local tangent/normal frames for X. Can be reused across experiments.
    configs : iterable of SwissRollExperimentConfig
        Experiment configurations to run.

    Returns
    -------
    results : list of dict
        List of result dicts from run_swiss_roll_experiment.
    """
    results: List[Dict[str, Any]] = []
    for cfg in configs:
        print(f"\n=== Running experiment: {cfg.name} ===")
        res = run_swiss_roll_experiment(X=X, E=E, frames=frames, config=cfg)
        print(
            f" -> chamfer={res['chamfer']:.4f}, "
            f"swiss_dist={res['swiss_dist']:.4f}"
        )
        results.append(res)
    return results
