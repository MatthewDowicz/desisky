# SPDX-FileCopyrightText: 2025-present MatthewDowicz <mjdowicz@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Weights & Biases integration utilities for desisky training.

Provides a thin abstraction layer over wandb for experiment tracking.
All functions gracefully handle the case where wandb is not installed
(``pip install desisky[wandb]``).

Typical usage
-------------
::

    from desisky.training.wandb_utils import WandbConfig, init_wandb_run

    wconfig = WandbConfig(project="desisky-vae", run_name="my-experiment")
    run = init_wandb_run(wconfig, training_config)
    # ... training loop ...
    log_epoch_metrics({"loss": 0.5}, epoch=10, prefix="train/")
    finish_wandb_run()
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

# ---------------------------------------------------------------------------
# Optional wandb import — module-level so it can be mocked in tests
# ---------------------------------------------------------------------------

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------


def _check_wandb_available() -> None:
    """Raise ``ImportError`` with install instructions if wandb is missing."""
    if wandb is None:
        raise ImportError(
            "wandb is required for experiment tracking. "
            "Install with: pip install desisky[wandb]"
        ) from None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class WandbConfig:
    """Configuration for Weights & Biases experiment tracking.

    Parameters
    ----------
    project : str
        W&B project name.
    entity : str | None
        W&B entity (team or username). If ``None``, uses the default
        entity associated with the logged-in account.
    run_name : str | None
        Display name for the run. If ``None``, W&B auto-generates one
        (e.g. ``"sweet-dawn-42"``).
    tags : list[str]
        Tags for filtering runs on the dashboard. Examples:
        ``["vae", "dark-time", "infovae"]``, ``["sweep", "lr-search"]``,
        ``["v0.5", "production"]``.
    log_every : int
        Log scalar metrics every *N* epochs.
    viz_every : int
        Log visualization figures every *N* epochs (only relevant when
        ``on_epoch_end`` callbacks produce figures).
    extra_config : dict | None
        Additional key-value pairs merged into the ``wandb.init(config=...)``
        dictionary.  Useful for recording dataset metadata or hardware info,
        e.g. ``{"dataset_version": "v1.0", "n_training_samples": 8258}``.
    """

    project: str = "desisky"
    entity: Optional[str] = None
    run_name: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    log_every: int = 1
    viz_every: int = 10
    extra_config: Optional[dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Run lifecycle
# ---------------------------------------------------------------------------


def init_wandb_run(
    wandb_config: WandbConfig,
    training_config: Any,
    model_config: Optional[dict[str, Any]] = None,
) -> Any:
    """Initialize (or reuse) a W&B run.

    **Sweep-aware**: if ``wandb.run`` is already active (e.g. inside a
    ``wandb.agent`` callback), the existing run is returned without
    calling ``wandb.init`` again.

    Parameters
    ----------
    wandb_config : WandbConfig
        W&B-specific settings (project, entity, tags, ...).
    training_config : dataclass
        Training configuration (``VAETrainingConfig`` or
        ``LDMTrainingConfig``).  Converted to a dict via
        ``dataclasses.asdict()`` and stored under ``config["training"]``
        in the W&B run.
    model_config : dict | None
        Model architecture parameters (e.g.
        ``{"in_ch": 1, "hidden": 64, "levels": 3}``).
        Stored under ``config["model"]``.

    Returns
    -------
    run
        The active wandb run object (``wandb.sdk.wandb_run.Run``).
    """
    _check_wandb_available()

    # If a run is already active (e.g. inside wandb.agent), reuse it
    if wandb.run is not None:
        return wandb.run

    # Build config dict from training dataclass
    config: dict[str, Any] = {
        "training": asdict(training_config),
    }

    if model_config is not None:
        config["model"] = model_config

    if wandb_config.extra_config is not None:
        config.update(wandb_config.extra_config)

    run = wandb.init(
        project=wandb_config.project,
        entity=wandb_config.entity,
        name=wandb_config.run_name,
        tags=wandb_config.tags,
        config=config,
    )

    return run


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def log_epoch_metrics(
    metrics: dict[str, float],
    epoch: int,
    prefix: str = "",
) -> None:
    """Log scalar metrics to the active W&B run.

    Parameters
    ----------
    metrics : dict[str, float]
        Metric name -> value pairs.
    epoch : int
        Current epoch number (used as the ``step``).
    prefix : str
        Prefix prepended to each metric name.  For example, ``"train/"``
        produces keys like ``"train/loss"``.
    """
    _check_wandb_available()

    if wandb.run is None:
        return

    payload = {f"{prefix}{k}": v for k, v in metrics.items()}
    payload["epoch"] = epoch
    wandb.log(payload, step=epoch)


def log_figure(
    key: str,
    figure: Figure,
    epoch: int,
) -> None:
    """Log a matplotlib figure as an image to the active W&B run.

    The figure is converted to a PNG via ``wandb.Image``.

    Parameters
    ----------
    key : str
        W&B log key (e.g. ``"val/cdf_broadband"``).
    figure : matplotlib.figure.Figure
        The figure to log.
    epoch : int
        Current epoch number (used as the ``step``).
    """
    _check_wandb_available()

    if wandb.run is None:
        return

    wandb.log({key: wandb.Image(figure)}, step=epoch)


def finish_wandb_run() -> None:
    """Finish the active W&B run.  Safe to call if no run is active."""
    if wandb is None:
        return

    if wandb.run is not None:
        wandb.finish()


# ---------------------------------------------------------------------------
# Per-sigma loss evaluation (LDM-specific)
# ---------------------------------------------------------------------------

#: Default sigma levels for per-sigma loss tracking.
DEFAULT_SIGMA_LEVELS: list[float] = [0.01, 0.1, 0.5, 1.0, 5.0, 20.0]


def eval_per_sigma_losses(
    model: Any,
    data_loader: Any,
    sigma_data: float,
    sigma_levels: Optional[list[float]] = None,
    key_seed: int = 0,
) -> dict[str, float]:
    """Compute EDM loss at fixed sigma levels over a data loader.

    This is useful for diagnosing which noise levels are hardest for the
    model.  Results are returned as a dict suitable for passing to
    :func:`log_epoch_metrics`.

    Parameters
    ----------
    model : eqx.Module
        The LDM model (raw F_theta network).
    data_loader : DataLoader
        Data loader yielding ``(latents, conditioning)`` batches.
    sigma_data : float
        Training data standard deviation (EDM parameter).
    sigma_levels : list[float] | None
        Sigma values to evaluate at.  Defaults to
        ``[0.01, 0.1, 0.5, 1.0, 5.0, 20.0]``.
    key_seed : int
        Seed for the PRNG key.

    Returns
    -------
    losses : dict[str, float]
        Keys like ``"loss_sigma_0.01"`` mapping to the average loss at
        each sigma level.
    """
    import jax
    import jax.numpy as jnp
    import jax.random as jr

    from desisky.models.ldm import edm_denoiser

    if sigma_levels is None:
        sigma_levels = list(DEFAULT_SIGMA_LEVELS)

    key = jr.PRNGKey(key_seed)
    n_sigmas = len(sigma_levels)
    total_loss = np.zeros(n_sigmas)
    total_count = 0

    for x, cond in data_loader:
        if not isinstance(x, jnp.ndarray):
            x = jnp.asarray(x)
        if not isinstance(cond, jnp.ndarray):
            cond = jnp.asarray(cond)

        B = x.shape[0]
        key, k_eps = jr.split(key)
        noise = jax.random.normal(k_eps, shape=x.shape)

        for i, sigma_val in enumerate(sigma_levels):
            sigma = jnp.full((B,), sigma_val)
            x_noisy = x + sigma[:, None, None] * noise

            # No CFG dropout during evaluation
            drop_keys = jr.split(jr.PRNGKey(0), B)
            D_out = jax.vmap(
                edm_denoiser, in_axes=(None, 0, 0, 0, None, 0, None)
            )(model, x_noisy, sigma, cond, sigma_data, drop_keys, 0.0)

            sq_error = jnp.mean((D_out - x) ** 2, axis=(1, 2))
            weights = (sigma_val**2 + sigma_data**2) / (
                sigma_val * sigma_data
            ) ** 2
            loss_val = float(jnp.mean(weights * sq_error))
            total_loss[i] += loss_val * B

        total_count += B

    return {
        f"loss_sigma_{s:.2f}": total_loss[i] / total_count
        for i, s in enumerate(sigma_levels)
    }
