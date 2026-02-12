# SPDX-FileCopyrightText: 2025-present MatthewDowicz <mjdowicz@gmail.com>
#
# SPDX-License-Identifier: MIT

"""
Latent Diffusion Model (LDM) training utilities.

This module provides a flexible trainer for conditional latent diffusion models
that can be used with different conditioning variables (dark-time, twilight, moon).

Training uses the EDM framework (Karras et al. 2022):
- Log-normal noise distribution for sigma sampling
- Preconditioned denoiser: D(x; sigma) = c_skip * x + c_out * F_theta(c_in * x; c_noise)
- EDM-weighted loss: lambda(sigma) * ||D(x + sigma*n; sigma) - x||^2
- Exponential Moving Average (EMA) of model weights for stable sampling
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from desisky.io import save
from desisky.models.ldm import (
    edm_denoiser,
    EDM_P_MEAN,
    EDM_P_STD,
)


# ============================================================================
# Conditioning Normalization Utilities
# ============================================================================


def fit_conditioning_scaler(
    conditioning: np.ndarray,
    columns: list[str],
) -> dict:
    """Compute per-feature mean and standard deviation for conditioning normalization.

    Fits a zero-mean, unit-variance scaler on the provided conditioning array
    (typically the **training** split only, to avoid data leakage).  The returned
    dictionary is compatible with :class:`LatentDiffusionSampler`'s
    ``conditioning_scaler`` parameter and is stored in model checkpoint metadata
    so that inference can automatically reproduce the same normalization.

    Parameters
    ----------
    conditioning : np.ndarray
        Conditioning feature matrix, shape ``(n_samples, n_features)``.
        Should contain only the training split.
    columns : list[str]
        Feature names corresponding to each column (e.g.
        ``["OBSALT", "TRANSPARENCY_GFA", ...]``).

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"mean"`` — per-feature means (list of float)
        - ``"scale"`` — per-feature standard deviations (list of float)
        - ``"columns"`` — feature names (list of str)

    Examples
    --------
    >>> scaler = fit_conditioning_scaler(cond_train, ["OBSALT", "SUNALT"])
    >>> scaler["mean"]   # [62.35, -45.40]
    >>> scaler["scale"]  # [11.64, 16.19]
    """
    mean = conditioning.mean(axis=0)
    scale = conditioning.std(axis=0)
    return {
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "columns": list(columns),
    }


def normalize_conditioning(
    conditioning: np.ndarray,
    scaler: dict,
) -> np.ndarray:
    """Apply zero-mean, unit-variance normalization to conditioning features.

    Uses the scaler parameters from :func:`fit_conditioning_scaler` to
    transform conditioning features via ``(x - mean) / scale``.  This is
    the same transformation that :class:`LatentDiffusionSampler` applies
    internally at inference time when ``conditioning_scaler`` is provided.

    Parameters
    ----------
    conditioning : np.ndarray
        Conditioning feature matrix, shape ``(n_samples, n_features)``.
    scaler : dict
        Scaler dictionary with ``"mean"`` and ``"scale"`` keys, as
        returned by :func:`fit_conditioning_scaler`.

    Returns
    -------
    np.ndarray
        Normalized conditioning array, same shape and dtype as input.

    Examples
    --------
    >>> scaler = fit_conditioning_scaler(cond_train, COLS)
    >>> cond_train_norm = normalize_conditioning(cond_train, scaler)
    >>> cond_val_norm = normalize_conditioning(cond_val, scaler)
    """
    mean = np.array(scaler["mean"])
    scale = np.array(scaler["scale"])
    return (conditioning - mean) / scale


# ============================================================================
# EDM Noise Distribution (Karras et al. 2022)
# ============================================================================


def sample_edm_sigma(
    key: jax.random.PRNGKey,
    shape: tuple,
    p_mean: float = EDM_P_MEAN,
    p_std: float = EDM_P_STD,
) -> jnp.ndarray:
    """
    Sample noise levels from EDM log-normal distribution.

    ln(sigma) ~ N(P_mean, P_std^2)  =>  sigma = exp(P_mean + P_std * z), z ~ N(0,1)

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key.
    shape : tuple
        Output shape.
    p_mean : float
        Mean of log-normal distribution.
    p_std : float
        Std of log-normal distribution.

    Returns
    -------
    jnp.ndarray
        Sampled sigma values.
    """
    z = jax.random.normal(key, shape)
    sigma = jnp.exp(p_mean + p_std * z)
    return sigma


def edm_loss_weight(sigma: jnp.ndarray, sigma_data: float) -> jnp.ndarray:
    """
    EDM loss weighting: lambda(sigma) = (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2.

    Parameters
    ----------
    sigma : jnp.ndarray
        Noise levels.
    sigma_data : float
        Standard deviation of training data.

    Returns
    -------
    jnp.ndarray
        Loss weights.
    """
    return (sigma**2 + sigma_data**2) / (sigma * sigma_data) ** 2


# ============================================================================
# EDM Loss Function (Karras et al. 2022)
# ============================================================================


def edm_loss(
    model: eqx.Module,
    x: jnp.ndarray,
    cond: jnp.ndarray,
    sigma_data: float,
    key: jax.random.PRNGKey,
    cfg_dropout_p: float = 0.1,
    p_mean: float = EDM_P_MEAN,
    p_std: float = EDM_P_STD,
) -> jnp.ndarray:
    """
    EDM loss with preconditioning (Karras et al. 2022).

    Trains the model to denoise data corrupted at various noise levels.
    The preconditioned denoiser D predicts the clean data x from noisy
    input (x + sigma * noise), weighted by lambda(sigma) so that all
    noise levels contribute equally to learning.

    Parameters
    ----------
    model : eqx.Module
        Raw UNet network F_theta (not the preconditioned denoiser).
    x : jnp.ndarray
        Clean latent samples, shape (batch, channels, latent_dim).
    cond : jnp.ndarray
        Conditioning metadata, shape (batch, meta_dim).
    sigma_data : float
        Standard deviation of training data. Compute with
        ``compute_sigma_data(training_latents)``.
    key : jax.random.PRNGKey
        Random key.
    cfg_dropout_p : float
        CFG dropout probability.
    p_mean : float
        Mean of log-normal noise distribution.
    p_std : float
        Std of log-normal noise distribution.

    Returns
    -------
    jnp.ndarray
        Scalar weighted EDM loss.
    """
    key, k_sigma, k_eps = jr.split(key, 3)
    B = x.shape[0]

    # Sample sigma from log-normal distribution
    sigma = sample_edm_sigma(k_sigma, (B,), p_mean, p_std)

    # Sample noise
    noise = jax.random.normal(k_eps, shape=x.shape)

    # EDM forward diffusion: x_noisy = x + sigma * noise
    x_noisy = x + sigma[:, None, None] * noise

    # Generate dropout keys for CFG
    drop_keys = jr.split(key, B)

    # Apply preconditioned denoiser (vectorized over batch)
    D_out = jax.vmap(edm_denoiser, in_axes=(None, 0, 0, 0, None, 0, None))(
        model, x_noisy, sigma, cond, sigma_data, drop_keys, cfg_dropout_p
    )

    # Per-sample squared error (target is clean data x)
    sq_error = jnp.mean((D_out - x) ** 2, axis=(1, 2))  # (B,)

    # Apply EDM loss weighting
    weights = edm_loss_weight(sigma, sigma_data)  # (B,)

    # Weighted mean loss
    loss = jnp.mean(weights * sq_error)
    return loss


# ============================================================================
# EMA Utilities
# ============================================================================


def ema_update(
    ema_model: eqx.Module, model: eqx.Module, decay: float
) -> eqx.Module:
    """
    Update EMA model weights: ema = decay * ema + (1 - decay) * model.

    Parameters
    ----------
    ema_model : eqx.Module
        Current EMA model.
    model : eqx.Module
        Current training model.
    decay : float
        EMA decay rate (e.g., 0.9999).

    Returns
    -------
    eqx.Module
        Updated EMA model.
    """
    ema_params, ema_static = eqx.partition(ema_model, eqx.is_array)
    model_params, _ = eqx.partition(model, eqx.is_array)

    new_ema_params = jax.tree.map(
        lambda e, m: decay * e + (1 - decay) * m,
        ema_params,
        model_params,
    )

    return eqx.combine(new_ema_params, ema_static)


@eqx.filter_jit
def ema_update_jit(
    ema_model: eqx.Module, model: eqx.Module, decay: float
) -> eqx.Module:
    """JIT-compiled version of ``ema_update``."""
    return ema_update(ema_model, model, decay)


# ============================================================================
# Configuration and History
# ============================================================================


@dataclass
class LDMTrainingConfig:
    """
    Configuration for LDM training with EDM (Karras et al. 2022).

    Parameters
    ----------
    epochs : int
        Number of training epochs.
    learning_rate : float
        Adam optimizer learning rate.
    meta_dim : int
        Number of conditioning features.
    sigma_data : float
        Standard deviation of training data (required).
        Compute with `compute_sigma_data(training_latents)`.
    dropout_p : float
        CFG dropout probability.
    p_mean : float
        Mean of EDM log-normal noise distribution.
    p_std : float
        Std of EDM log-normal noise distribution.
    ema_decay : float
        EMA decay rate. Set to 0.0 to disable EMA.
    print_every : int
        Print training progress every N epochs.
    validate_every : int
        Validate every N epochs.
    save_best : bool
        Save best model based on validation loss.
    run_name : str
        Name for saved checkpoint file.
    save_dir : Optional[str]
        Custom save directory. If None, uses ~/.cache/desisky/saved_models/ldm.
    random_seed : int
        Random seed for reproducibility.
    val_expids : list[int] | None
        Validation set exposure IDs. If provided, stored in checkpoint
        metadata so downstream code can identify which samples were
        held out during training.
    conditioning_scaler : dict | None
        Conditioning feature normalization parameters from training.
        Expected keys: ``"mean"`` (list), ``"scale"`` (list), and
        ``"columns"`` (list of feature names). If provided, stored in
        checkpoint metadata for automatic normalization at inference time.

    Examples
    --------
    >>> from desisky.models.ldm import compute_sigma_data
    >>> sigma_data = compute_sigma_data(training_latents)
    >>> config = LDMTrainingConfig(
    ...     epochs=500,
    ...     learning_rate=1e-4,
    ...     meta_dim=8,
    ...     sigma_data=sigma_data,
    ...     run_name="ldm_dark",
    ...     val_expids=[12345, 67890, ...],
    ...     conditioning_scaler={"mean": [...], "scale": [...], "columns": [...]},
    ... )
    """

    epochs: int
    learning_rate: float
    meta_dim: int
    sigma_data: float
    dropout_p: float = 0.1
    p_mean: float = EDM_P_MEAN
    p_std: float = EDM_P_STD
    ema_decay: float = 0.9999
    print_every: int = 50
    validate_every: int = 1
    save_best: bool = True
    run_name: str = "ldm_model"
    save_dir: Optional[str] = None
    random_seed: int = 42
    val_expids: Optional[list] = None
    conditioning_scaler: Optional[dict] = None


@dataclass
class LDMTrainingHistory:
    """
    Training history for LDM.

    Both training and validation use the same weighted EDM loss function.
    The only difference is that CFG dropout is disabled during validation
    (same as turning off regular dropout at evaluation time).

    Attributes
    ----------
    train_losses : list[float]
        Weighted EDM training loss per epoch.
    val_losses : list[float]
        Weighted EDM validation loss per epoch (no CFG dropout).
    best_val_loss : float
        Best validation loss achieved.
    best_epoch : int
        Epoch where best validation loss was achieved.
    """

    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    best_val_loss: float = float("inf")
    best_epoch: int = -1


# ============================================================================
# Trainer
# ============================================================================


class LatentDiffusionTrainer:
    """
    Trainer for latent diffusion models using EDM (Karras et al. 2022).

    Features:
    - EDM preconditioned loss with log-normal noise distribution
    - Exponential Moving Average (EMA) of model weights
    - Classifier-free guidance training via conditioning dropout
    - Automatic best-model checkpointing

    The EMA model is a smoothed copy of the training model used for
    sampling and generation. All loss curves (training and validation)
    are computed on the base training model so they are directly
    comparable for diagnosing overfitting.

    Parameters
    ----------
    model : eqx.Module
        Conditional UNet diffusion model (raw F_theta network).
    config : LDMTrainingConfig
        Training configuration (must include `sigma_data`).
    optimizer : optax.GradientTransformation, optional
        Custom optimizer. If None, uses Adam with config.learning_rate.

    Examples
    --------
    >>> from desisky.models.ldm import make_UNet1D_cond, compute_sigma_data
    >>> import jax.random as jr
    >>>
    >>> model = make_UNet1D_cond(
    ...     in_ch=1, out_ch=1, meta_dim=8, hidden=64, levels=4,
    ...     emb_dim=32, key=jr.PRNGKey(0)
    ... )
    >>> sigma_data = compute_sigma_data(training_latents)
    >>> config = LDMTrainingConfig(
    ...     epochs=500, learning_rate=1e-4, meta_dim=8,
    ...     sigma_data=sigma_data, run_name="ldm_dark"
    ... )
    >>> trainer = LatentDiffusionTrainer(model, config)
    >>> trained_model, ema_model, history = trainer.train(train_loader, val_loader)
    """

    def __init__(
        self,
        model: eqx.Module,
        config: LDMTrainingConfig,
        optimizer: Optional[optax.GradientTransformation] = None,
    ):
        self.model = model
        self.config = config
        self.optimizer = optimizer or optax.adam(config.learning_rate)
        self.history = LDMTrainingHistory()
        self.best_model = None

        # Initialize EMA model as a copy of the base model
        if config.ema_decay > 0.0:
            self.ema_model = model
        else:
            self.ema_model = None

    def train(self, train_loader, val_loader=None):
        """
        Train the LDM model with EDM loss and EMA.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader yielding (latents, conditioning) batches.
        val_loader : DataLoader, optional
            Validation data loader. If None, trains without validation.

        Returns
        -------
        model : eqx.Module
            Trained base model (final state).
        ema_model : eqx.Module or None
            EMA model (None if EMA disabled).
        history : LDMTrainingHistory
            Training history with losses.
        """
        cfg = self.config
        opt_state = self.optimizer.init(eqx.filter(self.model, eqx.is_array))

        # JIT-compile EDM training step
        @eqx.filter_jit
        def make_step(model, opt_state, x, cond, key):
            loss, grads = eqx.filter_value_and_grad(edm_loss)(
                model,
                x,
                cond,
                cfg.sigma_data,
                key,
                cfg.dropout_p,
                cfg.p_mean,
                cfg.p_std,
            )
            updates, opt_state = self.optimizer.update(grads, opt_state, model)
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss

        key = jr.PRNGKey(cfg.random_seed)

        for epoch in range(cfg.epochs):
            # ===== Training =====
            epoch_loss = 0.0
            n_samples = 0

            for x, cond in train_loader:
                key, subkey = jr.split(key)
                self.model, opt_state, loss_value = make_step(
                    self.model, opt_state, x, cond, subkey
                )

                # Update EMA
                if self.ema_model is not None:
                    self.ema_model = ema_update_jit(
                        self.ema_model, self.model, cfg.ema_decay
                    )

                bsz = len(x)
                n_samples += bsz
                epoch_loss += float(loss_value) * bsz

            epoch_loss /= n_samples
            self.history.train_losses.append(epoch_loss)

            # ===== Validation =====
            if val_loader is not None and epoch % cfg.validate_every == 0:
                key, subkey = jr.split(key)
                val_loss = self._evaluate(val_loader, subkey)
                self.history.val_losses.append(float(val_loss))

                # Track and save best model
                if val_loss < self.history.best_val_loss:
                    self.history.best_val_loss = float(val_loss)
                    self.history.best_epoch = epoch
                    self.best_model = self.model

                    if cfg.save_best:
                        self._save_checkpoint(epoch, val_loss)

                if epoch % cfg.print_every == 0:
                    print(
                        f"Epoch {epoch:4d}/{cfg.epochs} | "
                        f"Train: {epoch_loss:.6f} | "
                        f"Val: {val_loss:.6f} | "
                        f"Best: {self.history.best_val_loss:.6f}"
                    )
            elif epoch % cfg.print_every == 0:
                print(
                    f"Epoch {epoch:4d}/{cfg.epochs} | Train: {epoch_loss:.6f}"
                )

        return self.model, self.ema_model, self.history

    def _evaluate(self, val_loader, key: jax.random.PRNGKey) -> float:
        """
        Evaluate on validation set using the same weighted EDM loss.

        Uses the base training model with CFG dropout disabled,
        analogous to turning off regular dropout at evaluation time.
        """
        total_loss = 0.0
        n_samples = 0

        for x, cond in val_loader:
            key, subkey = jr.split(key)
            loss = edm_loss(
                self.model,
                x,
                cond,
                self.config.sigma_data,
                subkey,
                cfg_dropout_p=0.0,
                p_mean=self.config.p_mean,
                p_std=self.config.p_std,
            )
            bsz = len(x)
            n_samples += bsz
            total_loss += float(loss) * bsz

        return total_loss / n_samples

    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        """Save model checkpoint with metadata using desisky.io.save."""
        if self.config.save_dir is not None:
            save_dir = Path(self.config.save_dir)
        else:
            save_dir = Path.home() / ".cache" / "desisky" / "saved_models" / "ldm"

        save_path = save_dir / f"{self.config.run_name}.eqx"

        # Read architecture params from the model to ensure metadata
        # always matches the actual model (fix from old hardcoded values)
        metadata = {
            "schema": 1,
            "arch": {
                "in_ch": self.model.in_ch,
                "out_ch": self.model.out_ch,
                "meta_dim": self.model.meta_dim,
                "hidden": self.model.hidden,
                "levels": self.model.levels,
                "emb_dim": self.model.emb_dim,
            },
            "training": {
                "date": datetime.now().isoformat(),
                "epoch": epoch,
                "val_loss": float(val_loss),
                "train_loss": float(self.history.train_losses[-1]),
                "sigma_data": self.config.sigma_data,
                **({"val_expids": self.config.val_expids}
                   if self.config.val_expids is not None else {}),
                **({"conditioning_scaler": self.config.conditioning_scaler}
                   if self.config.conditioning_scaler is not None else {}),
                "config": {
                    "epochs": self.config.epochs,
                    "learning_rate": self.config.learning_rate,
                    "dropout_p": self.config.dropout_p,
                    "meta_dim": self.config.meta_dim,
                    "p_mean": self.config.p_mean,
                    "p_std": self.config.p_std,
                    "ema_decay": self.config.ema_decay,
                },
            },
        }

        save(save_path, self.model, metadata)
        print(f"  Saved checkpoint: {save_path}")

        # Also save EMA model if enabled
        if self.ema_model is not None:
            ema_path = save_dir / f"{self.config.run_name}_ema.eqx"
            save(ema_path, self.ema_model, metadata)
            print(f"  Saved EMA checkpoint: {ema_path}")