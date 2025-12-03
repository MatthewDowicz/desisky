# SPDX-FileCopyrightText: 2025-present MatthewDowicz <mjdowicz@gmail.com>
#
# SPDX-License-Identifier: MIT

"""
Latent Diffusion Model (LDM) training utilities.

This module provides a flexible trainer for conditional latent diffusion models
that can be used with different conditioning variables (dark-time, twilight, moon).
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from desisky.io import save


# ============================================================================
# Noise Schedule
# ============================================================================


def cosine_beta_schedule(T: int, s: float = 0.008) -> dict[str, jnp.ndarray]:
    """
    Cosine noise schedule from Nichol & Dhariwal (2021).

    This schedule provides a smoother noise curve compared to linear schedules,
    which helps with training stability and sample quality.

    Parameters
    ----------
    T : int
        Number of diffusion timesteps.
    s : float, optional
        Small offset to prevent singularities, by default 0.008.

    Returns
    -------
    dict[str, jnp.ndarray]
        Dictionary containing:
        - 'sqrtab': sqrt(alpha_bar_t), used for forward diffusion
        - 'sqrtmab': sqrt(1 - alpha_bar_t), noise coefficient
        Both arrays have shape (T+1,) indexed from 0 to T.

    Notes
    -----
    The forward diffusion process adds noise according to:
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon

    References
    ----------
    Nichol, A. Q., & Dhariwal, P. (2021). Improved denoising diffusion
    probabilistic models. ICML 2021.

    Examples
    --------
    >>> schedules = cosine_beta_schedule(T=1000)
    >>> schedules['sqrtab'].shape
    (1001,)
    >>> schedules['sqrtab'][0]  # At t=0, should be close to 1 (clean data)
    Array(1., dtype=float32)
    """
    t = jnp.linspace(0, T, T + 1, dtype=jnp.float32)
    f = lambda t_: jnp.cos((t_ / T + s) / (1 + s) * jnp.pi / 2) ** 2
    alphabar_t = f(t) / f(0)

    beta_t = 1 - alphabar_t[1:] / alphabar_t[:-1]
    beta_t = jnp.clip(beta_t, a_min=1e-5, a_max=0.999)
    beta_t = jnp.concatenate([jnp.array([1e-5]), beta_t])

    alpha_t = 1 - beta_t
    log_alpha_t = jnp.log(alpha_t)
    alphabar_t = jnp.exp(jnp.cumsum(log_alpha_t))

    sqrtab = jnp.sqrt(alphabar_t)
    sqrtmab = jnp.sqrt(1 - alphabar_t)

    return {
        "sqrtab": sqrtab,
        "sqrtmab": sqrtmab,
    }


# ============================================================================
# Loss Function
# ============================================================================


def diffusion_loss(
    model: eqx.Module,
    x: jnp.ndarray,
    cond: jnp.ndarray,
    schedules: dict[str, jnp.ndarray],
    n_T: int,
    key: jax.random.PRNGKey,
    dropout_p: float = 0.1,
) -> jnp.ndarray:
    """
    Diffusion model training loss (noise prediction MSE).

    This loss function trains the model to predict the noise added during the
    forward diffusion process. It implements classifier-free guidance training
    by randomly dropping conditioning with probability `dropout_p`.

    Parameters
    ----------
    model : eqx.Module
        UNet diffusion model that takes (x, t, conditioning, key, dropout_p)
        and outputs predicted noise.
    x : jnp.ndarray
        Clean latent samples, shape (batch, channels, latent_dim).
    cond : jnp.ndarray
        Conditioning metadata, shape (batch, meta_dim).
    schedules : dict[str, jnp.ndarray]
        Noise schedule with 'sqrtab' and 'sqrtmab' from cosine_beta_schedule.
    n_T : int
        Number of diffusion timesteps (e.g., 1000).
    key : jax.random.PRNGKey
        Random key for sampling timesteps and noise.
    dropout_p : float, optional
        Probability of dropping conditioning for classifier-free guidance,
        by default 0.1 (10% dropout).

    Returns
    -------
    jnp.ndarray
        Scalar MSE loss between predicted and true noise.

    Notes
    -----
    The forward diffusion process is:
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon

    The model learns to predict epsilon given (x_t, t, conditioning).

    Classifier-free guidance training randomly drops conditioning to enable
    guidance at inference time via:
        pred = (1 + w) * pred_cond - w * pred_uncond

    Examples
    --------
    >>> import jax.random as jr
    >>> model = make_UNet1D_cond(in_ch=1, out_ch=1, meta_dim=8, key=jr.PRNGKey(0))
    >>> x = jr.normal(jr.PRNGKey(1), (16, 1, 8))  # 16 samples
    >>> cond = jr.normal(jr.PRNGKey(2), (16, 8))  # 8 conditioning features
    >>> schedules = cosine_beta_schedule(T=1000)
    >>> loss = diffusion_loss(model, x, cond, schedules, 1000, jr.PRNGKey(3))
    >>> loss.shape
    ()
    """
    key, k_t, k_eps, k_drop = jr.split(key, 4)
    B = x.shape[0]

    # Sample random timesteps for each sample in batch
    t_int = jr.randint(k_t, (B,), 1, n_T + 1)  # [1, n_T]
    t_norm = t_int.astype(jnp.float32) / n_T   # Normalize to [0, 1]
    t_norm = t_norm[:, None]                   # (B, 1)

    # Sample noise from standard normal
    eps = jr.normal(k_eps, shape=x.shape)

    # Gather schedule values for each timestep
    sqrtab = schedules["sqrtab"][t_int][:, None, None]   # (B, 1, 1)
    sqrtmab = schedules["sqrtmab"][t_int][:, None, None]  # (B, 1, 1)

    # Forward diffusion: create noisy latents
    x_t = sqrtab * x + sqrtmab * eps

    # Generate dropout keys for classifier-free guidance
    drop_keys = jr.split(k_drop, B)

    # Predict noise (model handles CFG dropout internally)
    predicted_eps = jax.vmap(
        lambda xt, t, c, k: model(xt, t, c, key=k, dropout_p=dropout_p)
    )(x_t, t_norm, cond, drop_keys)

    # MSE loss between predicted and true noise
    loss = jnp.mean((predicted_eps - eps) ** 2)
    return loss


# ============================================================================
# Configuration and History
# ============================================================================


@dataclass
class LDMTrainingConfig:
    """
    Configuration for LDM training.

    This configuration is flexible and works with different conditioning
    variables (dark-time, twilight, moon) by accepting meta_dim dynamically.

    Attributes
    ----------
    epochs : int
        Number of training epochs.
    learning_rate : float
        Adam optimizer learning rate.
    meta_dim : int
        Number of conditioning features (automatically set from model).
    n_T : int, optional
        Number of diffusion timesteps, by default 1000.
    dropout_p : float, optional
        Classifier-free guidance dropout probability, by default 0.1.
    print_every : int, optional
        Print training progress every N epochs, by default 50.
    validate_every : int, optional
        Validate every N epochs, by default 1.
    save_best : bool, optional
        Save best model based on validation loss, by default True.
    run_name : str, optional
        Name for saved checkpoint file, by default "ldm_model".
    save_dir : Optional[str], optional
        Custom save directory. If None, uses ~/.cache/desisky/saved_models/ldm.
    random_seed : int, optional
        Random seed for reproducibility, by default 42.

    Examples
    --------
    >>> config = LDMTrainingConfig(
    ...     epochs=500,
    ...     learning_rate=1e-4,
    ...     meta_dim=8,  # 8 conditioning features for dark-time
    ...     run_name="ldm_dark"
    ... )
    """

    epochs: int
    learning_rate: float
    meta_dim: int  # Dynamically set based on conditioning columns
    n_T: int = 1000
    dropout_p: float = 0.1
    print_every: int = 50
    validate_every: int = 1
    save_best: bool = True
    run_name: str = "ldm_model"
    save_dir: Optional[str] = None
    random_seed: int = 42


@dataclass
class LDMTrainingHistory:
    """
    Training history for LDM.

    Tracks training and validation losses throughout training.

    Attributes
    ----------
    train_losses : list[float]
        Training loss per epoch.
    val_losses : list[float]
        Validation loss per epoch (if validation used).
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
    Trainer for latent diffusion models.

    This trainer is flexible and can be used for different conditioning
    scenarios (dark-time, twilight, moon) by accepting models with different
    meta_dim values.

    The trainer:
    - Handles training loop with optional validation
    - Saves best model checkpoints using desisky.io format
    - Tracks training history
    - Supports classifier-free guidance training

    Parameters
    ----------
    model : eqx.Module
        Conditional UNet diffusion model (e.g., from make_UNet1D_cond).
    config : LDMTrainingConfig
        Training configuration.
    optimizer : optax.GradientTransformation, optional
        Custom optimizer. If None, uses Adam with config.learning_rate.

    Attributes
    ----------
    model : eqx.Module
        The diffusion model being trained.
    config : LDMTrainingConfig
        Training configuration.
    optimizer : optax.GradientTransformation
        Optimizer for training.
    history : LDMTrainingHistory
        Training history tracker.
    schedules : dict[str, jnp.ndarray]
        Noise schedule (sqrtab, sqrtmab).
    best_model : eqx.Module or None
        Best model (lowest validation loss).

    Examples
    --------
    >>> from desisky.models.ldm import make_UNet1D_cond
    >>> import jax.random as jr
    >>>
    >>> # Create model for dark-time (8 conditioning features)
    >>> model = make_UNet1D_cond(
    ...     in_ch=1, out_ch=1, meta_dim=8, hidden=32, levels=3, key=jr.PRNGKey(0)
    ... )
    >>>
    >>> # Configure training
    >>> config = LDMTrainingConfig(
    ...     epochs=500, learning_rate=1e-4, meta_dim=8, run_name="ldm_dark"
    ... )
    >>>
    >>> # Train
    >>> trainer = LatentDiffusionTrainer(model, config)
    >>> trained_model, history = trainer.train(train_loader, val_loader)
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
        self.schedules = cosine_beta_schedule(T=config.n_T)
        self.best_model = None

    def train(self, train_loader, val_loader=None):
        """
        Train the LDM model.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader yielding (latents, conditioning) batches.
        val_loader : DataLoader, optional
            Validation data loader. If None, trains without validation.

        Returns
        -------
        model : eqx.Module
            Trained model (final state).
        history : LDMTrainingHistory
            Training history with losses.

        Notes
        -----
        For diffusion models, validation loss measures denoising performance
        on held-out data. This is useful for monitoring training stability
        but does NOT directly measure generation quality. Evaluate models
        via visual inspection of samples, FID scores, or other metrics.

        For production training, consider using 100% of data (val_loader=None)
        after hyperparameter tuning.
        """
        # Initialize optimizer state
        opt_state = self.optimizer.init(eqx.filter(self.model, eqx.is_array))

        # JIT-compile training step for performance
        @eqx.filter_jit
        def make_step(model, opt_state, x, cond, key):
            loss, grads = eqx.filter_value_and_grad(diffusion_loss)(
                model,
                x,
                cond,
                self.schedules,
                self.config.n_T,
                key,
                self.config.dropout_p,
            )
            updates, opt_state = self.optimizer.update(grads, opt_state, model)
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss

        # Training loop
        key = jr.PRNGKey(self.config.random_seed)

        for epoch in range(self.config.epochs):
            # ===== Training =====
            epoch_loss = 0.0
            n_samples = 0

            for x, cond in train_loader:
                key, subkey = jr.split(key)
                self.model, opt_state, loss_value = make_step(
                    self.model, opt_state, x, cond, subkey
                )

                bsz = len(x)
                n_samples += bsz
                epoch_loss += float(loss_value) * bsz

            epoch_loss /= n_samples
            self.history.train_losses.append(epoch_loss)

            # ===== Validation =====
            if val_loader is not None and epoch % self.config.validate_every == 0:
                key, subkey = jr.split(key)
                val_loss = self._evaluate(val_loader, subkey)
                self.history.val_losses.append(float(val_loss))

                # Track and save best model
                if val_loss < self.history.best_val_loss:
                    self.history.best_val_loss = float(val_loss)
                    self.history.best_epoch = epoch
                    self.best_model = self.model

                    # Save checkpoint if requested
                    if self.config.save_best:
                        self._save_checkpoint(epoch, val_loss)

                # Print progress
                if epoch % self.config.print_every == 0:
                    print(
                        f"Epoch {epoch:4d}/{self.config.epochs} | "
                        f"Train: {epoch_loss:.6f} | "
                        f"Val: {val_loss:.6f} | "
                        f"Best: {self.history.best_val_loss:.6f}"
                    )
            elif epoch % self.config.print_every == 0:
                # Print progress without validation
                print(
                    f"Epoch {epoch:4d}/{self.config.epochs} | " f"Train: {epoch_loss:.6f}"
                )

        return self.model, self.history

    def _evaluate(self, val_loader, key: jax.random.PRNGKey) -> float:
        """
        Evaluate model on validation set.

        Parameters
        ----------
        val_loader : DataLoader
            Validation data loader.
        key : jax.random.PRNGKey
            Random key for evaluation.

        Returns
        -------
        float
            Average validation loss.
        """
        total_loss = 0.0
        n_samples = 0

        for x, cond in val_loader:
            key, subkey = jr.split(key)
            # No dropout during evaluation
            loss = diffusion_loss(
                self.model, x, cond, self.schedules, self.config.n_T, subkey, dropout_p=0.0
            )
            bsz = len(x)
            n_samples += bsz
            total_loss += float(loss) * bsz

        return total_loss / n_samples

    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        """
        Save model checkpoint with metadata using desisky.io.save.

        The checkpoint includes full architecture information and training
        metadata, enabling easy loading and reproducibility.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        val_loss : float
            Current validation loss.
        """
        # Determine save directory (respects user's custom path)
        if self.config.save_dir is not None:
            save_dir = Path(self.config.save_dir)
        else:
            save_dir = Path.home() / ".cache" / "desisky" / "saved_models" / "ldm"

        # Full save path (desisky.io.save will create parent directories)
        save_path = save_dir / f"{self.config.run_name}.eqx"

        # Create metadata following desisky.io format
        # NOTE: Architecture parameters should match your model creation
        metadata = {
            "schema": 1,
            "arch": {
                "in_ch": 1,
                "out_ch": 1,
                "meta_dim": self.config.meta_dim,  # Dynamic based on conditioning
                "hidden": 32,  # Default from make_UNet1D_cond
                "levels": 3,   # Default from make_UNet1D_cond
                "emb_dim": 32,  # Default from make_UNet1D_cond
            },
            "training": {
                "date": datetime.now().isoformat(),
                "epoch": epoch,
                "val_loss": float(val_loss),
                "train_loss": float(self.history.train_losses[-1]),
                "config": {
                    "epochs": self.config.epochs,
                    "learning_rate": self.config.learning_rate,
                    "n_T": self.config.n_T,
                    "dropout_p": self.config.dropout_p,
                    "meta_dim": self.config.meta_dim,
                },
            },
        }

        # Save using desisky.io.save (handles directory creation)
        save(save_path, self.model, metadata)
        print(f"  💾 Saved checkpoint: {save_path}")
