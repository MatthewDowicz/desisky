# SPDX-FileCopyrightText: 2025-present MatthewDowicz <mjdowicz@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Trainer for Variational Autoencoder (VAE) sky spectral models."""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from datetime import datetime
import warnings

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
import numpy as np
from torch.utils.data import DataLoader

from .vae_losses import vae_loss_infovae, default_kernel_sigma
from desisky.io import save as save_model


@dataclass
class VAETrainingConfig:
    """
    Configuration for VAE model training with InfoVAE-MMD objective.

    The InfoVAE framework provides better control over the balance between
    reconstruction quality and latent space regularization compared to
    standard β-VAE.

    Parameters
    ----------
    epochs : int
        Number of training epochs.
    learning_rate : float
        Learning rate for the optimizer.
    beta : float, default 1e-3
        Weight for KL divergence term (corresponds to 1-α in InfoVAE).
        Lower values (e.g., 1e-3 to 1e-2) allow better reconstruction
        while maintaining a structured latent space.
    lam : float, default 4.0
        Total weight for latent regularization (λ in InfoVAE).
        The MMD term receives weight (λ-β). Typical range: 1.0 to 10.0.
    kernel_sigma : float | str, default "auto"
        Bandwidth for RBF kernel in MMD computation.
        If "auto", uses heuristic σ = √(2/latent_dim).
        Can also specify explicit float value.
    clip_gradients : bool, default False
        If True, apply gradient clipping by global norm (max norm = 1.0).
        Can improve training stability but may slow convergence.
    save_best : bool, default True
        If True, save the model checkpoint with the best test loss.
    save_dir : str | Path | None, default None
        Directory to save model checkpoints. If None and save_best=True,
        uses ~/.cache/desisky/saved_models/vae.
    run_name : str, default "vae_training"
        Name for this training run (used in checkpoint filename).
    print_every : int, default 10
        Print training progress every N epochs.
    validate_every : int, default 1
        Compute validation metrics every N epochs.
    random_seed : int, default 42
        Random seed for reproducible training.

    Examples
    --------
    >>> # Standard InfoVAE configuration
    >>> config = VAETrainingConfig(
    ...     epochs=100,
    ...     learning_rate=1e-4,
    ...     beta=1e-3,
    ...     lam=4.0,
    ...     run_name="sky_vae_v1"
    ... )
    >>>
    >>> # Higher reconstruction quality (lower beta)
    >>> config = VAETrainingConfig(
    ...     epochs=200,
    ...     learning_rate=1e-4,
    ...     beta=1e-4,
    ...     lam=2.0,
    ...     run_name="sky_vae_high_quality"
    ... )

    Notes
    -----
    Hyperparameter selection guidelines:
    - β controls KL weight: lower → better reconstruction, less regularization
    - λ controls total regularization: higher → more structured latent space
    - Typical combinations: (β=1e-3, λ=4), (β=1e-2, λ=10), (β=1e-4, λ=2)
    - kernel_sigma="auto" works well in most cases
    """

    epochs: int
    learning_rate: float
    beta: float = 1e-3
    lam: float = 4.0
    kernel_sigma: float | str = "auto"
    clip_gradients: bool = False
    save_best: bool = True
    save_dir: Optional[str | Path] = None
    run_name: str = "vae_training"
    print_every: int = 10
    validate_every: int = 1
    random_seed: int = 42


@dataclass
class VAETrainingHistory:
    """
    Training history for VAE, tracking all loss components.

    Attributes
    ----------
    train_losses : list[float]
        Total training loss at each epoch.
    test_losses : list[float]
        Total test/validation loss at each epoch.
    train_recon : list[float]
        Training reconstruction loss (MSE) at each epoch.
    test_recon : list[float]
        Test reconstruction loss at each epoch.
    train_kl : list[float]
        Training KL divergence (weighted by beta) at each epoch.
    test_kl : list[float]
        Test KL divergence at each epoch.
    train_mmd : list[float]
        Training MMD (weighted by lambda-beta) at each epoch.
    test_mmd : list[float]
        Test MMD at each epoch.
    best_test_loss : float
        Best test loss achieved during training.
    best_epoch : int
        Epoch at which best test loss was achieved.
    config : VAETrainingConfig
        Configuration used for this training run.
    """

    train_losses: list[float] = field(default_factory=list)
    test_losses: list[float] = field(default_factory=list)
    train_recon: list[float] = field(default_factory=list)
    test_recon: list[float] = field(default_factory=list)
    train_kl: list[float] = field(default_factory=list)
    test_kl: list[float] = field(default_factory=list)
    train_mmd: list[float] = field(default_factory=list)
    test_mmd: list[float] = field(default_factory=list)
    best_test_loss: float = float("inf")
    best_epoch: int = -1
    config: Optional[VAETrainingConfig] = None


class VAETrainer:
    """
    Trainer for Variational Autoencoder models with InfoVAE-MMD objective.

    This class handles the training loop, checkpointing, and metric tracking
    for VAE models that compress DESI sky spectra into a low-dimensional
    latent representation.

    Parameters
    ----------
    model : eqx.Module
        VAE model to train (e.g., SkyVAE from desisky.models.vae).
    config : VAETrainingConfig
        Training configuration.
    optimizer : optax.GradientTransformation | None, default None
        Optax optimizer. If None, uses Adam with config.learning_rate.
        If config.clip_gradients=True, gradient clipping is added automatically.

    Attributes
    ----------
    model : eqx.Module
        The VAE model being trained.
    config : VAETrainingConfig
        Training configuration.
    optimizer : optax.GradientTransformation
        Optimizer for training.
    history : VAETrainingHistory
        Training history with all loss components.

    Examples
    --------
    >>> import jax.random as jr
    >>> from desisky.models.vae import make_SkyVAE
    >>> from desisky.training import VAETrainer, VAETrainingConfig
    >>>
    >>> # Create VAE model
    >>> model = make_SkyVAE(in_channels=7781, latent_dim=8, key=jr.PRNGKey(0))
    >>>
    >>> # Configure training with InfoVAE
    >>> config = VAETrainingConfig(
    ...     epochs=100,
    ...     learning_rate=1e-4,
    ...     beta=1e-3,
    ...     lam=4.0,
    ...     run_name="sky_vae_v1"
    ... )
    >>>
    >>> # Train
    >>> trainer = VAETrainer(model, config)
    >>> model, history = trainer.train(train_loader, test_loader)
    >>>
    >>> # Inspect results
    >>> print(f"Best test loss: {history.best_test_loss:.4f}")
    >>> print(f"Final reconstruction: {history.test_recon[-1]:.4f}")
    """

    def __init__(
        self,
        model: eqx.Module,
        config: VAETrainingConfig,
        optimizer: Optional[optax.GradientTransformation] = None,
    ):
        self.model = model
        self.config = config

        # Build optimizer with optional gradient clipping
        if optimizer is None:
            if config.clip_gradients:
                optimizer = optax.chain(
                    optax.clip_by_global_norm(1.0),
                    optax.adam(config.learning_rate)
                )
            else:
                optimizer = optax.adam(config.learning_rate)

        self.optimizer = optimizer
        self.history = VAETrainingHistory(config=config)

    def train(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
    ) -> tuple[eqx.Module, VAETrainingHistory]:
        """
        Train the VAE model on the provided data loaders.

        Parameters
        ----------
        train_loader : DataLoader
            DataLoader for the training set. Should yield batches of spectra.
            Can use NumpyLoader from desisky.training for JAX compatibility.
        test_loader : DataLoader
            DataLoader for the test/validation set.

        Returns
        -------
        model : eqx.Module
            Trained VAE model (best checkpoint if config.save_best=True).
        history : VAETrainingHistory
            Training history with all loss components and metadata.

        Examples
        --------
        >>> model, history = trainer.train(train_loader, test_loader)
        >>> print(f"Training complete!")
        >>> print(f"Best test loss: {history.best_test_loss:.4f} at epoch {history.best_epoch}")
        >>>
        >>> # Plot training curves
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(history.train_losses, label='Train')
        >>> plt.plot(history.test_losses, label='Test')
        >>> plt.xlabel('Epoch')
        >>> plt.ylabel('Loss')
        >>> plt.legend()
        >>> plt.show()
        """
        # Initialize optimizer state
        opt_state = self.optimizer.init(eqx.filter(self.model, eqx.is_array))

        # Compute kernel bandwidth if auto
        kernel_sigma = self.config.kernel_sigma
        if kernel_sigma == "auto":
            kernel_sigma = default_kernel_sigma(self.model.latent_dim)

        # JIT-compile training step for performance
        @eqx.filter_jit
        def make_step(model, opt_state, x, key):
            """Single training step: compute loss, gradients, and update model."""
            (loss, aux), grads = eqx.filter_value_and_grad(vae_loss_infovae, has_aux=True)(
                model,
                x=x,
                key=key,
                beta=self.config.beta,
                lam=self.config.lam,
                kernel_sigma=kernel_sigma
            )
            updates, opt_state = self.optimizer.update(grads, opt_state, model)
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss, aux

        # Random key for sampling
        key = jr.PRNGKey(self.config.random_seed)

        # Main training loop
        for epoch in range(self.config.epochs):
            # Accumulators for epoch statistics
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_kl = 0.0
            epoch_mmd = 0.0
            n_samples = 0

            # Training step over all batches
            for x in train_loader:
                # Split key for this batch
                key, subkey = jr.split(key)

                # Ensure x is JAX array
                if not isinstance(x, jnp.ndarray):
                    x = jnp.asarray(x)

                # Perform training step
                self.model, opt_state, loss_value, aux = make_step(
                    self.model, opt_state, x, subkey
                )

                # Accumulate statistics
                bsz = len(x)
                n_samples += bsz
                epoch_loss += float(loss_value) * bsz
                epoch_recon += float(aux["recon"]) * bsz
                epoch_kl += float(aux["kl_weighted"]) * bsz
                epoch_mmd += float(aux["mmd_weighted"]) * bsz

            # Compute epoch-averaged losses
            epoch_loss /= n_samples
            epoch_recon /= n_samples
            epoch_kl /= n_samples
            epoch_mmd /= n_samples

            # Store training metrics
            self.history.train_losses.append(epoch_loss)
            self.history.train_recon.append(epoch_recon)
            self.history.train_kl.append(epoch_kl)
            self.history.train_mmd.append(epoch_mmd)

            # Validation step
            if epoch % self.config.validate_every == 0:
                key, subkey = jr.split(key)
                test_loss, test_aux = self._evaluate(test_loader, subkey, kernel_sigma)

                # Store test metrics
                self.history.test_losses.append(float(test_loss))
                self.history.test_recon.append(float(test_aux["recon"]))
                self.history.test_kl.append(float(test_aux["kl_weighted"]))
                self.history.test_mmd.append(float(test_aux["mmd_weighted"]))

                # Check if this is the best model
                if test_loss < self.history.best_test_loss:
                    self.history.best_test_loss = float(test_loss)
                    self.history.best_epoch = epoch

                    if self.config.save_best:
                        self._save_checkpoint(epoch, test_loss, test_aux)

                # Print progress
                if epoch % self.config.print_every == 0:
                    print(
                        f"Epoch {epoch:4d}/{self.config.epochs} | "
                        f"Train: {epoch_loss:.6f} (R:{epoch_recon:.4f} KL:{epoch_kl:.4f} MMD:{epoch_mmd:.4f}) | "
                        f"Test: {test_loss:.6f} (R:{test_aux['recon']:.4f}) | "
                        f"Best: {self.history.best_test_loss:.6f}"
                    )

        return self.model, self.history

    def _evaluate(
        self,
        test_loader: DataLoader,
        key: jax.random.PRNGKey,
        kernel_sigma: float
    ) -> tuple[jnp.ndarray, dict]:
        """
        Evaluate the VAE model on a test/validation set.

        Parameters
        ----------
        test_loader : DataLoader
            DataLoader for the test set.
        key : jax.random.PRNGKey
            Random key for latent sampling.
        kernel_sigma : float
            Kernel bandwidth for MMD computation.

        Returns
        -------
        avg_loss : jnp.ndarray
            Average total loss over the test set.
        avg_aux : dict
            Dictionary of average auxiliary metrics (recon, kl_weighted, etc.).
        """
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        total_mmd = 0.0
        n_samples = 0

        for x in test_loader:
            # Split key for this batch
            key, subkey = jr.split(key)

            # Ensure x is JAX array
            if not isinstance(x, jnp.ndarray):
                x = jnp.asarray(x)

            # Compute loss
            loss, aux = vae_loss_infovae(
                self.model,
                x,
                subkey,
                beta=self.config.beta,
                lam=self.config.lam,
                kernel_sigma=kernel_sigma
            )

            # Accumulate
            bsz = len(x)
            n_samples += bsz
            total_loss += float(loss) * bsz
            total_recon += float(aux["recon"]) * bsz
            total_kl += float(aux["kl_weighted"]) * bsz
            total_mmd += float(aux["mmd_weighted"]) * bsz

        # Compute averages
        avg_aux = {
            "recon": total_recon / n_samples,
            "kl_weighted": total_kl / n_samples,
            "mmd_weighted": total_mmd / n_samples,
            "loss_z": (total_kl + total_mmd) / n_samples
        }

        return total_loss / n_samples, avg_aux

    def _save_checkpoint(
        self,
        epoch: int,
        test_loss: float,
        test_aux: dict
    ) -> None:
        """
        Save VAE model checkpoint with metadata.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        test_loss : float
            Test loss at this epoch.
        test_aux : dict
            Dictionary of test metrics (recon, kl, mmd).
        """
        # Determine save path
        if self.config.save_dir is None:
            from desisky.io import get_user_model_dir
            save_dir = get_user_model_dir("vae")
        else:
            save_dir = Path(self.config.save_dir)

        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{self.config.run_name}.eqx"

        # Extract model architecture
        arch = self._extract_architecture()

        # Create metadata
        meta = {
            "schema": 1,
            "arch": arch,
            "training": {
                "date": datetime.now().isoformat(),
                "epoch": epoch,
                "test_loss": float(test_loss),
                "train_loss": float(self.history.train_losses[-1])
                if self.history.train_losses
                else None,
                "test_metrics": {
                    "recon": float(test_aux["recon"]),
                    "kl_weighted": float(test_aux["kl_weighted"]),
                    "mmd_weighted": float(test_aux["mmd_weighted"]),
                },
                "config": {
                    "epochs": self.config.epochs,
                    "learning_rate": self.config.learning_rate,
                    "beta": self.config.beta,
                    "lam": self.config.lam,
                    "kernel_sigma": self.config.kernel_sigma,
                },
            },
        }

        # Save checkpoint
        save_model(save_path, self.model, meta)

    def _extract_architecture(self) -> dict:
        """
        Extract architecture parameters from the VAE model.

        Returns
        -------
        arch : dict
            Dictionary of architecture parameters (in_channels, latent_dim).
        """
        # Check if model has the expected attributes
        if hasattr(self.model, 'in_channels') and hasattr(self.model, 'latent_dim'):
            return {
                "in_channels": self.model.in_channels,
                "latent_dim": self.model.latent_dim,
            }
        else:
            warnings.warn(
                f"Model {type(self.model).__name__} does not have expected "
                "attributes (in_channels, latent_dim). Architecture extraction "
                "may be incomplete.",
                UserWarning,
            )
            return {}
