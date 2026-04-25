# SPDX-FileCopyrightText: 2025-present MatthewDowicz <mjdowicz@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Trainer for broadband sky brightness models."""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional
from datetime import datetime
import warnings

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
from torch.utils.data import DataLoader

from .losses import loss_func
from .dataset import gather_full_data
from desisky.io import save as save_model
from .wandb_utils import WandbConfig


@dataclass
class TrainingConfig:
    """
    Configuration for broadband model training.

    Parameters
    ----------
    epochs : int
        Number of training epochs.
    learning_rate : float
        Learning rate for the optimizer.
    loss : {"l2", "huber"}, default "l2"
        Loss function to use.
    huber_delta : float, default 1.0
        Delta parameter for Huber loss (ignored if loss="l2").
    save_best : bool, default True
        If True, save the model checkpoint with the best test loss.
    save_dir : str | Path | None, default None
        Directory to save model checkpoints. If None and save_best=True,
        uses ~/.cache/desisky/saved_models/broadband.
    run_name : str, default "broadband_training"
        Name for this training run (used in checkpoint filename).
    print_every : int, default 50
        Print training progress every N epochs.
    validate_every : int, default 1
        Compute validation metrics every N epochs.

    Examples
    --------
    >>> config = TrainingConfig(
    ...     epochs=500,
    ...     learning_rate=1e-4,
    ...     loss="huber",
    ...     huber_delta=0.25,
    ...     run_name="moon_model_v1"
    ... )
    """

    epochs: int
    learning_rate: float
    loss: Literal["l2", "huber"] = "l2"
    huber_delta: float = 1.0
    save_best: bool = True
    save_dir: Optional[str | Path] = None
    run_name: str = "broadband_training"
    print_every: int = 50
    validate_every: int = 1


@dataclass
class TrainingHistory:
    """
    Training history tracking loss curves and metadata.

    Attributes
    ----------
    train_losses : list[float]
        Training loss at each epoch.
    test_losses : list[float]
        Test/validation loss at each epoch.
    best_test_loss : float
        Best test loss achieved during training.
    best_epoch : int
        Epoch at which best test loss was achieved.
    config : TrainingConfig
        Configuration used for this training run.
    """

    train_losses: list[float] = field(default_factory=list)
    test_losses: list[float] = field(default_factory=list)
    best_test_loss: float = float("inf")
    best_epoch: int = -1
    config: Optional[TrainingConfig] = None


class BroadbandTrainer:
    """
    Trainer for broadband sky brightness models.

    This class handles the training loop, checkpointing, and metric tracking
    for Equinox-based models predicting multi-band sky magnitudes.

    Parameters
    ----------
    model : eqx.Module
        Equinox model to train (e.g., eqx.nn.MLP).
    config : TrainingConfig
        Training configuration.
    optimizer : optax.GradientTransformation | None, default None
        Optax optimizer. If None, uses Adam with config.learning_rate.

    Attributes
    ----------
    model : eqx.Module
        The model being trained.
    config : TrainingConfig
        Training configuration.
    optimizer : optax.GradientTransformation
        Optimizer for training.
    history : TrainingHistory
        Training history and metrics.

    Examples
    --------
    >>> import equinox as eqx
    >>> import jax
    >>> from desisky.training import BroadbandTrainer, TrainingConfig
    >>>
    >>> # Create model
    >>> model = eqx.nn.MLP(in_size=6, out_size=4, width_size=128, depth=5,
    ...                     key=jax.random.PRNGKey(42))
    >>>
    >>> # Configure training
    >>> config = TrainingConfig(epochs=500, learning_rate=1e-4, loss="huber",
    ...                          huber_delta=0.25)
    >>>
    >>> # Train
    >>> trainer = BroadbandTrainer(model, config)
    >>> model, history = trainer.train(train_loader, test_loader)
    """

    def __init__(
        self,
        model: eqx.Module,
        config: TrainingConfig,
        optimizer: Optional[optax.GradientTransformation] = None,
        wandb_config: Optional[WandbConfig] = None,
        on_epoch_end: Optional[callable] = None,
    ):
        self.model = model
        self.config = config
        self.optimizer = optimizer or optax.adam(config.learning_rate)
        self.wandb_config = wandb_config
        self.on_epoch_end = on_epoch_end
        self.history = TrainingHistory(config=config)

    def _resolve_run_name(self) -> str:
        """Determine checkpoint filename (mirrors VAETrainer pattern).

        If the user explicitly set run_name (differs from default), use it.
        Otherwise fall back to the wandb-generated run name when available.
        """
        user_changed_default = self.config.run_name != "broadband_training"
        if user_changed_default:
            return self.config.run_name
        if self.wandb_config is not None:
            try:
                import wandb
                if wandb.run is not None:
                    return wandb.run.name
            except ImportError:
                pass
        return self.config.run_name

    def train(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
    ) -> tuple[eqx.Module, TrainingHistory]:
        """
        Train the model on the provided data loaders.

        Initializes wandb (if configured), runs the training loop,
        and ensures wandb is finalized on exit.

        Parameters
        ----------
        train_loader : DataLoader
            DataLoader for the training set.
        test_loader : DataLoader
            DataLoader for the test/validation set.

        Returns
        -------
        model : eqx.Module
            Trained model (best checkpoint if config.save_best=True).
        history : TrainingHistory
            Training history with loss curves and metadata.

        Examples
        --------
        >>> model, history = trainer.train(train_loader, test_loader)
        >>> print(f"Best test loss: {history.best_test_loss:.4f} at epoch {history.best_epoch}")
        """
        # Initialize wandb run if configured
        if self.wandb_config is not None:
            from .wandb_utils import init_wandb_run
            model_config = self._extract_architecture()
            run = init_wandb_run(self.wandb_config, self.config, model_config)
            print(f"  wandb run: {run.url}")

        try:
            return self._train_loop(train_loader, test_loader)
        finally:
            # Ensure wandb is properly finalized
            if self.wandb_config is not None:
                from .wandb_utils import finish_wandb_run
                try:
                    import wandb
                    if wandb.run is not None:
                        print(f"  wandb run: {wandb.run.url}")
                except ImportError:
                    pass
                finish_wandb_run()

    def _train_loop(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
    ) -> tuple[eqx.Module, TrainingHistory]:
        """Core training loop with wandb logging and callback support."""
        opt_state = self.optimizer.init(eqx.filter(self.model, eqx.is_array))

        @eqx.filter_jit
        def make_step(model, opt_state, inputs, targets):
            lval, grads = eqx.filter_value_and_grad(loss_func)(
                model,
                jnp.asarray(inputs),
                jnp.asarray(targets),
                loss_name=self.config.loss,
                huber_delta=self.config.huber_delta,
            )
            updates, opt_state = self.optimizer.update(grads, opt_state, model)
            model = eqx.apply_updates(model, updates)
            return model, opt_state, lval

        # Main training loop
        for epoch in range(self.config.epochs):
            epoch_train_loss = 0.0
            batch_count = 0

            # Training step
            for inputs, targets, _ in train_loader:
                self.model, opt_state, train_loss = make_step(
                    self.model, opt_state, inputs, targets
                )
                epoch_train_loss += float(train_loss)
                batch_count += 1

            epoch_train_loss /= batch_count
            self.history.train_losses.append(epoch_train_loss)

            # Log training metrics to wandb
            if self.wandb_config is not None:
                from .wandb_utils import log_epoch_metrics
                if epoch % self.wandb_config.log_every == 0:
                    log_epoch_metrics({"loss": epoch_train_loss}, epoch, prefix="train/")

            # Validation step
            if epoch % self.config.validate_every == 0:
                test_loss = self._evaluate(test_loader)
                self.history.test_losses.append(float(test_loss))

                # Log validation metrics to wandb
                if self.wandb_config is not None:
                    from .wandb_utils import log_epoch_metrics
                    if epoch % self.wandb_config.log_every == 0:
                        log_epoch_metrics({"loss": float(test_loss)}, epoch, prefix="val/")

                # Check if this is the best model
                if test_loss < self.history.best_test_loss:
                    self.history.best_test_loss = float(test_loss)
                    self.history.best_epoch = epoch
                    if self.config.save_best:
                        self._save_checkpoint(epoch, test_loss, test_loader)

                # Print progress
                if epoch % self.config.print_every == 0:
                    print(
                        f"Epoch {epoch:4d} | Train: {epoch_train_loss:.6f} | "
                        f"Test: {test_loss:.6f} | Best: {self.history.best_test_loss:.6f}"
                    )

            # Epoch callback (wandb-agnostic — callers handle logging)
            if self.on_epoch_end is not None:
                self.on_epoch_end(self.model, self.history, epoch)

        return self.model, self.history

    def _evaluate(self, test_loader: DataLoader) -> jnp.ndarray:
        """
        Evaluate the model on a test/validation set.

        Parameters
        ----------
        test_loader : DataLoader
            DataLoader for the test set.

        Returns
        -------
        avg_loss : jnp.ndarray
            Average loss over the test set.
        """
        avg_loss = 0.0
        n_batches = 0

        for inputs, targets, _ in test_loader:
            batch_loss = loss_func(
                self.model,
                jnp.asarray(inputs),
                jnp.asarray(targets),
                loss_name=self.config.loss,
                huber_delta=self.config.huber_delta,
            )
            avg_loss += batch_loss
            n_batches += 1

        return avg_loss / max(n_batches, 1)

    def _save_checkpoint(
        self, epoch: int, test_loss: float, test_loader: DataLoader
    ) -> None:
        """
        Save model checkpoint with metadata.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        test_loss : float
            Test loss at this epoch.
        test_loader : DataLoader
            Test loader for computing per-band RMSE.
        """
        # Determine save path
        if self.config.save_dir is None:
            from desisky.io import get_user_model_dir

            save_dir = get_user_model_dir("broadband")
        else:
            save_dir = Path(self.config.save_dir)

        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{self._resolve_run_name()}.eqx"

        # Compute per-band RMSE on test set
        try:
            all_inputs, all_targets, _, _ = gather_full_data(test_loader)
        except AttributeError:
            # User data uses TensorDataset (no .metadata attribute).
            # Gather inputs/targets directly from batches.
            inputs_list, targets_list = [], []
            for batch in test_loader:
                inputs_list.append(np.asarray(batch[0]))
                targets_list.append(np.asarray(batch[1]))
            all_inputs = np.concatenate(inputs_list)
            all_targets = np.concatenate(targets_list)
        per_band_rmse = self._compute_per_band_rmse(all_inputs, all_targets)

        # Extract model architecture from the model itself
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
                "per_band_rmse": {
                    "V": per_band_rmse[0],
                    "g": per_band_rmse[1],
                    "r": per_band_rmse[2],
                    "z": per_band_rmse[3],
                },
                "config": {
                    "epochs": self.config.epochs,
                    "learning_rate": self.config.learning_rate,
                    "loss": self.config.loss,
                    "huber_delta": self.config.huber_delta,
                },
            },
        }

        # Save checkpoint
        save_model(save_path, self.model, meta)

    def _compute_per_band_rmse(
        self, inputs: np.ndarray, targets: np.ndarray
    ) -> list[float]:
        """
        Compute RMSE for each output band separately.

        Parameters
        ----------
        inputs : np.ndarray
            Input features. Shape: (N, n_features).
        targets : np.ndarray
            Target magnitudes. Shape: (N, 4).

        Returns
        -------
        rmse_per_band : list[float]
            RMSE for each of the 4 bands [V, g, r, z].
        """
        pred = jax.vmap(self.model)(jnp.asarray(inputs))
        rmse = jnp.sqrt(jnp.mean((pred - jnp.asarray(targets)) ** 2, axis=0))
        return [float(x) for x in rmse]

    def _extract_architecture(self) -> dict:
        """
        Extract architecture parameters from the model.

        Returns
        -------
        arch : dict
            Dictionary of architecture parameters (in_size, out_size, etc.).
        """
        # For eqx.nn.MLP, extract the architecture parameters
        if isinstance(self.model, eqx.nn.MLP):
            return {
                "in_size": self.model.in_size,
                "out_size": self.model.out_size,
                "width_size": self.model.width_size,
                "depth": self.model.depth,
            }
        else:
            warnings.warn(
                f"Unknown model type {type(self.model).__name__}. "
                "Architecture extraction may be incomplete.",
                UserWarning,
            )
            return {}
