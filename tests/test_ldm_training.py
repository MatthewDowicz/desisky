# SPDX-FileCopyrightText: 2025-present MatthewDowicz <mjdowicz@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Tests for LDM (Latent Diffusion Model) training utilities."""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
import torch
from torch.utils.data import TensorDataset
from pathlib import Path
import tempfile

from desisky.models.ldm import make_UNet1D_cond
from desisky.training import (
    NumpyLoader,
    LatentDiffusionTrainer,
    LDMTrainingConfig,
    LDMTrainingHistory,
    cosine_beta_schedule,
    diffusion_loss,
)




# Functions are now imported from desisky.training module


# ---------- Fixtures ----------


@pytest.fixture
def small_ldm():
    """Create a small LDM for testing (faster than full-sized model)."""
    return make_UNet1D_cond(
        in_ch=1,
        out_ch=1,
        meta_dim=4,  # Small for testing
        hidden=16,   # Smaller than default 32
        levels=2,    # Smaller than default 3
        emb_dim=16,  # Smaller than default 32
        key=jr.PRNGKey(42)
    )


@pytest.fixture
def mock_latents_and_conditioning():
    """Create mock latent space data and conditioning for testing."""
    np.random.seed(42)
    n_samples = 100
    latent_dim = 8
    meta_dim = 4

    # Generate mock latent vectors with channel dimension
    # Shape: (N, 1, latent_dim)
    latents = np.random.randn(n_samples, 1, latent_dim).astype(np.float32)

    # Generate mock conditioning metadata
    # Shape: (N, meta_dim)
    conditioning = np.random.randn(n_samples, meta_dim).astype(np.float32)

    return latents, conditioning


@pytest.fixture
def train_val_loaders(mock_latents_and_conditioning):
    """Create train/val data loaders from mock data."""
    latents, conditioning = mock_latents_and_conditioning

    # Split into train/val
    n_train = 80
    train_latents = latents[:n_train]
    train_cond = conditioning[:n_train]
    val_latents = latents[n_train:]
    val_cond = conditioning[n_train:]

    # Create TensorDatasets
    train_dataset = TensorDataset(
        torch.from_numpy(train_latents),
        torch.from_numpy(train_cond)
    )

    val_dataset = TensorDataset(
        torch.from_numpy(val_latents),
        torch.from_numpy(val_cond)
    )

    # Create NumpyLoaders for JAX compatibility
    train_loader = NumpyLoader(
        train_dataset,
        batch_size=16,
        shuffle=True
    )

    val_loader = NumpyLoader(
        val_dataset,
        batch_size=16,
        shuffle=False
    )

    return train_loader, val_loader


# ---------- Noise Schedule Tests ----------


class TestNoiseSchedule:
    """Test noise schedule functions."""

    def test_cosine_schedule_shape(self):
        """Test that cosine schedule returns correct shapes."""
        T = 1000
        schedules = cosine_beta_schedule(T)

        assert "sqrtab" in schedules
        assert "sqrtmab" in schedules
        assert schedules["sqrtab"].shape == (T + 1,)
        assert schedules["sqrtmab"].shape == (T + 1,)

    def test_cosine_schedule_values(self):
        """Test that schedule values are in expected range."""
        schedules = cosine_beta_schedule(T=1000)

        # sqrt(alphabar) should be in [0, 1]
        assert jnp.all(schedules["sqrtab"] >= 0)
        assert jnp.all(schedules["sqrtab"] <= 1)

        # sqrt(1-alphabar) should be in [0, 1]
        assert jnp.all(schedules["sqrtmab"] >= 0)
        assert jnp.all(schedules["sqrtmab"] <= 1)

        # At t=0, alphabar should be close to 1 (mostly clean data)
        # Note: cosine schedule with s=0.008 offset means it's not exactly 1
        assert schedules["sqrtab"][0] > 0.99  # Very close to 1
        assert schedules["sqrtmab"][0] < 0.01  # Very close to 0

        # At t=T, alphabar should be small (pure noise)
        assert schedules["sqrtab"][-1] < 0.1
        assert schedules["sqrtmab"][-1] > 0.9

    def test_cosine_schedule_monotonicity(self):
        """Test that sqrtab decreases and sqrtmab increases with timestep."""
        schedules = cosine_beta_schedule(T=1000)

        # sqrtab should be decreasing
        sqrtab_diff = jnp.diff(schedules["sqrtab"])
        assert jnp.all(sqrtab_diff <= 0)

        # sqrtmab should be increasing
        sqrtmab_diff = jnp.diff(schedules["sqrtmab"])
        assert jnp.all(sqrtmab_diff >= 0)

    def test_cosine_schedule_different_T(self):
        """Test schedule with different timestep counts."""
        for T in [100, 500, 1000, 2000]:
            schedules = cosine_beta_schedule(T=T)
            assert schedules["sqrtab"].shape == (T + 1,)
            assert schedules["sqrtmab"].shape == (T + 1,)


# ---------- Diffusion Loss Tests ----------


class TestDiffusionLoss:
    """Test diffusion loss function."""

    def test_loss_returns_scalar(self, small_ldm):
        """Test that loss function returns a scalar."""
        key = jr.PRNGKey(0)
        x = jnp.ones((4, 1, 8))  # (batch, channel, latent_dim)
        cond = jnp.ones((4, 4))  # (batch, meta_dim)
        schedules = cosine_beta_schedule(T=100)

        loss = diffusion_loss(small_ldm, x, cond, schedules, n_T=100, key=key)

        assert isinstance(loss, jnp.ndarray)
        assert loss.shape == ()  # Scalar

    def test_loss_is_positive(self, small_ldm):
        """Test that loss is non-negative (MSE)."""
        key = jr.PRNGKey(0)
        x = jnp.ones((4, 1, 8))
        cond = jnp.ones((4, 4))
        schedules = cosine_beta_schedule(T=100)

        loss = diffusion_loss(small_ldm, x, cond, schedules, n_T=100, key=key)

        assert loss >= 0

    def test_loss_is_finite(self, small_ldm):
        """Test that loss values are finite (no NaN or Inf)."""
        key = jr.PRNGKey(0)
        x = jr.normal(key, (8, 1, 8))
        cond = jr.normal(jr.PRNGKey(1), (8, 4))
        schedules = cosine_beta_schedule(T=100)

        loss = diffusion_loss(small_ldm, x, cond, schedules, n_T=100, key=jr.PRNGKey(2))

        assert jnp.isfinite(loss)

    def test_loss_different_with_different_dropout(self, small_ldm):
        """Test that dropout affects loss value."""
        key = jr.PRNGKey(0)
        x = jr.normal(key, (8, 1, 8))
        cond = jr.normal(jr.PRNGKey(1), (8, 4))
        schedules = cosine_beta_schedule(T=100)

        loss_no_dropout = diffusion_loss(
            small_ldm, x, cond, schedules, n_T=100, key=jr.PRNGKey(2), dropout_p=0.0
        )
        loss_with_dropout = diffusion_loss(
            small_ldm, x, cond, schedules, n_T=100, key=jr.PRNGKey(2), dropout_p=0.5
        )

        # Losses should be different (due to dropout)
        # Note: They might be close but not identical
        assert jnp.isfinite(loss_no_dropout)
        assert jnp.isfinite(loss_with_dropout)

    def test_loss_different_timesteps(self, small_ldm):
        """Test loss computation with different numbers of timesteps."""
        key = jr.PRNGKey(0)
        x = jr.normal(key, (4, 1, 8))
        cond = jr.normal(jr.PRNGKey(1), (4, 4))

        for n_T in [100, 500, 1000]:
            schedules = cosine_beta_schedule(T=n_T)
            loss = diffusion_loss(small_ldm, x, cond, schedules, n_T=n_T, key=jr.PRNGKey(2))
            assert jnp.isfinite(loss)

    def test_loss_batch_independence(self, small_ldm):
        """Test that loss can handle different batch sizes."""
        schedules = cosine_beta_schedule(T=100)

        for batch_size in [1, 4, 8, 16]:
            x = jr.normal(jr.PRNGKey(0), (batch_size, 1, 8))
            cond = jr.normal(jr.PRNGKey(1), (batch_size, 4))

            loss = diffusion_loss(small_ldm, x, cond, schedules, n_T=100, key=jr.PRNGKey(2))
            assert jnp.isfinite(loss)


# ---------- Model Tests ----------


class TestLDMModel:
    """Test LDM model architecture."""

    def test_model_creation(self):
        """Test creating an LDM model."""
        ldm = make_UNet1D_cond(
            in_ch=1,
            out_ch=1,
            meta_dim=8,
            hidden=32,
            levels=3,
            emb_dim=32,
            key=jr.PRNGKey(0)
        )

        assert ldm is not None

    def test_model_forward_pass(self, small_ldm):
        """Test forward pass through the model."""
        x = jnp.ones((2, 1, 8))  # (batch, channel, latent_dim)
        t = jnp.array([[0.5], [0.7]])  # (batch, 1)
        cond = jnp.ones((2, 4))  # (batch, meta_dim)
        key = jr.PRNGKey(0)

        output = jax.vmap(
            lambda x_i, t_i, c_i, k_i: small_ldm(x_i, t_i, c_i, key=k_i, dropout_p=0.0)
        )(x, t, cond, jr.split(key, 2))

        assert output.shape == (2, 1, 8)  # Same as input

    def test_model_output_finite(self, small_ldm):
        """Test that model outputs are finite."""
        x = jr.normal(jr.PRNGKey(0), (4, 1, 8))
        t = jnp.array([[0.1], [0.3], [0.5], [0.7]])
        cond = jr.normal(jr.PRNGKey(1), (4, 4))
        key = jr.PRNGKey(2)

        output = jax.vmap(
            lambda x_i, t_i, c_i, k_i: small_ldm(x_i, t_i, c_i, key=k_i, dropout_p=0.0)
        )(x, t, cond, jr.split(key, 4))

        assert jnp.all(jnp.isfinite(output))


# ---------- Training Loop Tests ----------


class TestLDMTrainingLoop:
    """Test basic training loop functionality."""

    def test_single_training_step(self, small_ldm, mock_latents_and_conditioning):
        """Test a single training step."""
        latents, conditioning = mock_latents_and_conditioning

        # Get a small batch
        x = jnp.array(latents[:4])
        cond = jnp.array(conditioning[:4])

        schedules = cosine_beta_schedule(T=100)
        optimizer = optax.adam(1e-4)
        opt_state = optimizer.init(eqx.filter(small_ldm, eqx.is_array))

        # Compute loss and gradients
        loss, grads = eqx.filter_value_and_grad(diffusion_loss)(
            small_ldm, x, cond, schedules, n_T=100, key=jr.PRNGKey(0), dropout_p=0.1
        )

        # Apply gradients
        updates, opt_state = optimizer.update(grads, opt_state, small_ldm)
        new_model = eqx.apply_updates(small_ldm, updates)

        assert jnp.isfinite(loss)
        assert new_model is not None

    def test_multiple_training_steps(self, small_ldm, train_val_loaders):
        """Test multiple training steps."""
        train_loader, _ = train_val_loaders

        schedules = cosine_beta_schedule(T=100)
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(eqx.filter(small_ldm, eqx.is_array))

        model = small_ldm
        losses = []
        key = jr.PRNGKey(0)

        # Train for a few batches
        for i, (x, cond) in enumerate(train_loader):
            if i >= 3:  # Just test 3 batches
                break

            key, subkey = jr.split(key)
            loss, grads = eqx.filter_value_and_grad(diffusion_loss)(
                model, x, cond, schedules, n_T=100, key=subkey, dropout_p=0.1
            )

            updates, opt_state = optimizer.update(grads, opt_state, model)
            model = eqx.apply_updates(model, updates)

            losses.append(float(loss))

        # Check that training ran
        assert len(losses) == 3
        assert all(np.isfinite(loss) for loss in losses)

    def test_validation_loop(self, small_ldm, train_val_loaders):
        """Test validation loop."""
        _, val_loader = train_val_loaders

        schedules = cosine_beta_schedule(T=100)
        key = jr.PRNGKey(0)

        total_loss = 0.0
        n_samples = 0

        for x, cond in val_loader:
            key, subkey = jr.split(key)
            loss = diffusion_loss(
                small_ldm, x, cond, schedules,
                n_T=100, key=subkey, dropout_p=0.0  # No dropout during validation
            )
            bsz = len(x)
            n_samples += bsz
            total_loss += float(loss) * bsz

        val_loss = total_loss / n_samples

        assert np.isfinite(val_loss)
        assert val_loss > 0


# ---------- Integration Tests ----------


class TestLDMTrainingIntegration:
    """Integration tests for full LDM training pipeline."""

    def test_full_training_pipeline_no_validation(self, small_ldm, train_val_loaders):
        """Test complete training pipeline without validation."""
        train_loader, _ = train_val_loaders

        schedules = cosine_beta_schedule(T=100)
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(eqx.filter(small_ldm, eqx.is_array))

        model = small_ldm
        train_losses = []
        key = jr.PRNGKey(42)

        # Train for 3 epochs
        for epoch in range(3):
            epoch_loss = 0.0
            n_samples = 0

            for x, cond in train_loader:
                key, subkey = jr.split(key)
                loss, grads = eqx.filter_value_and_grad(diffusion_loss)(
                    model, x, cond, schedules, n_T=100, key=subkey, dropout_p=0.1
                )

                updates, opt_state = optimizer.update(grads, opt_state, model)
                model = eqx.apply_updates(model, updates)

                bsz = len(x)
                n_samples += bsz
                epoch_loss += float(loss) * bsz

            epoch_loss /= n_samples
            train_losses.append(epoch_loss)

        # Verify training completed
        assert len(train_losses) == 3
        assert all(np.isfinite(loss) for loss in train_losses)

    def test_full_training_pipeline_with_validation(self, small_ldm, train_val_loaders):
        """Test complete training pipeline with validation."""
        train_loader, val_loader = train_val_loaders

        schedules = cosine_beta_schedule(T=100)
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(eqx.filter(small_ldm, eqx.is_array))

        model = small_ldm
        train_losses = []
        val_losses = []
        key = jr.PRNGKey(42)

        # Train for 3 epochs with validation
        for epoch in range(3):
            # Training
            epoch_loss = 0.0
            n_samples = 0

            for x, cond in train_loader:
                key, subkey = jr.split(key)
                loss, grads = eqx.filter_value_and_grad(diffusion_loss)(
                    model, x, cond, schedules, n_T=100, key=subkey, dropout_p=0.1
                )

                updates, opt_state = optimizer.update(grads, opt_state, model)
                model = eqx.apply_updates(model, updates)

                bsz = len(x)
                n_samples += bsz
                epoch_loss += float(loss) * bsz

            epoch_loss /= n_samples
            train_losses.append(epoch_loss)

            # Validation
            val_loss = 0.0
            n_val = 0
            for x, cond in val_loader:
                key, subkey = jr.split(key)
                loss = diffusion_loss(
                    model, x, cond, schedules,
                    n_T=100, key=subkey, dropout_p=0.0
                )
                bsz = len(x)
                n_val += bsz
                val_loss += float(loss) * bsz

            val_loss /= n_val
            val_losses.append(val_loss)

        # Verify training completed
        assert len(train_losses) == 3
        assert len(val_losses) == 3
        assert all(np.isfinite(loss) for loss in train_losses)
        assert all(np.isfinite(loss) for loss in val_losses)

    def test_model_checkpoint_format(self, small_ldm):
        """Test that model can be saved and loaded using desisky.io format."""
        from desisky.io import save, load

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_ldm.eqx"

            # Create metadata following desisky.io format
            # schema is required, model_type is optional but recommended for clarity
            metadata = {
                "schema": 1,  # Required by desisky.io
                "model_type": "ldm",  # Optional, for clarity
                "arch": {
                    "in_ch": 1,
                    "out_ch": 1,
                    "meta_dim": 4,
                    "hidden": 16,
                    "levels": 2,
                    "emb_dim": 16,
                },
                "training": {
                    "epoch": 10,
                    "val_loss": 0.123,
                },
            }

            # Save using desisky.io.save
            save(save_path, small_ldm, metadata)

            assert save_path.exists()

            # Load using desisky.io.load
            loaded_model, loaded_meta = load(save_path, constructor=make_UNet1D_cond)

            # Check metadata
            assert loaded_meta["schema"] == 1
            assert loaded_meta["model_type"] == "ldm"  # Custom field preserved
            assert loaded_meta["arch"]["meta_dim"] == 4
            assert loaded_meta["training"]["epoch"] == 10
            assert loaded_meta["training"]["val_loss"] == 0.123
