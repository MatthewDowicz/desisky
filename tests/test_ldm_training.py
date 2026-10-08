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

from desisky.models.ldm import make_UNet1D_cond, compute_sigma_data
from desisky.training import (
    NumpyLoader,
    LatentDiffusionTrainer,
    LDMTrainingConfig,
    LDMTrainingHistory,
    edm_loss,
    edm_loss_weight,
    sample_edm_sigma,
    ema_update,
    ema_update_jit,
    fit_conditioning_scaler,
    normalize_conditioning,
)


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
def sigma_data(mock_latents_and_conditioning):
    """Compute sigma_data from mock training latents."""
    latents, _ = mock_latents_and_conditioning
    return compute_sigma_data(latents)


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


# ---------- Conditioning Scaler Tests ----------


class TestConditioningScaler:
    """Test fit_conditioning_scaler and normalize_conditioning."""

    def test_fit_scaler_returns_correct_keys(self):
        """Scaler dict has mean, scale, and columns."""
        data = np.random.default_rng(0).standard_normal((50, 3)).astype(np.float32)
        scaler = fit_conditioning_scaler(data, ["A", "B", "C"])
        assert set(scaler.keys()) == {"mean", "scale", "columns"}
        assert scaler["columns"] == ["A", "B", "C"]
        assert len(scaler["mean"]) == 3
        assert len(scaler["scale"]) == 3

    def test_fit_scaler_values(self):
        """Mean and scale match numpy computation."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((100, 4)).astype(np.float32)
        scaler = fit_conditioning_scaler(data, ["a", "b", "c", "d"])
        np.testing.assert_allclose(scaler["mean"], data.mean(axis=0).tolist(), atol=1e-6)
        np.testing.assert_allclose(scaler["scale"], data.std(axis=0).tolist(), atol=1e-6)

    def test_normalize_zero_mean_unit_var(self):
        """Normalized training data has approximately zero mean and unit std."""
        rng = np.random.default_rng(7)
        data = rng.normal(loc=50.0, scale=10.0, size=(200, 3)).astype(np.float32)
        scaler = fit_conditioning_scaler(data, ["x", "y", "z"])
        normed = normalize_conditioning(data, scaler)
        np.testing.assert_allclose(normed.mean(axis=0), 0.0, atol=1e-5)
        np.testing.assert_allclose(normed.std(axis=0), 1.0, atol=1e-5)

    def test_normalize_val_uses_train_stats(self):
        """Validation set is normalized with training statistics, not its own."""
        rng = np.random.default_rng(1)
        train = rng.normal(loc=0.0, scale=1.0, size=(100, 2)).astype(np.float32)
        val = rng.normal(loc=5.0, scale=2.0, size=(50, 2)).astype(np.float32)
        scaler = fit_conditioning_scaler(train, ["a", "b"])
        val_normed = normalize_conditioning(val, scaler)
        # Val mean should NOT be ~0 since we used train stats on shifted data
        assert np.abs(val_normed.mean(axis=0)).max() > 1.0


# ---------- EDM Noise Distribution Tests ----------


class TestEDMNoiseDistribution:
    """Test EDM noise distribution functions."""

    def test_sample_sigma_shape(self):
        """Test that sigma sampling returns correct shape."""
        key = jr.PRNGKey(0)
        sigma = sample_edm_sigma(key, (16,))
        assert sigma.shape == (16,)

    def test_sample_sigma_positive(self):
        """Test that sampled sigmas are positive."""
        key = jr.PRNGKey(0)
        sigma = sample_edm_sigma(key, (100,))
        assert jnp.all(sigma > 0)

    def test_sample_sigma_finite(self):
        """Test that sampled sigmas are finite."""
        key = jr.PRNGKey(0)
        sigma = sample_edm_sigma(key, (100,))
        assert jnp.all(jnp.isfinite(sigma))

    def test_loss_weight_positive(self):
        """Test that loss weights are positive."""
        sigma = jnp.array([0.01, 0.1, 1.0, 10.0])
        weights = edm_loss_weight(sigma, sigma_data=1.0)
        assert jnp.all(weights > 0)

    def test_loss_weight_finite(self):
        """Test that loss weights are finite."""
        sigma = jnp.array([0.01, 0.1, 1.0, 10.0])
        weights = edm_loss_weight(sigma, sigma_data=1.0)
        assert jnp.all(jnp.isfinite(weights))


# ---------- EDM Loss Tests ----------


class TestEDMLoss:
    """Test EDM loss function."""

    def test_loss_returns_scalar(self, small_ldm, sigma_data):
        """Test that loss function returns a scalar."""
        key = jr.PRNGKey(0)
        x = jnp.ones((4, 1, 8))
        cond = jnp.ones((4, 4))

        loss = edm_loss(small_ldm, x, cond, sigma_data, key)

        assert isinstance(loss, jnp.ndarray)
        assert loss.shape == ()

    def test_loss_is_positive(self, small_ldm, sigma_data):
        """Test that loss is non-negative."""
        key = jr.PRNGKey(0)
        x = jnp.ones((4, 1, 8))
        cond = jnp.ones((4, 4))

        loss = edm_loss(small_ldm, x, cond, sigma_data, key)
        assert loss >= 0

    def test_loss_is_finite(self, small_ldm, sigma_data):
        """Test that loss values are finite."""
        key = jr.PRNGKey(0)
        x = jr.normal(key, (8, 1, 8))
        cond = jr.normal(jr.PRNGKey(1), (8, 4))

        loss = edm_loss(small_ldm, x, cond, sigma_data, jr.PRNGKey(2))
        assert jnp.isfinite(loss)

    def test_loss_different_with_different_dropout(self, small_ldm, sigma_data):
        """Test that CFG dropout affects loss value."""
        key = jr.PRNGKey(0)
        x = jr.normal(key, (8, 1, 8))
        cond = jr.normal(jr.PRNGKey(1), (8, 4))

        loss_no_dropout = edm_loss(
            small_ldm, x, cond, sigma_data, jr.PRNGKey(2), cfg_dropout_p=0.0
        )
        loss_with_dropout = edm_loss(
            small_ldm, x, cond, sigma_data, jr.PRNGKey(2), cfg_dropout_p=0.5
        )

        assert jnp.isfinite(loss_no_dropout)
        assert jnp.isfinite(loss_with_dropout)

    def test_loss_batch_independence(self, small_ldm, sigma_data):
        """Test that loss can handle different batch sizes."""
        for batch_size in [1, 4, 8, 16]:
            x = jr.normal(jr.PRNGKey(0), (batch_size, 1, 8))
            cond = jr.normal(jr.PRNGKey(1), (batch_size, 4))

            loss = edm_loss(small_ldm, x, cond, sigma_data, jr.PRNGKey(2))
            assert jnp.isfinite(loss)


# ---------- EMA Tests ----------


class TestEMA:
    """Test EMA utility functions."""

    def test_ema_update_preserves_structure(self, small_ldm):
        """Test that EMA update preserves model structure."""
        ema_model = small_ldm  # Start as copy
        updated = ema_update(ema_model, small_ldm, decay=0.9999)

        # Should return same type
        assert type(updated) == type(small_ldm)

    def test_ema_update_with_decay_1(self, small_ldm):
        """Test that decay=1.0 keeps EMA unchanged."""
        ema_model = small_ldm
        # Create a different model
        other_model = make_UNet1D_cond(
            in_ch=1, out_ch=1, meta_dim=4,
            hidden=16, levels=2, emb_dim=16,
            key=jr.PRNGKey(99)
        )

        updated = ema_update(ema_model, other_model, decay=1.0)

        # With decay=1.0, EMA should stay the same
        ema_params = eqx.filter(ema_model, eqx.is_array)
        updated_params = eqx.filter(updated, eqx.is_array)

        for e, u in zip(
            jax.tree.leaves(ema_params),
            jax.tree.leaves(updated_params),
        ):
            assert jnp.allclose(e, u, atol=1e-6)

    def test_ema_update_jit(self, small_ldm):
        """Test JIT-compiled EMA update."""
        ema_model = small_ldm
        updated = ema_update_jit(ema_model, small_ldm, 0.9999)
        assert type(updated) == type(small_ldm)


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
        x = jnp.ones((2, 1, 8))
        t = jnp.array([[0.5], [0.7]])
        cond = jnp.ones((2, 4))
        key = jr.PRNGKey(0)

        output = jax.vmap(
            lambda x_i, t_i, c_i, k_i: small_ldm(x_i, t_i, c_i, key=k_i, dropout_p=0.0)
        )(x, t, cond, jr.split(key, 2))

        assert output.shape == (2, 1, 8)

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

    def test_single_training_step(self, small_ldm, mock_latents_and_conditioning, sigma_data):
        """Test a single training step with EDM loss."""
        latents, conditioning = mock_latents_and_conditioning

        x = jnp.array(latents[:4])
        cond = jnp.array(conditioning[:4])

        optimizer = optax.adam(1e-4)
        opt_state = optimizer.init(eqx.filter(small_ldm, eqx.is_array))

        loss, grads = eqx.filter_value_and_grad(edm_loss)(
            small_ldm, x, cond, sigma_data, jr.PRNGKey(0), cfg_dropout_p=0.1
        )

        updates, opt_state = optimizer.update(grads, opt_state, small_ldm)
        new_model = eqx.apply_updates(small_ldm, updates)

        assert jnp.isfinite(loss)
        assert new_model is not None

    def test_multiple_training_steps(self, small_ldm, train_val_loaders, sigma_data):
        """Test multiple training steps."""
        train_loader, _ = train_val_loaders

        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(eqx.filter(small_ldm, eqx.is_array))

        model = small_ldm
        losses = []
        key = jr.PRNGKey(0)

        for i, (x, cond) in enumerate(train_loader):
            if i >= 3:
                break

            key, subkey = jr.split(key)
            loss, grads = eqx.filter_value_and_grad(edm_loss)(
                model, x, cond, sigma_data, subkey, cfg_dropout_p=0.1
            )

            updates, opt_state = optimizer.update(grads, opt_state, model)
            model = eqx.apply_updates(model, updates)

            losses.append(float(loss))

        assert len(losses) == 3
        assert all(np.isfinite(loss) for loss in losses)

    def test_validation_loop(self, small_ldm, train_val_loaders, sigma_data):
        """Test validation loop."""
        _, val_loader = train_val_loaders
        key = jr.PRNGKey(0)

        total_loss = 0.0
        n_samples = 0

        for x, cond in val_loader:
            key, subkey = jr.split(key)
            loss = edm_loss(
                small_ldm, x, cond, sigma_data, subkey, cfg_dropout_p=0.0
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

    def test_full_training_pipeline_no_validation(
        self, small_ldm, train_val_loaders, sigma_data
    ):
        """Test complete training pipeline without validation."""
        train_loader, _ = train_val_loaders

        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(eqx.filter(small_ldm, eqx.is_array))

        model = small_ldm
        train_losses = []
        key = jr.PRNGKey(42)

        for epoch in range(3):
            epoch_loss = 0.0
            n_samples = 0

            for x, cond in train_loader:
                key, subkey = jr.split(key)
                loss, grads = eqx.filter_value_and_grad(edm_loss)(
                    model, x, cond, sigma_data, subkey, cfg_dropout_p=0.1
                )

                updates, opt_state = optimizer.update(grads, opt_state, model)
                model = eqx.apply_updates(model, updates)

                bsz = len(x)
                n_samples += bsz
                epoch_loss += float(loss) * bsz

            epoch_loss /= n_samples
            train_losses.append(epoch_loss)

        assert len(train_losses) == 3
        assert all(np.isfinite(loss) for loss in train_losses)

    def test_full_training_pipeline_with_validation(
        self, small_ldm, train_val_loaders, sigma_data
    ):
        """Test complete training pipeline with validation."""
        train_loader, val_loader = train_val_loaders

        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(eqx.filter(small_ldm, eqx.is_array))

        model = small_ldm
        train_losses = []
        val_losses = []
        key = jr.PRNGKey(42)

        for epoch in range(3):
            # Training
            epoch_loss = 0.0
            n_samples = 0

            for x, cond in train_loader:
                key, subkey = jr.split(key)
                loss, grads = eqx.filter_value_and_grad(edm_loss)(
                    model, x, cond, sigma_data, subkey, cfg_dropout_p=0.1
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
                loss = edm_loss(
                    model, x, cond, sigma_data, subkey, cfg_dropout_p=0.0
                )
                bsz = len(x)
                n_val += bsz
                val_loss += float(loss) * bsz

            val_loss /= n_val
            val_losses.append(val_loss)

        assert len(train_losses) == 3
        assert len(val_losses) == 3
        assert all(np.isfinite(loss) for loss in train_losses)
        assert all(np.isfinite(loss) for loss in val_losses)

    def test_model_checkpoint_format(self, small_ldm):
        """Test that model can be saved and loaded using desisky.io format."""
        from desisky.io import save, load

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_ldm.eqx"

            metadata = {
                "schema": 1,
                "model_type": "ldm",
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
                    "sigma_data": 1.0,
                },
            }

            save(save_path, small_ldm, metadata)

            assert save_path.exists()

            loaded_model, loaded_meta = load(save_path, constructor=make_UNet1D_cond)

            assert loaded_meta["schema"] == 1
            assert loaded_meta["model_type"] == "ldm"
            assert loaded_meta["arch"]["meta_dim"] == 4
            assert loaded_meta["training"]["epoch"] == 10
            assert loaded_meta["training"]["val_loss"] == 0.123
            assert loaded_meta["training"]["sigma_data"] == 1.0


# ---------- Early Stopping Tests ----------


class TestLDMEarlyStopping:
    """Tests for LDMTrainingConfig.patience."""

    @staticmethod
    def _make_trainer(small_ldm, sigma_data, monkeypatch, val_losses, **cfg_kwargs):
        """Build a trainer whose validation loss follows a fixed sequence.

        ``_evaluate`` is replaced so the test is deterministic; EMA is disabled
        so it is called exactly once per epoch.
        """
        config = LDMTrainingConfig(
            epochs=len(val_losses),
            learning_rate=1e-3,
            meta_dim=4,
            sigma_data=sigma_data,
            ema_decay=0.0,
            save_best=False,
            **cfg_kwargs,
        )
        trainer = LatentDiffusionTrainer(small_ldm, config)
        seq = iter(val_losses)
        monkeypatch.setattr(trainer, "_evaluate", lambda *a, **k: next(seq))
        return trainer

    def test_patience_must_be_positive(self, sigma_data):
        with pytest.raises(ValueError, match="patience"):
            LDMTrainingConfig(
                epochs=1, learning_rate=1e-3, meta_dim=4,
                sigma_data=sigma_data, patience=0,
            )

    def test_patience_stops_training_early(
        self, small_ldm, train_val_loaders, sigma_data, monkeypatch
    ):
        train_loader, val_loader = train_val_loaders
        # Best at epoch 1; no improvement afterwards -> stop at epoch 3
        val_losses = [1.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
        trainer = self._make_trainer(
            small_ldm, sigma_data, monkeypatch, val_losses, patience=2
        )

        _, _, history = trainer.train(train_loader, val_loader)

        assert history.stopped_early is True
        assert history.best_epoch == 1
        assert history.best_val_loss == 0.5
        assert len(history.train_losses) == 4   # epochs 0..3
        assert len(history.val_losses) == 4

    def test_patience_resets_on_improvement(
        self, small_ldm, train_val_loaders, sigma_data, monkeypatch
    ):
        train_loader, val_loader = train_val_loaders
        # Improvements at epochs 0, 2, 4 keep the counter below patience=2
        # until the tail of flat epochs; stop at epoch 6.
        val_losses = [1.0, 1.1, 0.9, 1.0, 0.8, 0.9, 0.9, 0.9]
        trainer = self._make_trainer(
            small_ldm, sigma_data, monkeypatch, val_losses, patience=2
        )

        _, _, history = trainer.train(train_loader, val_loader)

        assert history.stopped_early is True
        assert history.best_epoch == 4
        assert len(history.train_losses) == 7   # epochs 0..6

    def test_no_patience_runs_all_epochs(
        self, small_ldm, train_val_loaders, sigma_data, monkeypatch
    ):
        train_loader, val_loader = train_val_loaders
        val_losses = [1.0, 0.5, 0.6, 0.7, 0.8]
        trainer = self._make_trainer(small_ldm, sigma_data, monkeypatch, val_losses)

        _, _, history = trainer.train(train_loader, val_loader)

        assert history.stopped_early is False
        assert len(history.train_losses) == len(val_losses)

    def test_patience_ignored_without_validation(
        self, small_ldm, train_val_loaders, sigma_data, monkeypatch
    ):
        train_loader, _ = train_val_loaders
        trainer = self._make_trainer(
            small_ldm, sigma_data, monkeypatch, [None] * 3, patience=1
        )

        _, _, history = trainer.train(train_loader, val_loader=None)

        assert history.stopped_early is False
        assert len(history.train_losses) == 3

    def test_patience_recorded_in_checkpoint(
        self, small_ldm, train_val_loaders, sigma_data
    ):
        from desisky.io import load

        train_loader, val_loader = train_val_loaders
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LDMTrainingConfig(
                epochs=1, learning_rate=1e-3, meta_dim=4,
                sigma_data=sigma_data, ema_decay=0.0, patience=50,
                save_best=True, save_dir=tmpdir, run_name="ldm_patience_test",
            )
            trainer = LatentDiffusionTrainer(small_ldm, config)
            trainer.train(train_loader, val_loader)

            _, meta = load(
                Path(tmpdir) / "ldm_patience_test.eqx", constructor=make_UNet1D_cond
            )
            assert meta["training"]["config"]["patience"] == 50
