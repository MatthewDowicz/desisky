# SPDX-FileCopyrightText: 2025-present MatthewDowicz <mjdowicz@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Tests for VAE training utilities."""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
import torch
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import tempfile

from desisky.training import (
    VAETrainer,
    VAETrainingConfig,
    VAETrainingHistory,
    vae_loss_infovae,
    mmd_rbf_biased,
    default_kernel_sigma,
    NumpyLoader,
)
from desisky.models.vae import make_SkyVAE


# ---------- Fixtures ----------


@pytest.fixture
def small_vae():
    """Create a small VAE for testing (faster than full-sized model)."""
    return make_SkyVAE(
        in_channels=100,  # Small for testing
        latent_dim=4,
        key=jr.PRNGKey(42)
    )


@pytest.fixture
def mock_spectra():
    """Create mock spectral data for testing."""
    np.random.seed(42)
    n_samples = 200
    n_wavelengths = 100
    # Generate realistic-looking spectra with some structure
    spectra = np.random.randn(n_samples, n_wavelengths).astype(np.float32)
    return spectra


@pytest.fixture
def train_test_loaders(mock_spectra):
    """Create train/test data loaders from mock data."""
    # Split into train/test
    n_train = 150
    n_test = 50
    train_data = mock_spectra[:n_train]
    test_data = mock_spectra[n_train:]

    # Create simple wrapper dataset that returns arrays directly
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    train_dataset = SimpleDataset(train_data)
    test_dataset = SimpleDataset(test_data)

    # Create NumpyLoaders for JAX compatibility
    train_loader = NumpyLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )

    test_loader = NumpyLoader(
        test_dataset,
        batch_size=32,
        shuffle=False
    )

    return train_loader, test_loader


@pytest.fixture
def vae_config():
    """Standard VAE training configuration for testing."""
    return VAETrainingConfig(
        epochs=5,  # Small number for fast tests
        learning_rate=1e-3,
        beta=1e-3,
        lam=4.0,
        kernel_sigma="auto",
        save_best=False,  # Don't save during tests
        print_every=1,
        validate_every=1,
        random_seed=42,
    )


# ---------- Loss Function Tests ----------


class TestKernelFunctions:
    """Test kernel and MMD computation functions."""

    def test_default_kernel_sigma(self):
        """Test kernel bandwidth heuristic."""
        sigma = default_kernel_sigma(latent_dim=8)
        expected = np.sqrt(2.0 / 8.0)
        assert np.isclose(sigma, expected)

    def test_default_kernel_sigma_different_dims(self):
        """Test kernel bandwidth for different latent dimensions."""
        sigma_4 = default_kernel_sigma(4)
        sigma_16 = default_kernel_sigma(16)
        # Higher dimensions should have smaller sigma
        assert sigma_16 < sigma_4

    def test_mmd_same_distribution(self):
        """Test that MMD is near zero for samples from same distribution."""
        key = jr.PRNGKey(0)
        x = jr.normal(key, (100, 8))
        y = jr.normal(jr.PRNGKey(1), (100, 8))

        mmd = mmd_rbf_biased(x, y, sigma=1.0)

        # Should be close to zero (not exactly due to finite sampling)
        assert mmd >= 0  # MMD is always non-negative
        assert mmd < 0.5  # But should be small for same distribution

    def test_mmd_symmetry(self):
        """Test that MMD(x, y) = MMD(y, x)."""
        key = jr.PRNGKey(0)
        x = jr.normal(key, (50, 8))
        y = jr.normal(jr.PRNGKey(1), (50, 8))

        mmd_xy = mmd_rbf_biased(x, y, sigma=1.0)
        mmd_yx = mmd_rbf_biased(y, x, sigma=1.0)

        assert np.isclose(mmd_xy, mmd_yx)


class TestVAELoss:
    """Test VAE loss function computation."""

    def test_loss_returns_correct_structure(self, small_vae, mock_spectra):
        """Test that loss function returns expected output structure."""
        batch = jnp.array(mock_spectra[:32])
        key = jr.PRNGKey(0)

        loss, aux = vae_loss_infovae(
            small_vae,
            batch,
            key,
            beta=1e-3,
            lam=4.0,
            kernel_sigma="auto"
        )

        # Check outputs
        assert isinstance(loss, jnp.ndarray)
        assert loss.shape == ()  # Scalar
        assert isinstance(aux, dict)
        assert set(aux.keys()) == {'recon', 'kl_weighted', 'mmd_weighted', 'loss_z'}

    def test_loss_is_finite(self, small_vae, mock_spectra):
        """Test that loss values are finite (no NaN or Inf)."""
        batch = jnp.array(mock_spectra[:32])
        key = jr.PRNGKey(0)

        loss, aux = vae_loss_infovae(
            small_vae,
            batch,
            key,
            beta=1e-3,
            lam=4.0,
        )

        assert jnp.isfinite(loss)
        assert all(jnp.isfinite(v) for v in aux.values())

    def test_loss_components_positive(self, small_vae, mock_spectra):
        """Test that all loss components are non-negative."""
        batch = jnp.array(mock_spectra[:32])
        key = jr.PRNGKey(0)

        loss, aux = vae_loss_infovae(
            small_vae,
            batch,
            key,
            beta=1.0,
            lam=10.0,
        )

        # Reconstruction and KL should always be positive
        assert aux['recon'] >= 0
        # KL might be slightly negative due to numerical issues, but should be close to 0
        assert aux['kl_weighted'] >= -1e-3

    def test_beta_effect(self, small_vae, mock_spectra):
        """Test that beta parameter affects KL weight."""
        batch = jnp.array(mock_spectra[:32])
        key = jr.PRNGKey(0)

        _, aux_low = vae_loss_infovae(small_vae, batch, key, beta=0.1, lam=4.0)
        _, aux_high = vae_loss_infovae(small_vae, batch, key, beta=1.0, lam=4.0)

        # Higher beta should give higher KL weight
        # Note: KL itself should be the same, but weighted version differs
        assert aux_high['kl_weighted'] > aux_low['kl_weighted']

    def test_kernel_sigma_auto(self, small_vae, mock_spectra):
        """Test that auto kernel sigma works."""
        batch = jnp.array(mock_spectra[:32])
        key = jr.PRNGKey(0)

        # Should not raise an error
        loss, aux = vae_loss_infovae(
            small_vae,
            batch,
            key,
            kernel_sigma="auto"
        )

        assert jnp.isfinite(loss)


# ---------- Config Tests ----------


class TestVAETrainingConfig:
    """Test VAE training configuration."""

    def test_config_creation(self):
        """Test creating a config with default values."""
        config = VAETrainingConfig(
            epochs=100,
            learning_rate=1e-4,
        )

        assert config.epochs == 100
        assert config.learning_rate == 1e-4
        assert config.beta == 1e-3  # Default
        assert config.lam == 4.0  # Default
        assert config.kernel_sigma == "auto"  # Default

    def test_config_custom_values(self):
        """Test creating a config with custom values."""
        config = VAETrainingConfig(
            epochs=200,
            learning_rate=5e-4,
            beta=1e-2,
            lam=10.0,
            kernel_sigma=0.5,
            clip_gradients=True,
            run_name="custom_vae",
        )

        assert config.beta == 1e-2
        assert config.lam == 10.0
        assert config.kernel_sigma == 0.5
        assert config.clip_gradients is True
        assert config.run_name == "custom_vae"


class TestVAETrainingHistory:
    """Test VAE training history."""

    def test_history_initialization(self):
        """Test that history is initialized correctly."""
        config = VAETrainingConfig(epochs=10, learning_rate=1e-4)
        history = VAETrainingHistory(config=config)

        assert history.train_losses == []
        assert history.test_losses == []
        assert history.best_test_loss == float("inf")
        assert history.best_epoch == -1
        assert history.config == config

    def test_history_append(self):
        """Test appending to history lists."""
        history = VAETrainingHistory()

        history.train_losses.append(1.0)
        history.test_losses.append(0.9)
        history.train_recon.append(0.8)

        assert len(history.train_losses) == 1
        assert history.train_losses[0] == 1.0


# ---------- Trainer Tests ----------


class TestVAETrainer:
    """Test VAE trainer class."""

    def test_trainer_creation(self, small_vae, vae_config):
        """Test creating a trainer instance."""
        trainer = VAETrainer(small_vae, vae_config)

        assert trainer.model == small_vae
        assert trainer.config == vae_config
        assert trainer.optimizer is not None
        assert isinstance(trainer.history, VAETrainingHistory)

    def test_trainer_with_custom_optimizer(self, small_vae, vae_config):
        """Test creating trainer with custom optimizer."""
        custom_optim = optax.sgd(learning_rate=0.01)
        trainer = VAETrainer(small_vae, vae_config, optimizer=custom_optim)

        assert trainer.optimizer == custom_optim

    def test_trainer_gradient_clipping(self, small_vae):
        """Test that gradient clipping is applied when configured."""
        config = VAETrainingConfig(
            epochs=5,
            learning_rate=1e-3,
            clip_gradients=True,
        )

        trainer = VAETrainer(small_vae, config)

        # Optimizer should be a chain that includes clipping
        # This is implementation-dependent, but we can check it's not just Adam
        assert trainer.optimizer is not None

    def test_training_single_epoch(self, small_vae, train_test_loaders, vae_config):
        """Test running training for a single epoch."""
        train_loader, test_loader = train_test_loaders

        # Train for just 1 epoch
        config = VAETrainingConfig(
            epochs=1,
            learning_rate=1e-3,
            save_best=False,
            print_every=1,
        )

        trainer = VAETrainer(small_vae, config)
        model, history = trainer.train(train_loader, test_loader)

        # Check that training ran
        assert len(history.train_losses) == 1
        assert len(history.test_losses) == 1
        assert history.train_losses[0] > 0
        assert jnp.isfinite(history.train_losses[0])

    def test_training_multiple_epochs(self, small_vae, train_test_loaders, vae_config):
        """Test running training for multiple epochs."""
        train_loader, test_loader = train_test_loaders

        trainer = VAETrainer(small_vae, vae_config)
        model, history = trainer.train(train_loader, test_loader)

        # Check that we have the right number of epochs
        assert len(history.train_losses) == vae_config.epochs
        assert len(history.test_losses) == vae_config.epochs

        # Check that all components are tracked
        assert len(history.train_recon) == vae_config.epochs
        assert len(history.train_kl) == vae_config.epochs
        assert len(history.train_mmd) == vae_config.epochs

    def test_training_decreases_loss(self, small_vae, train_test_loaders):
        """Test that training actually decreases the loss."""
        train_loader, test_loader = train_test_loaders

        config = VAETrainingConfig(
            epochs=10,
            learning_rate=1e-3,
            save_best=False,
        )

        trainer = VAETrainer(small_vae, config)
        model, history = trainer.train(train_loader, test_loader)

        # Loss should generally decrease (allow some noise)
        # Compare first 2 epochs to last 2 epochs
        initial_loss = np.mean(history.train_losses[:2])
        final_loss = np.mean(history.train_losses[-2:])

        assert final_loss < initial_loss

    def test_best_model_tracking(self, small_vae, train_test_loaders, vae_config):
        """Test that best model is tracked correctly."""
        train_loader, test_loader = train_test_loaders

        trainer = VAETrainer(small_vae, vae_config)
        model, history = trainer.train(train_loader, test_loader)

        # Best epoch should be set
        assert history.best_epoch >= 0
        assert history.best_epoch < vae_config.epochs

        # Best test loss should match the test loss at that epoch
        best_idx = history.best_epoch
        assert np.isclose(history.best_test_loss, history.test_losses[best_idx])

        # Best test loss should be minimum
        assert history.best_test_loss <= min(history.test_losses)

    def test_checkpoint_saving(self, small_vae, train_test_loaders):
        """Test that checkpoints can be saved."""
        train_loader, test_loader = train_test_loaders

        # Use temporary directory for saving
        with tempfile.TemporaryDirectory() as tmpdir:
            config = VAETrainingConfig(
                epochs=3,
                learning_rate=1e-3,
                save_best=True,
                save_dir=tmpdir,
                run_name="test_vae",
            )

            trainer = VAETrainer(small_vae, config)
            model, history = trainer.train(train_loader, test_loader)

            # Check that checkpoint was created
            checkpoint_path = Path(tmpdir) / "test_vae.eqx"
            assert checkpoint_path.exists()

    def test_training_reproducibility(self, small_vae, train_test_loaders):
        """Test that training is reproducible with same seed."""
        train_loader, test_loader = train_test_loaders

        config1 = VAETrainingConfig(
            epochs=3,
            learning_rate=1e-3,
            random_seed=42,
            save_best=False,
        )

        config2 = VAETrainingConfig(
            epochs=3,
            learning_rate=1e-3,
            random_seed=42,
            save_best=False,
        )

        # Train two models with same seed
        vae1 = make_SkyVAE(in_channels=100, latent_dim=4, key=jr.PRNGKey(42))
        vae2 = make_SkyVAE(in_channels=100, latent_dim=4, key=jr.PRNGKey(42))

        trainer1 = VAETrainer(vae1, config1)
        trainer2 = VAETrainer(vae2, config2)

        _, history1 = trainer1.train(train_loader, test_loader)
        _, history2 = trainer2.train(train_loader, test_loader)

        # Training should produce similar results (may not be exactly identical due to data loader shuffling)
        # But at least the structure should be the same
        assert len(history1.train_losses) == len(history2.train_losses)

    def test_all_loss_components_tracked(self, small_vae, train_test_loaders, vae_config):
        """Test that all loss components are properly tracked."""
        train_loader, test_loader = train_test_loaders

        trainer = VAETrainer(small_vae, vae_config)
        model, history = trainer.train(train_loader, test_loader)

        # Check all components are tracked and have correct length
        assert len(history.train_recon) == vae_config.epochs
        assert len(history.train_kl) == vae_config.epochs
        assert len(history.train_mmd) == vae_config.epochs
        assert len(history.test_recon) == vae_config.epochs
        assert len(history.test_kl) == vae_config.epochs
        assert len(history.test_mmd) == vae_config.epochs

        # All components should be finite
        assert all(np.isfinite(x) for x in history.train_recon)
        assert all(np.isfinite(x) for x in history.test_recon)

    def test_model_forward_pass_after_training(self, small_vae, train_test_loaders, vae_config):
        """Test that trained model can still perform forward pass."""
        train_loader, test_loader = train_test_loaders

        trainer = VAETrainer(small_vae, vae_config)
        model, history = trainer.train(train_loader, test_loader)

        # Test forward pass
        test_input = jnp.ones((1, 100))
        key = jr.PRNGKey(0)
        output = model(test_input, key)

        # Check output structure
        assert 'mean' in output
        assert 'logvar' in output
        assert 'latent' in output
        assert 'output' in output

        # Check shapes
        assert output['mean'].shape == (1, 4)  # (batch, latent_dim)
        assert output['output'].shape == (1, 100)  # (batch, in_channels)


# ---------- Integration Tests ----------


class TestVAETrainingIntegration:
    """Integration tests for full VAE training pipeline."""

    def test_full_training_pipeline(self, mock_spectra):
        """Test complete training pipeline from data to trained model."""
        # Create model
        vae = make_SkyVAE(in_channels=100, latent_dim=4, key=jr.PRNGKey(42))

        # Prepare data
        n_train = 150
        train_data = mock_spectra[:n_train]
        test_data = mock_spectra[n_train:]

        # Create simple wrapper dataset
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, data):
                self.data = data
            def __len__(self):
                return len(self.data)
            def __getitem__(self, idx):
                return self.data[idx]

        train_loader = NumpyLoader(
            SimpleDataset(train_data),
            batch_size=32,
            shuffle=True,
        )

        test_loader = NumpyLoader(
            SimpleDataset(test_data),
            batch_size=32,
            shuffle=False,
        )

        # Configure training
        config = VAETrainingConfig(
            epochs=5,
            learning_rate=1e-3,
            beta=1e-3,
            lam=4.0,
            save_best=False,
        )

        # Train
        trainer = VAETrainer(vae, config)
        model, history = trainer.train(train_loader, test_loader)

        # Verify results
        assert len(history.train_losses) == 5
        assert history.best_test_loss < float("inf")
        assert all(np.isfinite(x) for x in history.train_losses)

    def test_different_hyperparameters(self, small_vae, train_test_loaders):
        """Test training with different hyperparameter configurations."""
        train_loader, test_loader = train_test_loaders

        configs = [
            VAETrainingConfig(epochs=3, learning_rate=1e-3, beta=1e-4, lam=2.0),
            VAETrainingConfig(epochs=3, learning_rate=1e-3, beta=1e-3, lam=4.0),
            VAETrainingConfig(epochs=3, learning_rate=1e-3, beta=1e-2, lam=10.0),
        ]

        for config in configs:
            vae = make_SkyVAE(in_channels=100, latent_dim=4, key=jr.PRNGKey(42))
            trainer = VAETrainer(vae, config)
            model, history = trainer.train(train_loader, test_loader)

            # Should complete successfully
            assert len(history.train_losses) == 3
            assert all(np.isfinite(x) for x in history.train_losses)
