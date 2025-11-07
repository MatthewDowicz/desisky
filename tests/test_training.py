# SPDX-FileCopyrightText: 2025-present MatthewDowicz <mjdowicz@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Tests for training utilities."""

import pytest
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import torch
from torch.utils.data import random_split

from desisky.training import (
    SkyBrightnessDataset,
    NumpyLoader,
    numpy_collate,
    gather_full_data,
    loss_l2,
    loss_huber,
    loss_func,
    BroadbandTrainer,
    TrainingConfig,
)


# ---------- Fixtures ----------


@pytest.fixture
def mock_metadata():
    """Create mock metadata DataFrame."""
    np.random.seed(42)
    n_samples = 100
    return pd.DataFrame(
        {
            "MOONSEP": np.random.uniform(0, 90, n_samples),
            "MOONFRAC": np.random.uniform(0.5, 1.0, n_samples),
            "MOONALT": np.random.uniform(5, 60, n_samples),
            "OBSALT": np.random.uniform(30, 80, n_samples),
            "TRANSPARENCY_GFA": np.random.uniform(0.5, 1.0, n_samples),
            "ECLIPSE_FRAC": np.random.uniform(0, 0.3, n_samples),
            "SKY_MAG_V_SPEC": np.random.uniform(18, 21, n_samples),
            "SKY_MAG_G_SPEC": np.random.uniform(18, 21, n_samples),
            "SKY_MAG_R_SPEC": np.random.uniform(18, 21, n_samples),
            "SKY_MAG_Z_SPEC": np.random.uniform(18, 21, n_samples),
        }
    )


@pytest.fixture
def mock_flux():
    """Create mock flux array."""
    np.random.seed(42)
    n_samples = 100
    n_wavelengths = 50
    return np.random.uniform(0.1, 10.0, (n_samples, n_wavelengths)).astype("float32")


@pytest.fixture
def input_features():
    """Standard input feature list."""
    return [
        "MOONSEP",
        "MOONFRAC",
        "MOONALT",
        "OBSALT",
        "TRANSPARENCY_GFA",
        "ECLIPSE_FRAC",
    ]


@pytest.fixture
def mock_model():
    """Create a small MLP for testing."""
    return eqx.nn.MLP(
        in_size=6, out_size=4, width_size=16, depth=2, key=jax.random.PRNGKey(0)
    )


# ---------- Dataset Tests ----------


def test_dataset_creation(mock_metadata, mock_flux, input_features):
    """Test that SkyBrightnessDataset can be created."""
    dataset = SkyBrightnessDataset(mock_metadata, mock_flux, input_features)

    assert len(dataset) == 100
    assert dataset.input_features == input_features
    assert dataset.targets.shape == (100, 4)


def test_dataset_getitem(mock_metadata, mock_flux, input_features):
    """Test that __getitem__ returns correct shapes."""
    dataset = SkyBrightnessDataset(mock_metadata, mock_flux, input_features)

    inputs, targets, spectrum = dataset[0]

    assert inputs.shape == (6,)
    assert targets.shape == (4,)
    assert spectrum.shape == (50,)
    assert inputs.dtype == np.float32


def test_dataset_missing_features_raises(mock_metadata, mock_flux):
    """Test that missing input features raise ValueError."""
    with pytest.raises(ValueError, match="Input features not found"):
        SkyBrightnessDataset(mock_metadata, mock_flux, ["NONEXISTENT_FEATURE"])


def test_dataset_missing_targets_raises(mock_metadata, mock_flux, input_features):
    """Test that missing target columns raise ValueError."""
    meta_bad = mock_metadata.drop(columns=["SKY_MAG_V_SPEC"])

    with pytest.raises(ValueError, match="Target columns not found"):
        SkyBrightnessDataset(meta_bad, mock_flux, input_features)


# ---------- DataLoader Tests ----------


def test_numpy_collate_arrays():
    """Test numpy_collate with arrays."""
    batch = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    result = numpy_collate(batch)

    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 3)
    np.testing.assert_array_equal(result, [[1, 2, 3], [4, 5, 6]])


def test_numpy_collate_tuples():
    """Test numpy_collate with tuples."""
    batch = [
        (np.array([1, 2]), np.array([3, 4])),
        (np.array([5, 6]), np.array([7, 8])),
    ]
    result = numpy_collate(batch)

    assert isinstance(result, list)
    assert len(result) == 2
    np.testing.assert_array_equal(result[0], [[1, 2], [5, 6]])
    np.testing.assert_array_equal(result[1], [[3, 4], [7, 8]])


def test_numpy_loader(mock_metadata, mock_flux, input_features):
    """Test NumpyLoader returns numpy arrays."""
    dataset = SkyBrightnessDataset(mock_metadata, mock_flux, input_features)
    loader = NumpyLoader(dataset, batch_size=8, shuffle=False)

    batch = next(iter(loader))
    inputs, targets, spectra = batch

    assert isinstance(inputs, np.ndarray)
    assert isinstance(targets, np.ndarray)
    assert isinstance(spectra, np.ndarray)
    assert inputs.shape == (8, 6)
    assert targets.shape == (8, 4)
    assert spectra.shape == (8, 50)


def test_gather_full_data(mock_metadata, mock_flux, input_features):
    """Test gather_full_data returns all data correctly."""
    dataset = SkyBrightnessDataset(mock_metadata, mock_flux, input_features)
    loader = NumpyLoader(dataset, batch_size=16, shuffle=False)

    all_inputs, all_targets, all_spectra, meta_df = gather_full_data(loader)

    assert all_inputs.shape == (100, 6)
    assert all_targets.shape == (100, 4)
    assert all_spectra.shape == (100, 50)
    assert len(meta_df) == 100
    assert isinstance(meta_df, pd.DataFrame)


def test_gather_full_data_with_subset(mock_metadata, mock_flux, input_features):
    """Test gather_full_data works with train/test split."""
    dataset = SkyBrightnessDataset(mock_metadata, mock_flux, input_features)
    train_set, test_set = random_split(
        dataset, [70, 30], generator=torch.Generator().manual_seed(42)
    )

    train_loader = NumpyLoader(train_set, batch_size=16, shuffle=False)
    all_inputs, all_targets, all_spectra, meta_df = gather_full_data(train_loader)

    assert all_inputs.shape == (70, 6)
    assert all_targets.shape == (70, 4)
    assert all_spectra.shape == (70, 50)
    assert len(meta_df) == 70


# ---------- Loss Function Tests ----------


def test_loss_l2():
    """Test L2 loss computation."""
    pred = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    targets = jnp.array([[1.1, 2.1], [2.9, 3.9]])

    loss = loss_l2(pred, targets)

    expected = jnp.mean((pred - targets) ** 2)
    np.testing.assert_allclose(loss, expected, rtol=1e-5)


def test_loss_huber():
    """Test Huber loss computation."""
    pred = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    targets = jnp.array([[1.1, 2.1], [2.9, 3.9]])

    loss = loss_huber(pred, targets, delta=0.5)

    assert isinstance(loss, jnp.ndarray)
    assert loss.shape == ()  # scalar


def test_loss_func_l2(mock_model):
    """Test loss_func with L2 loss."""
    inputs = jnp.ones((16, 6))
    targets = jnp.zeros((16, 4))

    loss = loss_func(mock_model, inputs, targets, loss_name="l2")

    assert isinstance(loss, jnp.ndarray)
    assert loss.shape == ()  # scalar


def test_loss_func_huber(mock_model):
    """Test loss_func with Huber loss."""
    inputs = jnp.ones((16, 6))
    targets = jnp.zeros((16, 4))

    loss = loss_func(mock_model, inputs, targets, loss_name="huber", huber_delta=0.5)

    assert isinstance(loss, jnp.ndarray)
    assert loss.shape == ()  # scalar


def test_loss_func_invalid_loss_raises(mock_model):
    """Test that invalid loss name raises ValueError."""
    inputs = jnp.ones((16, 6))
    targets = jnp.zeros((16, 4))

    with pytest.raises(ValueError, match="Unknown loss"):
        loss_func(mock_model, inputs, targets, loss_name="invalid")


# ---------- Trainer Tests ----------


def test_training_config_creation():
    """Test TrainingConfig can be created."""
    config = TrainingConfig(
        epochs=10, learning_rate=1e-3, loss="huber", huber_delta=0.5
    )

    assert config.epochs == 10
    assert config.learning_rate == 1e-3
    assert config.loss == "huber"
    assert config.huber_delta == 0.5


def test_broadband_trainer_creation(mock_model):
    """Test BroadbandTrainer can be created."""
    config = TrainingConfig(epochs=10, learning_rate=1e-3)
    trainer = BroadbandTrainer(mock_model, config)

    assert trainer.model == mock_model
    assert trainer.config == config
    assert trainer.history.best_test_loss == float("inf")


def test_trainer_with_custom_optimizer(mock_model):
    """Test BroadbandTrainer with custom optimizer."""
    config = TrainingConfig(epochs=10, learning_rate=1e-3)
    optimizer = optax.sgd(1e-3)
    trainer = BroadbandTrainer(mock_model, config, optimizer=optimizer)

    assert trainer.optimizer == optimizer


def test_trainer_train_smoke(mock_metadata, mock_flux, input_features, mock_model, tmp_path):
    """Smoke test for training loop (few epochs)."""
    dataset = SkyBrightnessDataset(mock_metadata, mock_flux, input_features)
    train_set, test_set = random_split(
        dataset, [70, 30], generator=torch.Generator().manual_seed(42)
    )

    train_loader = NumpyLoader(train_set, batch_size=16, shuffle=True)
    test_loader = NumpyLoader(test_set, batch_size=16, shuffle=False)

    config = TrainingConfig(
        epochs=3,
        learning_rate=1e-3,
        loss="l2",
        save_best=True,
        save_dir=tmp_path,
        run_name="test_model",
        print_every=1,
    )

    trainer = BroadbandTrainer(mock_model, config)
    trained_model, history = trainer.train(train_loader, test_loader)

    # Check that training ran
    assert len(history.train_losses) == 3
    assert len(history.test_losses) == 3
    assert history.best_test_loss < float("inf")
    assert history.best_epoch >= 0

    # Check that checkpoint was saved
    checkpoint_path = tmp_path / "test_model.eqx"
    assert checkpoint_path.exists()


def test_trainer_extract_architecture(mock_model):
    """Test architecture extraction from MLP."""
    config = TrainingConfig(epochs=1, learning_rate=1e-3)
    trainer = BroadbandTrainer(mock_model, config)

    arch = trainer._extract_architecture()

    assert arch["in_size"] == 6
    assert arch["out_size"] == 4
    assert arch["width_size"] == 16
    assert arch["depth"] == 2


def test_trainer_compute_per_band_rmse(mock_model):
    """Test per-band RMSE computation."""
    config = TrainingConfig(epochs=1, learning_rate=1e-3)
    trainer = BroadbandTrainer(mock_model, config)

    inputs = np.random.randn(50, 6).astype("float32")
    targets = np.random.randn(50, 4).astype("float32")

    rmse = trainer._compute_per_band_rmse(inputs, targets)

    assert len(rmse) == 4
    assert all(isinstance(x, float) for x in rmse)
    assert all(x >= 0 for x in rmse)
