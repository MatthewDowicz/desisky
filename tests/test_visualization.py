# SPDX-FileCopyrightText: 2025-present MatthewDowicz <mjdowicz@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Tests for visualization utilities."""

import pytest
import numpy as np
import pandas as pd
import jax
import equinox as eqx

try:
    import matplotlib

    matplotlib.use("Agg")  # Use non-interactive backend for testing
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from desisky.visualization import plot_loss_curve, plot_nn_outlier_analysis


# ---------- Fixtures ----------


@pytest.fixture
def mock_losses():
    """Create mock training and test losses."""
    np.random.seed(42)
    epochs = 50
    train_losses = 1.0 * np.exp(-np.arange(epochs) * 0.05) + np.random.randn(epochs) * 0.05
    test_losses = 1.2 * np.exp(-np.arange(epochs) * 0.04) + np.random.randn(epochs) * 0.06
    return train_losses.tolist(), test_losses.tolist()


@pytest.fixture
def mock_model():
    """Create a small MLP for testing."""
    return eqx.nn.MLP(
        in_size=6, out_size=4, width_size=16, depth=2, key=jax.random.PRNGKey(0)
    )


@pytest.fixture
def mock_data():
    """Create mock training and test data."""
    np.random.seed(42)
    n_train, n_test = 100, 50

    X_train = np.random.randn(n_train, 6).astype("float32")
    y_train = np.random.uniform(18, 21, (n_train, 4)).astype("float32")
    meta_train = pd.DataFrame(
        {
            "TRANSPARENCY_GFA": np.random.uniform(0.5, 1.0, n_train),
            "MOONFRAC": np.random.uniform(0.5, 1.0, n_train),
        }
    )

    X_test = np.random.randn(n_test, 6).astype("float32")
    y_test = np.random.uniform(18, 21, (n_test, 4)).astype("float32")
    meta_test = pd.DataFrame(
        {
            "TRANSPARENCY_GFA": np.random.uniform(0.5, 1.0, n_test),
            "MOONFRAC": np.random.uniform(0.5, 1.0, n_test),
        }
    )

    return X_train, y_train, meta_train, X_test, y_test, meta_test


# ---------- Skip if matplotlib not available ----------


pytestmark = pytest.mark.skipif(
    not MATPLOTLIB_AVAILABLE, reason="matplotlib not installed"
)


# ---------- Loss Curve Tests ----------


def test_plot_loss_curve_basic(mock_losses):
    """Test basic loss curve plotting."""
    train_losses, test_losses = mock_losses

    fig = plot_loss_curve(train_losses, test_losses)

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1

    plt.close(fig)


def test_plot_loss_curve_custom_title(mock_losses):
    """Test loss curve with custom title."""
    train_losses, test_losses = mock_losses

    fig = plot_loss_curve(train_losses, test_losses, title="Custom Title")

    assert isinstance(fig, Figure)
    assert fig._suptitle.get_text() == "Custom Title"

    plt.close(fig)


def test_plot_loss_curve_custom_figsize(mock_losses):
    """Test loss curve with custom figure size."""
    train_losses, test_losses = mock_losses

    fig = plot_loss_curve(train_losses, test_losses, figsize=(10, 6))

    assert isinstance(fig, Figure)
    # Figure size is in inches
    assert fig.get_size_inches()[0] == pytest.approx(10, abs=0.1)
    assert fig.get_size_inches()[1] == pytest.approx(6, abs=0.1)

    plt.close(fig)


def test_plot_loss_curve_save(mock_losses, tmp_path):
    """Test saving loss curve to file."""
    train_losses, test_losses = mock_losses
    save_path = tmp_path / "loss_curve.png"

    fig = plot_loss_curve(train_losses, test_losses, save_path=str(save_path))

    assert save_path.exists()

    plt.close(fig)


# ---------- Outlier Analysis Tests ----------


def test_plot_nn_outlier_analysis_basic(mock_model, mock_data):
    """Test basic outlier analysis plotting."""
    X_train, y_train, meta_train, X_test, y_test, meta_test = mock_data

    fig = plot_nn_outlier_analysis(
        mock_model,
        X_train,
        y_train,
        meta_train,
        X_test,
        y_test,
        meta_test,
        band_idx=0,
    )

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 6  # 2 rows × 3 columns

    plt.close(fig)


def test_plot_nn_outlier_analysis_all_bands(mock_model, mock_data):
    """Test outlier analysis for all bands."""
    X_train, y_train, meta_train, X_test, y_test, meta_test = mock_data

    for band_idx in range(4):
        fig = plot_nn_outlier_analysis(
            mock_model,
            X_train,
            y_train,
            meta_train,
            X_test,
            y_test,
            meta_test,
            band_idx=band_idx,
        )

        assert isinstance(fig, Figure)
        plt.close(fig)


def test_plot_nn_outlier_analysis_custom_outlier_threshold(mock_model, mock_data):
    """Test outlier analysis with custom threshold."""
    X_train, y_train, meta_train, X_test, y_test, meta_test = mock_data

    fig = plot_nn_outlier_analysis(
        mock_model,
        X_train,
        y_train,
        meta_train,
        X_test,
        y_test,
        meta_test,
        band_idx=0,
        outlier_mag=0.5,
    )

    assert isinstance(fig, Figure)

    plt.close(fig)


def test_plot_nn_outlier_analysis_custom_xlim(mock_model, mock_data):
    """Test outlier analysis with custom x-limits."""
    X_train, y_train, meta_train, X_test, y_test, meta_test = mock_data

    fig = plot_nn_outlier_analysis(
        mock_model,
        X_train,
        y_train,
        meta_train,
        X_test,
        y_test,
        meta_test,
        band_idx=0,
        xlim=(0.4, 1.0),
    )

    assert isinstance(fig, Figure)

    plt.close(fig)


def test_plot_nn_outlier_analysis_save(mock_model, mock_data, tmp_path):
    """Test saving outlier analysis to file."""
    X_train, y_train, meta_train, X_test, y_test, meta_test = mock_data
    save_path = tmp_path / "outlier_analysis.png"

    fig = plot_nn_outlier_analysis(
        mock_model,
        X_train,
        y_train,
        meta_train,
        X_test,
        y_test,
        meta_test,
        band_idx=0,
        save_path=str(save_path),
    )

    assert save_path.exists()

    plt.close(fig)


def test_plot_nn_outlier_analysis_with_pwv(mock_model, mock_data):
    """Test outlier analysis with PWV_los column (instead of TRANSPARENCY_GFA)."""
    X_train, y_train, meta_train, X_test, y_test, meta_test = mock_data

    # Add PWV_los column (should be preferred over TRANSPARENCY_GFA)
    meta_train["PWV_los"] = np.random.uniform(2, 8, len(meta_train))
    meta_test["PWV_los"] = np.random.uniform(2, 8, len(meta_test))

    fig = plot_nn_outlier_analysis(
        mock_model,
        X_train,
        y_train,
        meta_train,
        X_test,
        y_test,
        meta_test,
        band_idx=0,
    )

    assert isinstance(fig, Figure)

    plt.close(fig)
