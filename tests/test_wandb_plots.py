# SPDX-FileCopyrightText: 2025-present MatthewDowicz <mjdowicz@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Tests for wandb visualization functions.

Each plotting function is exercised with small random data.  We verify
that a Figure is returned, has the expected subplot structure, and can
be saved to disk.
"""

import pytest
import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend for testing
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not MATPLOTLIB_AVAILABLE, reason="matplotlib not installed"
)

from desisky.visualization import (
    plot_vae_reconstructions,
    plot_latent_corner,
    plot_latent_corner_comparison,
    plot_cdf_comparison,
    plot_conditional_validation_grid,
)


# ---------- Fixtures ----------


@pytest.fixture
def wavelength():
    """Mock wavelength grid (100 bins)."""
    return np.linspace(3600, 9800, 100)


@pytest.fixture
def spectra_pair(wavelength):
    """Original and reconstructed spectra (positive, for log-scale)."""
    rng = np.random.default_rng(42)
    n = 8
    originals = np.abs(rng.standard_normal((n, len(wavelength)))) + 0.1
    reconstructions = originals + rng.standard_normal(originals.shape) * 0.01
    reconstructions = np.abs(reconstructions) + 0.1
    return originals, reconstructions


@pytest.fixture
def latents():
    """Random latent codes, shape (200, 6)."""
    return np.random.default_rng(0).standard_normal((200, 6)).astype(np.float32)


@pytest.fixture
def feature_data():
    """Real and generated feature arrays, shape (100, 4)."""
    rng = np.random.default_rng(7)
    real = rng.standard_normal((100, 4))
    generated = rng.standard_normal((100, 4)) * 0.9  # slightly different
    return real, generated


# ---------- plot_vae_reconstructions ----------


class TestPlotVAEReconstructions:

    def test_returns_figure(self, spectra_pair, wavelength):
        originals, recons = spectra_pair
        fig = plot_vae_reconstructions(originals, recons, wavelength, n_samples=3)
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 3
        plt.close(fig)

    def test_single_sample(self, spectra_pair, wavelength):
        originals, recons = spectra_pair
        fig = plot_vae_reconstructions(originals, recons, wavelength, n_samples=1)
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_clamps_to_available(self, spectra_pair, wavelength):
        """n_samples > len(originals) should clamp without error."""
        originals, recons = spectra_pair
        fig = plot_vae_reconstructions(originals, recons, wavelength, n_samples=999)
        assert isinstance(fig, Figure)
        assert len(fig.axes) == len(originals)
        plt.close(fig)


# ---------- plot_latent_corner ----------


class TestPlotLatentCorner:

    def test_returns_figure(self, latents):
        fig = plot_latent_corner(latents)
        assert isinstance(fig, Figure)
        D = latents.shape[1]
        assert len(fig.axes) == D * D  # full grid (upper tri turned off)
        plt.close(fig)

    def test_with_sky_conditions(self, latents):
        rng = np.random.default_rng(1)
        conditions = rng.choice(["dark", "moon", "twilight"], size=len(latents))
        fig = plot_latent_corner(latents, sky_conditions=conditions)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_custom_labels(self, latents):
        labels = [f"dim{i}" for i in range(latents.shape[1])]
        fig = plot_latent_corner(latents, labels=labels)
        assert isinstance(fig, Figure)
        plt.close(fig)


# ---------- plot_latent_corner_comparison ----------


class TestPlotLatentCornerComparison:

    def test_returns_figure(self, latents):
        gen = np.random.default_rng(99).standard_normal(latents.shape).astype(np.float32)
        fig = plot_latent_corner_comparison(latents, gen)
        assert isinstance(fig, Figure)
        D = latents.shape[1]
        assert len(fig.axes) == D * D
        plt.close(fig)

    def test_different_sample_sizes(self):
        """Real and generated can have different N."""
        rng = np.random.default_rng(10)
        real = rng.standard_normal((100, 4)).astype(np.float32)
        gen = rng.standard_normal((50, 4)).astype(np.float32)
        fig = plot_latent_corner_comparison(real, gen)
        assert isinstance(fig, Figure)
        plt.close(fig)


# ---------- plot_cdf_comparison ----------


class TestPlotCDFComparison:

    def test_returns_figure_and_emd(self, feature_data):
        real, gen = feature_data
        names = ["V", "g", "r", "z"]
        fig, emd_dict = plot_cdf_comparison(real, gen, names)

        assert isinstance(fig, Figure)
        assert set(emd_dict.keys()) == set(names)
        assert all(v >= 0 for v in emd_dict.values())
        plt.close(fig)

    def test_single_feature(self):
        rng = np.random.default_rng(3)
        real = rng.standard_normal((50, 1))
        gen = rng.standard_normal((50, 1))
        fig, emd_dict = plot_cdf_comparison(real, gen, ["x"])

        assert isinstance(fig, Figure)
        assert "x" in emd_dict
        plt.close(fig)

    def test_1d_arrays_accepted(self):
        """1D arrays should be auto-expanded to (N, 1)."""
        rng = np.random.default_rng(5)
        real = rng.standard_normal(80)
        gen = rng.standard_normal(80)
        fig, emd_dict = plot_cdf_comparison(real, gen, ["feat"])

        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_emd_zero_for_identical(self):
        """EMD should be 0 when real == generated."""
        data = np.random.default_rng(9).standard_normal((100, 2))
        _, emd_dict = plot_cdf_comparison(data, data, ["a", "b"])
        for v in emd_dict.values():
            assert v == pytest.approx(0.0, abs=1e-10)
        plt.close("all")


# ---------- plot_conditional_validation_grid ----------


class TestPlotConditionalValidationGrid:

    def test_returns_figure(self, feature_data):
        real, gen = feature_data
        cond = np.random.default_rng(2).uniform(0.5, 1.5, size=len(real))
        names = ["V", "g", "r", "z"]

        fig = plot_conditional_validation_grid(
            real, gen, cond, cond_name="TRANSPARENCY", feature_names=names,
        )

        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_fewer_features_than_ncols(self):
        """Should work when n_features < ncols (unused axes hidden)."""
        rng = np.random.default_rng(4)
        real = rng.standard_normal((80, 2))
        gen = rng.standard_normal((80, 2))
        cond = rng.uniform(size=80)

        fig = plot_conditional_validation_grid(
            real, gen, cond, "x", ["a", "b"], ncols=4,
        )

        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_save_to_file(self, feature_data, tmp_path):
        real, gen = feature_data
        cond = np.random.default_rng(6).uniform(size=len(real))

        fig = plot_conditional_validation_grid(
            real, gen, cond, "cond", ["a", "b", "c", "d"],
        )
        path = tmp_path / "grid.png"
        fig.savefig(path)
        assert path.exists()
        plt.close(fig)
