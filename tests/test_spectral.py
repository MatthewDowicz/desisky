# SPDX-FileCopyrightText: 2025-present MatthewDowicz <mjdowicz@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Tests for spectral feature extraction and convenience CDF visualizations."""

import pytest
import numpy as np

from desisky.data._spectral import (
    _integrate_band_with_linear_continuum,
    measure_airglow_intensities,
    compute_broadband_mags,
    LINE_BANDS,
    AIRGLOW_CDF_NAMES,
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from speclite.filters import load_filter  # noqa: F401
    import astropy.units  # noqa: F401

    SPECLITE_AVAILABLE = True
except ImportError:
    SPECLITE_AVAILABLE = False


@pytest.fixture
def desi_wavelength():
    return np.linspace(3600, 9824, 7781)


@pytest.fixture
def random_spectra(desi_wavelength):
    rng = np.random.default_rng(42)
    return rng.exponential(scale=5.0, size=(10, len(desi_wavelength))).astype(
        np.float32
    )


# ============================================================================
# Airglow Measurement
# ============================================================================


class TestMeasureAirglowIntensities:
    def test_returns_dataframe_with_all_lines(self, desi_wavelength, random_spectra):
        df = measure_airglow_intensities(desi_wavelength, random_spectra)
        assert len(df) == len(random_spectra)
        for name in LINE_BANDS:
            assert name in df.columns

    def test_oh_composite_is_sum_of_bands(self, desi_wavelength, random_spectra):
        df = measure_airglow_intensities(desi_wavelength, random_spectra)
        oh_cols = [c for c in df.columns if c.startswith("OH(")]
        np.testing.assert_allclose(df["OH"], df[oh_cols].sum(axis=1))

    def test_oi_doublet_is_sum(self, desi_wavelength, random_spectra):
        df = measure_airglow_intensities(desi_wavelength, random_spectra)
        np.testing.assert_allclose(
            df["OI doublet"], df["OI 6300"] + df["OI 6364"]
        )

    def test_returns_nan_for_out_of_range_wavelengths(self):
        wl = np.linspace(1000, 2000, 100)
        flux = np.ones((2, 100))
        df = measure_airglow_intensities(wl, flux)
        assert df["OI 5577"].isna().all()

    def test_custom_line_bands(self, desi_wavelength, random_spectra):
        custom = {"OI 5577": LINE_BANDS["OI 5577"]}
        df = measure_airglow_intensities(
            desi_wavelength, random_spectra, line_bands=custom
        )
        assert "OI 5577" in df.columns
        assert "Na I D" not in df.columns


# ============================================================================
# Broadband Magnitudes
# ============================================================================


@pytest.mark.skipif(not SPECLITE_AVAILABLE, reason="speclite not installed")
class TestComputeBroadbandMags:
    def test_shape_and_finite(self, desi_wavelength, random_spectra):
        mags = compute_broadband_mags(desi_wavelength, random_spectra)
        assert mags.shape == (len(random_spectra), 4)
        assert np.all(np.isfinite(mags))

    def test_custom_filters(self, desi_wavelength, random_spectra):
        mags = compute_broadband_mags(
            desi_wavelength,
            random_spectra[:3],
            filter_names=["bessell-V", "decam2014-g"],
        )
        assert mags.shape == (3, 2)


# ============================================================================
# Convenience Visualization Functions
# ============================================================================


@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib not installed")
@pytest.mark.skipif(not SPECLITE_AVAILABLE, reason="speclite not installed")
class TestPlotBroadbandCDFs:
    def test_returns_dict_of_figures(self, desi_wavelength, random_spectra):
        from desisky.visualization import plot_broadband_cdfs

        results = plot_broadband_cdfs(
            desi_wavelength, random_spectra[:5], random_spectra[5:]
        )
        assert set(results.keys()) == {"V", "g", "r", "z"}
        for fig, emd in results.values():
            assert isinstance(fig, Figure)
            assert emd >= 0
            plt.close(fig)


@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib not installed")
class TestPlotAirglowCDFs:
    def test_returns_dict_of_figures(self, desi_wavelength, random_spectra):
        from desisky.visualization import plot_airglow_cdfs

        results = plot_airglow_cdfs(
            desi_wavelength, random_spectra[:5], random_spectra[5:]
        )
        assert len(results) == len(AIRGLOW_CDF_NAMES)
        for fig, emd in results.values():
            assert isinstance(fig, Figure)
            plt.close(fig)

    def test_custom_line_names(self, desi_wavelength, random_spectra):
        from desisky.visualization import plot_airglow_cdfs

        results = plot_airglow_cdfs(
            desi_wavelength,
            random_spectra[:5],
            random_spectra[5:],
            line_names=["OI 5577", "OH"],
        )
        assert set(results.keys()) == {"OI 5577", "OH"}
        for fig, _ in results.values():
            plt.close(fig)
