# SPDX-FileCopyrightText: 2025-present MatthewDowicz <mjdowicz@gmail.com>
#
# SPDX-License-Identifier: MIT

"""
Tests for VAC enrichment functionality (_enrich.py module and SkySpecVAC enrichment).
"""

import socket

import pytest
import numpy as np
import pandas as pd
from pathlib import Path


def _nasa_reachable(host="eclipse.gsfc.nasa.gov", port=443, timeout=5):
    """Return True if NASA eclipse server is reachable (best-effort check)."""
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.close()
        return True
    except OSError:
        return False


_skip_no_nasa = pytest.mark.skipif(
    not _nasa_reachable(),
    reason="NASA eclipse server (eclipse.gsfc.nasa.gov) unreachable",
)


class TestVBandComputation:
    """Tests for V-band magnitude computation."""

    def test_compute_vband_requires_speclite(self):
        """Test that compute_vband_magnitudes raises ImportError without speclite."""
        from desisky.data._enrich import compute_vband_magnitudes

        # This will only fail if speclite is actually not installed
        # In normal test environment, speclite should be available
        flux = np.random.rand(10, 100)
        wavelength = np.linspace(3600, 9800, 100)

        try:
            result = compute_vband_magnitudes(flux, wavelength)
            assert result.shape == (10,)
            assert np.all(np.isfinite(result))
        except ImportError as e:
            assert "speclite" in str(e).lower()

    def test_vband_output_shape(self):
        """Test that V-band computation returns correct shape."""
        from desisky.data._enrich import compute_vband_magnitudes

        pytest.importorskip("speclite")

        n_spectra = 5
        n_wavelength = 100
        flux = np.random.rand(n_spectra, n_wavelength) * 1e-17
        wavelength = np.linspace(3600, 9800, n_wavelength)

        result = compute_vband_magnitudes(flux, wavelength)

        assert result.shape == (n_spectra,)
        assert result.dtype == np.float64

    def test_vband_values_reasonable(self):
        """Test that V-band magnitudes are in reasonable range for sky spectra."""
        from desisky.data._enrich import compute_vband_magnitudes

        pytest.importorskip("speclite")

        # Typical DESI sky spectrum flux levels (0.1-1 in units of 1e-17)
        flux = np.random.rand(10, 100) * 0.5 + 0.5  # range ~0.5-1.0
        wavelength = np.linspace(3600, 9800, 100)

        result = compute_vband_magnitudes(flux, wavelength)

        # Sky magnitudes typically 16-23 mag/arcsec^2
        assert np.all(result > 10)  # Not absurdly bright
        assert np.all(result < 30)  # Not absurdly faint
        assert np.all(np.isfinite(result))  # No NaNs or infs


class TestEclipseCatalog:
    """Tests for eclipse catalog loading."""

    @_skip_no_nasa
    def test_load_eclipse_catalog_downloads(self, tmp_path, monkeypatch):
        """Test that eclipse catalog downloads if missing."""
        from desisky.data._enrich import load_eclipse_catalog

        pytest.importorskip("pandas")
        pytest.importorskip("astropy")

        # Use temporary directory
        df = load_eclipse_catalog(root=tmp_path, download=True)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 30  # DESI DR1 window
        assert 'MJD' in df.columns
        assert 'P1' in df.columns
        assert 'P4' in df.columns

    def test_eclipse_catalog_raises_without_download(self, tmp_path):
        """Test that missing catalog raises FileNotFoundError when download=False."""
        from desisky.data._enrich import load_eclipse_catalog

        pytest.importorskip("pandas")
        pytest.importorskip("astropy")

        fake_path = tmp_path / "nonexistent" / "5MKLEcatalog.txt"

        with pytest.raises(FileNotFoundError, match="Eclipse catalog not found"):
            load_eclipse_catalog(catalog_path=fake_path, download=False)

    @_skip_no_nasa
    def test_eclipse_catalog_contact_times(self):
        """Test that contact times are computed correctly."""
        from desisky.data._enrich import load_eclipse_catalog

        pytest.importorskip("pandas")
        pytest.importorskip("astropy")

        df = load_eclipse_catalog(download=True)

        # All eclipses should have P1 < P4
        assert np.all(df['P1'] < df['P4'])

        # Partial/Total eclipses should have U1 < U4 (where not NaN)
        partial_total = df[df['Ecl_Type'].str.startswith(('P', 'T'))]
        assert np.all(partial_total['U1'] < partial_total['U4'])

        # Total eclipses should have U2 < U3 (where not NaN)
        total = df[df['Ecl_Type'].str.startswith('T')]
        valid_u2u3 = total.dropna(subset=['U2', 'U3'])
        if len(valid_u2u3) > 0:
            assert np.all(valid_u2u3['U2'] < valid_u2u3['U3'])


class TestEclipseFractionComputation:
    """Tests for ECLIPSE_FRAC computation."""

    @_skip_no_nasa
    def test_eclipse_fraction_shape(self):
        """Test that eclipse fraction returns correct shape."""
        from desisky.data._enrich import compute_eclipse_fraction

        pytest.importorskip("pandas")
        pytest.importorskip("astropy")

        # Create fake metadata
        metadata = pd.DataFrame({
            'MJD': np.linspace(59000, 59800, 100)
        })

        result = compute_eclipse_fraction(metadata, download=True)

        assert result.shape == (100,)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    @_skip_no_nasa
    def test_eclipse_fraction_zeros_outside_eclipses(self):
        """Test that eclipse fraction is zero for times with no eclipses."""
        from desisky.data._enrich import compute_eclipse_fraction

        pytest.importorskip("pandas")
        pytest.importorskip("astropy")

        # Times way outside DESI DR1 window
        metadata = pd.DataFrame({
            'MJD': np.linspace(50000, 50100, 100)  # ~1995
        })

        result = compute_eclipse_fraction(metadata, download=True)

        # Should be all zeros (no eclipses in catalog for this time range)
        assert np.all(result == 0)


class TestSolarFluxAttachment:
    """Tests for solar flux attachment functionality."""

    def test_attach_solar_flux_basic(self):
        """Test basic solar flux attachment."""
        from desisky.data import attach_solar_flux

        pytest.importorskip("pandas")

        # Create fake metadata (MJD 59000 = 2020-05-31)
        metadata = pd.DataFrame({
            'MJD': np.array([59000.0, 59001.0, 59002.0]),
            'EXPID': [1, 2, 3]
        })

        # Create fake solar flux data matching the MJD dates
        solar_df = pd.DataFrame({
            'datetime': pd.to_datetime(['2020-05-31', '2020-06-01', '2020-06-02']),
            'fluxobsflux': [150.0, 155.0, 160.0]
        })

        result = attach_solar_flux(metadata, solar_df, verbose=False)

        assert 'SOLFLUX' in result.columns
        assert len(result) == 3
        assert result['SOLFLUX'].notna().all()
        assert result['EXPID'].tolist() == [1, 2, 3]

    def test_attach_solar_flux_preserves_data(self):
        """Test that original metadata columns are preserved."""
        from desisky.data import attach_solar_flux

        pytest.importorskip("pandas")

        metadata = pd.DataFrame({
            'MJD': [59000.0, 59001.0],
            'EXPID': [1, 2],
            'TILERA': [150.0, 160.0],
            'TILEDEC': [30.0, 35.0]
        })

        solar_df = pd.DataFrame({
            'datetime': pd.to_datetime(['2020-05-31', '2020-06-01']),
            'fluxobsflux': [150.0, 155.0]
        })

        result = attach_solar_flux(metadata, solar_df, verbose=False)

        # Check original columns preserved
        assert 'TILERA' in result.columns
        assert 'TILEDEC' in result.columns
        assert result['TILERA'].tolist() == [150.0, 160.0]
        assert result['TILEDEC'].tolist() == [30.0, 35.0]

    def test_attach_solar_flux_no_mutation(self):
        """Test that original dataframes are not mutated."""
        from desisky.data import attach_solar_flux

        pytest.importorskip("pandas")

        metadata = pd.DataFrame({
            'MJD': [59000.0, 59001.0],
            'EXPID': [1, 2]
        })

        solar_df = pd.DataFrame({
            'datetime': pd.to_datetime(['2020-05-31', '2020-06-01']),
            'fluxobsflux': [150.0, 155.0]
        })

        # Store original lengths
        orig_meta_cols = set(metadata.columns)
        orig_solar_cols = set(solar_df.columns)

        result = attach_solar_flux(metadata, solar_df, verbose=False)

        # Originals should be unchanged
        assert set(metadata.columns) == orig_meta_cols
        assert set(solar_df.columns) == orig_solar_cols
        assert 'SOLFLUX' not in metadata.columns

    def test_attach_solar_flux_with_gaps(self):
        """Test handling of time gaps beyond tolerance."""
        from desisky.data import attach_solar_flux

        pytest.importorskip("pandas")

        # Observations far from solar flux measurements
        metadata = pd.DataFrame({
            'MJD': [59000.0, 59010.0]  # 10 days apart (2020-05-31 and 2020-06-10)
        })

        # Solar flux only on day 59000 (2020-05-31)
        solar_df = pd.DataFrame({
            'datetime': pd.to_datetime(['2020-05-31']),
            'fluxobsflux': [150.0]
        })

        # With 1-day tolerance, second observation should have NaN
        result = attach_solar_flux(metadata, solar_df, time_tolerance="1D", verbose=False)

        assert result['SOLFLUX'].notna().sum() == 1
        assert result['SOLFLUX'].isna().sum() == 1

    def test_load_solar_flux(self):
        """Test load_solar_flux downloads and loads data correctly."""
        from desisky.data import load_solar_flux

        pytest.importorskip("pandas")

        # This will download from HuggingFace on first run, or use cached version
        solar_df = load_solar_flux(download=True, verify=True)

        # Check structure
        assert 'datetime' in solar_df.columns
        assert 'fluxobsflux' in solar_df.columns
        assert len(solar_df) > 0

        # Check that datetime column is parsed correctly
        assert solar_df['datetime'].dtype.name.startswith('datetime')

    def test_attach_solar_flux_auto_download(self):
        """Test that attach_solar_flux auto-downloads data if not provided."""
        from desisky.data import attach_solar_flux

        pytest.importorskip("pandas")

        metadata = pd.DataFrame({
            'MJD': [59000.0, 59001.0],
            'EXPID': [1, 2]
        })

        # Don't provide solar_flux_df - should auto-download
        result = attach_solar_flux(metadata, solar_flux_df=None, verbose=False)

        # Should have SOLFLUX column
        assert 'SOLFLUX' in result.columns
        assert len(result) == 2


class TestGalacticCoordinates:
    """Tests for Galactic coordinate transformation."""

    def test_add_galactic_coordinates_basic(self):
        """Test basic Galactic coordinate addition."""
        from desisky.data import add_galactic_coordinates

        pytest.importorskip("astropy")

        metadata = pd.DataFrame({
            'TILERA': [0.0, 90.0, 180.0],
            'TILEDEC': [0.0, 0.0, 0.0]
        })

        result = add_galactic_coordinates(metadata)

        assert 'GALLON' in result.columns
        assert 'GALLAT' in result.columns
        assert len(result) == 3
        assert result['GALLON'].notna().all()
        assert result['GALLAT'].notna().all()

    def test_galactic_coordinates_range(self):
        """Test that Galactic coordinates are in valid ranges."""
        from desisky.data import add_galactic_coordinates

        pytest.importorskip("astropy")

        metadata = pd.DataFrame({
            'TILERA': np.linspace(0, 360, 20),
            'TILEDEC': np.linspace(-90, 90, 20)
        })

        result = add_galactic_coordinates(metadata)

        # Galactic longitude: 0-360
        assert (result['GALLON'] >= 0).all()
        assert (result['GALLON'] <= 360).all()

        # Galactic latitude: -90 to 90
        assert (result['GALLAT'] >= -90).all()
        assert (result['GALLAT'] <= 90).all()

    def test_galactic_coordinates_no_mutation(self):
        """Test that original metadata is not mutated."""
        from desisky.data import add_galactic_coordinates

        pytest.importorskip("astropy")

        metadata = pd.DataFrame({
            'TILERA': [150.0],
            'TILEDEC': [30.0]
        })

        orig_cols = set(metadata.columns)
        result = add_galactic_coordinates(metadata)

        # Original unchanged
        assert set(metadata.columns) == orig_cols
        assert 'GALLON' not in metadata.columns


class TestEclipticCoordinates:
    """Tests for Ecliptic coordinate transformation."""

    def test_add_ecliptic_coordinates_basic(self):
        """Test basic Ecliptic coordinate addition."""
        from desisky.data import add_ecliptic_coordinates

        pytest.importorskip("astropy")

        metadata = pd.DataFrame({
            'TILERA': [0.0, 90.0, 180.0],
            'TILEDEC': [0.0, 0.0, 0.0]
        })

        result = add_ecliptic_coordinates(metadata)

        assert 'ECLLON' in result.columns
        assert 'ECLLAT' in result.columns
        assert len(result) == 3
        assert result['ECLLON'].notna().all()
        assert result['ECLLAT'].notna().all()

    def test_ecliptic_coordinates_range(self):
        """Test that Ecliptic coordinates are in valid ranges."""
        from desisky.data import add_ecliptic_coordinates

        pytest.importorskip("astropy")

        metadata = pd.DataFrame({
            'TILERA': np.linspace(0, 360, 20),
            'TILEDEC': np.linspace(-90, 90, 20)
        })

        result = add_ecliptic_coordinates(metadata)

        # Ecliptic longitude: 0-360
        assert (result['ECLLON'] >= 0).all()
        assert (result['ECLLON'] <= 360).all()

        # Ecliptic latitude: -90 to 90
        assert (result['ECLLAT'] >= -90).all()
        assert (result['ECLLAT'] <= 90).all()

    def test_ecliptic_geocentric_default(self):
        """Test that geocentric frame works correctly."""
        from desisky.data import add_ecliptic_coordinates

        pytest.importorskip("astropy")

        metadata = pd.DataFrame({
            'TILERA': [150.0],
            'TILEDEC': [30.0]
        })

        geo = add_ecliptic_coordinates(metadata)

        # Should have ecliptic coordinates
        assert 'ECLLON' in geo.columns
        assert 'ECLLAT' in geo.columns
        assert geo['ECLLON'].notna().all()
        assert geo['ECLLAT'].notna().all()


    def test_ecliptic_coordinates_no_mutation(self):
        """Test that original metadata is not mutated."""
        from desisky.data import add_ecliptic_coordinates

        pytest.importorskip("astropy")

        metadata = pd.DataFrame({
            'TILERA': [150.0],
            'TILEDEC': [30.0]
        })

        orig_cols = set(metadata.columns)
        result = add_ecliptic_coordinates(metadata)

        # Original unchanged
        assert set(metadata.columns) == orig_cols
        assert 'ECLLON' not in metadata.columns


class TestSkySpecVACEnrichment:
    """Tests for SkySpecVAC auto-enrichment functionality."""

    def test_load_enriches_automatically(self):
        """Test that load() automatically adds all enriched columns."""
        from desisky.data import SkySpecVAC

        pytest.importorskip("speclite")
        pytest.importorskip("astropy")

        vac = SkySpecVAC(version="v1.0", download=False)
        wave, flux, meta = vac.load()

        assert wave.shape == (7781,)
        # 9176 total minus known contaminated observations (e.g., June 2021 fires)
        assert flux.shape[0] == len(meta)
        assert len(meta) < 9176

        # All 5 enrichments should be present
        assert 'SKY_MAG_V_SPEC' in meta.columns
        assert 'ECLIPSE_FRAC' in meta.columns
        assert 'SOLFLUX' in meta.columns
        assert 'GALLON' in meta.columns
        assert 'GALLAT' in meta.columns
        assert 'ECLLON' in meta.columns
        assert 'ECLLAT' in meta.columns

        # Check V-band values are reasonable
        assert meta['SKY_MAG_V_SPEC'].min() > 10
        assert meta['SKY_MAG_V_SPEC'].max() < 30

        # Check ECLIPSE_FRAC values are in [0, 1]
        assert (meta['ECLIPSE_FRAC'] >= 0).all()
        assert (meta['ECLIPSE_FRAC'] <= 1).all()

        # Check coordinate ranges
        assert (meta['GALLON'] >= 0).all() and (meta['GALLON'] <= 360).all()
        assert (meta['GALLAT'] >= -90).all() and (meta['GALLAT'] <= 90).all()
        assert (meta['ECLLON'] >= 0).all() and (meta['ECLLON'] <= 360).all()
        assert (meta['ECLLAT'] >= -90).all() and (meta['ECLLAT'] <= 90).all()

    def test_enrichment_caching(self):
        """Test that enriched data is cached after first load."""
        from desisky.data import SkySpecVAC

        pytest.importorskip("speclite")
        pytest.importorskip("astropy")

        vac = SkySpecVAC(version="v1.0", download=False)

        _, _, meta1 = vac.load()
        _, _, meta2 = vac.load()

        # Should return the same cached object
        assert meta1 is meta2

    def test_load_moon_contaminated_subset(self):
        """Test moon-contaminated subset filtering."""
        from desisky.data import SkySpecVAC

        pytest.importorskip("speclite")
        pytest.importorskip("astropy")

        vac = SkySpecVAC(version="v1.0", download=False)
        wave, flux, meta = vac.load_moon_contaminated()

        # Should be a subset
        assert len(meta) < 9176
        assert len(meta) > 0

        # All observations should meet moon criteria
        assert (meta['SUNALT'] < -20).all()
        assert (meta['MOONALT'] > 5).all()
        assert (meta['MOONFRAC'] > 0.5).all()
        assert (meta['MOONSEP'] <= 90).all()
        assert (meta['TRANSPARENCY_GFA'] > 0).all()

        # Should have enrichment columns
        assert 'SKY_MAG_V_SPEC' in meta.columns
        assert 'ECLIPSE_FRAC' in meta.columns
        assert 'SOLFLUX' in meta.columns
        assert 'GALLON' in meta.columns

        # Flux shape should match metadata
        assert flux.shape[0] == len(meta)
        assert flux.shape[1] == len(wave)

    def test_load_as_ndarray_skips_enrichment(self):
        """Test that as_dataframe=False returns raw data without enrichment."""
        from desisky.data import SkySpecVAC

        vac = SkySpecVAC(version="v1.0", download=False)
        wave, flux, meta = vac.load(as_dataframe=False)

        assert wave.shape == (7781,)
        assert flux.shape[0] == 9176
        # Should be a structured numpy array, not a DataFrame
        assert not hasattr(meta, 'columns')

    def test_load_dark_time_filtering(self):
        """Test dark time subset filtering criteria."""
        from desisky.data import SkySpecVAC

        pytest.importorskip("speclite")
        pytest.importorskip("astropy")

        vac = SkySpecVAC(version="v1.0", download=False)
        wave, flux, meta = vac.load_dark_time()

        # Should be a subset
        assert len(meta) < 9176
        assert len(meta) > 0

        # Verify filtering criteria
        assert (meta['SUNALT'] < -20).all(), "Dark time requires SUNALT < -20"
        assert (meta['MOONALT'] < -5).all(), "Dark time requires MOONALT < -5"
        assert (meta['TRANSPARENCY_GFA'] > 0).all(), "Dark time requires valid transparency"

        # Verify enrichment
        assert 'SKY_MAG_V_SPEC' in meta.columns
        assert 'ECLIPSE_FRAC' in meta.columns

        # Verify data integrity
        assert flux.shape[0] == len(meta), "Flux and metadata should match"
        assert flux.shape[1] == len(wave), "Flux should match wavelength dimension"

    def test_load_sun_contaminated_filtering(self):
        """Test sun contaminated subset filtering criteria."""
        from desisky.data import SkySpecVAC

        pytest.importorskip("speclite")
        pytest.importorskip("astropy")

        vac = SkySpecVAC(version="v1.0", download=False)
        wave, flux, meta = vac.load_sun_contaminated()

        # Should be a subset
        assert len(meta) < 9176
        assert len(meta) > 0

        # Verify filtering criteria
        assert (meta['SUNALT'] > -20).all(), "Sun contaminated requires SUNALT > -20"
        assert (meta['MOONALT'] <= -5).all(), "Sun contaminated requires MOONALT <= -5"
        assert (meta['SUNSEP'] <= 110).all(), "Sun contaminated requires SUNSEP <= 110"
        assert (meta['TRANSPARENCY_GFA'] > 0).all(), "Sun contaminated requires valid transparency"

        # Verify enrichment
        assert 'SKY_MAG_V_SPEC' in meta.columns
        assert 'ECLIPSE_FRAC' in meta.columns

        # Verify data integrity
        assert flux.shape[0] == len(meta)
        assert flux.shape[1] == len(wave)

    def test_subset_sizes_reasonable(self):
        """Test that subset sizes are reasonable and mutually exclusive where expected."""
        from desisky.data import SkySpecVAC

        pytest.importorskip("speclite")
        pytest.importorskip("astropy")

        vac = SkySpecVAC(version="v1.0", download=False)

        _, _, meta_dark = vac.load_dark_time()
        _, _, meta_sun = vac.load_sun_contaminated()
        _, _, meta_moon = vac.load_moon_contaminated()

        # All should be non-empty
        assert len(meta_dark) > 0, "Dark time subset should not be empty"
        assert len(meta_sun) > 0, "Sun contaminated subset should not be empty"
        assert len(meta_moon) > 0, "Moon contaminated subset should not be empty"

        # Dark and sun are mutually exclusive by SUNALT
        # (Dark: SUNALT < -20, Sun: SUNALT > -20)
        assert len(meta_dark) + len(meta_sun) < 9176, "Dark and sun subsets should not cover all data"

        # Dark and moon can overlap in principle (both have SUNALT < -20)
        # but moon requires MOONALT > 5, dark requires MOONALT < -5, so they're exclusive
        # Total should be reasonable
        total = len(meta_dark) + len(meta_sun) + len(meta_moon)
        assert total < 9176 * 1.5, "Total subset sizes should be reasonable"

    def test_subset_wavelength_consistency(self):
        """Test that all subsets return the same wavelength array."""
        from desisky.data import SkySpecVAC

        pytest.importorskip("speclite")
        pytest.importorskip("astropy")

        vac = SkySpecVAC(version="v1.0", download=False)

        wave_dark, _, _ = vac.load_dark_time()
        wave_sun, _, _ = vac.load_sun_contaminated()
        wave_moon, _, _ = vac.load_moon_contaminated()

        assert np.array_equal(wave_dark, wave_sun), "Wavelength arrays should be identical"
        assert np.array_equal(wave_dark, wave_moon), "Wavelength arrays should be identical"
        assert len(wave_dark) == 7781, "Wavelength array should have 7781 points"


class TestQualityFilter:
    """Tests for the known contamination quality filter."""

    def test_filter_removes_bad_nights(self):
        """Test that filter_known_contamination removes the expected nights."""
        import pandas as pd
        from desisky.data import filter_known_contamination

        meta = pd.DataFrame({
            "NIGHT": [20210601, 20210605, 20210610, 20210615, 20220101],
        })
        flux = np.ones((5, 10))

        meta_out, flux_out, n_removed = filter_known_contamination(
            meta, flux, verbose=False,
        )
        assert n_removed == 2  # June 5 and June 10
        assert len(meta_out) == 3
        assert flux_out.shape[0] == 3
        assert set(meta_out["NIGHT"]) == {20210601, 20210615, 20220101}

    def test_filter_no_bad_nights(self):
        """Test that filter passes through data with no bad nights."""
        import pandas as pd
        from desisky.data import filter_known_contamination

        meta = pd.DataFrame({"NIGHT": [20220101, 20220201, 20220301]})
        meta_out, flux_out, n_removed = filter_known_contamination(
            meta, verbose=False,
        )
        assert n_removed == 0
        assert len(meta_out) == 3
        assert flux_out is None

    def test_filter_works_with_mjd(self):
        """Test that filter can compute NIGHT from MJD column."""
        import pandas as pd
        from desisky.data import filter_known_contamination

        # MJD for June 6, 2021 is approximately 59371
        meta = pd.DataFrame({"MJD": [59371.5, 59400.5]})
        meta_out, _, n_removed = filter_known_contamination(
            meta, verbose=False,
        )
        assert n_removed == 1
        assert len(meta_out) == 1

    def test_filter_warns_without_night_or_mjd(self):
        """Test that filter warns when no NIGHT or MJD column exists."""
        import pandas as pd
        import warnings
        from desisky.data import filter_known_contamination

        meta = pd.DataFrame({"SUNALT": [-30.0, -25.0]})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            meta_out, _, n_removed = filter_known_contamination(
                meta, verbose=True,
            )
            assert n_removed == 0
            assert len(meta_out) == 2
            assert len(w) == 1
            assert "Quality filter skipped" in str(w[0].message)

    def test_exclude_known_bad_false_keeps_all(self):
        """Test that SkySpecVAC with exclude_known_bad=False keeps all data."""
        from desisky.data import SkySpecVAC

        pytest.importorskip("speclite")
        pytest.importorskip("astropy")

        vac = SkySpecVAC(version="v1.0", download=False, exclude_known_bad=False)
        _, flux, meta = vac.load()
        assert len(meta) == 9176
        assert flux.shape[0] == 9176

    def test_exclude_known_bad_true_removes_data(self):
        """Test that SkySpecVAC with exclude_known_bad=True removes data."""
        from desisky.data import SkySpecVAC

        pytest.importorskip("speclite")
        pytest.importorskip("astropy")

        vac = SkySpecVAC(version="v1.0", download=False, exclude_known_bad=True)
        _, flux, meta = vac.load()
        assert len(meta) < 9176
        assert flux.shape[0] == len(meta)
