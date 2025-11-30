# SPDX-FileCopyrightText: 2025-present MatthewDowicz <mjdowicz@gmail.com>
#
# SPDX-License-Identifier: MIT

"""
Tests for VAC enrichment functionality (_enrich.py module and SkySpecVAC enrichment).
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path


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
    """Tests for SkySpecVAC enrichment functionality."""

    def test_load_without_enrichment(self):
        """Test basic load without enrichment."""
        from desisky.data import SkySpecVAC

        vac = SkySpecVAC(version="v1.0", download=False)
        wave, flux, meta = vac.load(enrich=False)

        assert wave.shape == (7781,)
        assert flux.shape[0] == 9176
        assert len(meta) == 9176
        assert 'SKY_MAG_V_SPEC' not in meta.columns
        assert 'ECLIPSE_FRAC' not in meta.columns

    def test_load_with_enrichment(self):
        """Test load with enrichment adds V-band and ECLIPSE_FRAC."""
        from desisky.data import SkySpecVAC

        pytest.importorskip("speclite")
        pytest.importorskip("astropy")

        vac = SkySpecVAC(version="v1.0", download=False)
        wave, flux, meta = vac.load(enrich=True)

        assert wave.shape == (7781,)
        assert flux.shape[0] == 9176
        assert len(meta) == 9176
        assert 'SKY_MAG_V_SPEC' in meta.columns
        assert 'ECLIPSE_FRAC' in meta.columns

        # Check V-band values are reasonable
        assert meta['SKY_MAG_V_SPEC'].min() > 10
        assert meta['SKY_MAG_V_SPEC'].max() < 30

        # Check ECLIPSE_FRAC values are in [0, 1]
        assert (meta['ECLIPSE_FRAC'] >= 0).all()
        assert (meta['ECLIPSE_FRAC'] <= 1).all()

    def test_enrichment_caching(self):
        """Test that enriched and non-enriched data are cached separately."""
        from desisky.data import SkySpecVAC

        pytest.importorskip("speclite")
        pytest.importorskip("astropy")

        vac = SkySpecVAC(version="v1.0", download=False)

        # Load without enrichment
        _, _, meta1 = vac.load(enrich=False)
        assert 'SKY_MAG_V_SPEC' not in meta1.columns

        # Load with enrichment
        _, _, meta2 = vac.load(enrich=True)
        assert 'SKY_MAG_V_SPEC' in meta2.columns

        # Load without enrichment again (should use cache)
        _, _, meta3 = vac.load(enrich=False)
        assert 'SKY_MAG_V_SPEC' not in meta3.columns

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
        assert (meta['TRANSPARENCY_GFA'] > 0).all()  # New: filter invalid transparency

        # Should have enrichment columns by default
        assert 'SKY_MAG_V_SPEC' in meta.columns
        assert 'ECLIPSE_FRAC' in meta.columns

        # Flux shape should match metadata
        assert flux.shape[0] == len(meta)
        assert flux.shape[1] == len(wave)

    def test_moon_subset_without_enrichment(self):
        """Test moon subset can be loaded without enrichment."""
        from desisky.data import SkySpecVAC

        vac = SkySpecVAC(version="v1.0", download=False)
        wave, flux, meta = vac.load_moon_contaminated(enrich=False)

        # Should still be moon-filtered
        assert len(meta) < 9176
        assert (meta['SUNALT'] < -20).all()

        # But no enrichment columns
        assert 'SKY_MAG_V_SPEC' not in meta.columns
        assert 'ECLIPSE_FRAC' not in meta.columns

    def test_enrichment_only_for_v10(self):
        """Test that enrichment only applies to v1.0."""
        from desisky.data import SkySpecVAC

        pytest.importorskip("speclite")
        pytest.importorskip("astropy")

        # For future versions, enrichment should be skipped
        # For now, only v1.0 exists, so this test documents the behavior
        vac = SkySpecVAC(version="v1.0", download=False)

        # This should work
        wave, flux, meta = vac.load(enrich=True)
        assert 'SKY_MAG_V_SPEC' in meta.columns

    def test_enrichment_warns_without_dataframe(self):
        """Test that enrichment warns if as_dataframe=False."""
        from desisky.data import SkySpecVAC
        import warnings

        vac = SkySpecVAC(version="v1.0", download=False)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            wave, flux, meta = vac.load(as_dataframe=False, enrich=True)

            # Should warn about enrichment requiring DataFrame
            assert len(w) > 0
            assert "as_dataframe=True" in str(w[0].message)
