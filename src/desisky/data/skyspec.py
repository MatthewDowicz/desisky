# SPDX-FileCopyrightText: 2025-present MatthewDowicz <mjdowicz@gmail.com>
#
# SPDX-License-Identifier: MIT

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from ._core import default_root, ensure_dir, download_file


@dataclass(frozen=True)
class DataSpec:
    """Specification for a versioned dataset."""

    url: str
    filename: str
    subdir: str
    sha256: Optional[str] = None


REGISTRY: dict[str, DataSpec] = {
    "v1.0": DataSpec(
        url="https://data.desi.lbl.gov/public/dr1/vac/dr1/skyspec/v1.0/sky_spectra_vac_v1.fits",
        filename="sky_spectra_vac_v1.fits",
        subdir="dr1",
        sha256="e943bcf046965090c4566b2b132bd48aba4646f0e2c49a53eb6904e98c471a1b",
    ),
}


def load_skyspec_vac(path: Path, *, as_dataframe: bool = True):
    """
    Read the VAC FITS file from ``path`` and return (wavelength, flux, metadata).

    Parameters
    ----------
    path : Path
        Path to the FITS file.
    as_dataframe : bool, default True
        If True, return metadata as a pandas DataFrame. If False, return as
        a structured numpy array.

    Returns
    -------
    wavelength : np.ndarray
        1D array of wavelengths in Angstroms. Shape: (n_wavelengths,)
    flux : np.ndarray
        2D array of flux values. Shape: (n_spectra, n_wavelengths)
    metadata : pd.DataFrame or np.ndarray
        Metadata for each spectrum. If ``as_dataframe=True``, returns a
        DataFrame with columns like NIGHT, EXPID, TILEID, AIRMASS, etc.
        Otherwise, returns a structured numpy array.

    Raises
    ------
    ImportError
        If fitsio is not installed, or if pandas is not installed and
        ``as_dataframe=True``.
    AssertionError
        If the FITS file structure is unexpected (mismatched dimensions).
    """
    try:
        import fitsio
    except ImportError as e:
        raise ImportError(
            "fitsio is required to read the VAC (pip install fitsio)"
        ) from e

    if as_dataframe:
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for as_dataframe=True (pip install pandas) "
                "or call load_skyspec_vac(..., as_dataframe=False)."
            ) from e

    with fitsio.FITS(str(path)) as f:
        wavelength = f["WAVELENGTH"].read()
        flux = f["FLUX"].read()
        meta_raw = f["METADATA"].read()

    # Convert to native byte order (FITS uses big-endian, pandas needs native)
    import numpy as np
    wavelength = np.asarray(wavelength, dtype=wavelength.dtype.newbyteorder('='))
    flux = np.asarray(flux, dtype=flux.dtype.newbyteorder('='))

    # Convert metadata if requested
    if as_dataframe:
        import pandas as pd

        # Convert structured array to native byte order before DataFrame conversion
        meta_native = np.asarray(meta_raw, dtype=meta_raw.dtype.newbyteorder('='))
        metadata = pd.DataFrame(meta_native)
    else:
        metadata = np.asarray(meta_raw, dtype=meta_raw.dtype.newbyteorder('='))

    # Basic sanity checks
    assert flux.shape[1] == wavelength.shape[0], "flux axis != wavelength length"
    if as_dataframe:
        assert flux.shape[0] == len(metadata), "Nsamples != len(metadata)"
    else:
        assert flux.shape[0] == metadata.shape[0], "Nsamples != metadata rows"

    return wavelength, flux, metadata


class SkySpecVAC:
    """
    DESI Sky Spectra Value-Added Catalog (VAC) dataset.

    Provides a PyTorch-like interface for loading the DESI DR1 sky spectra data.
    The dataset contains observed sky spectra with metadata including observing
    conditions, moon/sun information, and photometric sky magnitudes.

    Parameters
    ----------
    root : str | Path | None
        Root data directory. If None, uses ``~/.desisky/data`` or the path
        specified by the ``DESISKY_DATA_DIR`` environment variable.
    version : str, default "v1.0"
        Dataset version key (e.g., "v1.0").
    download : bool, default False
        If True, downloads the dataset when missing.
    verify : bool, default True
        If True and a SHA-256 checksum is known, verify integrity after download.

    Attributes
    ----------
    dir : Path
        Directory where the FITS file is stored.
    path : Path
        Full path to the FITS file.
    version : str
        Dataset version.

    Examples
    --------
    >>> # Basic usage
    >>> vac = SkySpecVAC(download=True)
    >>> wave, flux, meta = vac.load()
    >>> print(wave.shape)  # (7781,)
    >>> print(flux.shape)  # (9176, 7781)
    >>>
    >>> # With enriched columns (V-band, ECLIPSE_FRAC) for v1.0 only
    >>> wave, flux, meta = vac.load(enrich=True)
    >>> print('SKY_MAG_V_SPEC' in meta.columns)  # True (v1.0 only)
    >>> print('ECLIPSE_FRAC' in meta.columns)  # True (v1.0 only)
    >>>
    >>> # Load dark-time subset (non-contaminated)
    >>> wave, flux, meta = vac.load_dark_time()
    >>> print(f"Dark-time: {len(meta)} observations")
    >>>
    >>> # Load sun-contaminated subset (twilight)
    >>> wave, flux, meta = vac.load_sun_contaminated()
    >>> print(f"Sun-contaminated: {len(meta)} observations")
    >>>
    >>> # Load moon-contaminated subset
    >>> wave, flux, meta = vac.load_moon_contaminated()
    >>> print(f"Moon-contaminated: {len(meta)} observations")

    Raises
    ------
    KeyError
        If the specified version is not in the registry.
    FileNotFoundError
        If the data file doesn't exist and ``download=False``.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        version: str = "v1.0",
        download: bool = False,
        verify: bool = True,
    ):
        if version not in REGISTRY:
            raise KeyError(f"Unknown SkySpec VAC version: {version!r}")
        spec = REGISTRY[version]

        base = Path(root) if root is not None else default_root()
        self.dir = ensure_dir(base / spec.subdir)
        self.path = self.dir / spec.filename
        self.version = version
        self._loaded: Optional[Tuple] = None  # memoized data (without enrichment)
        self._loaded_enriched: Optional[Tuple] = None  # memoized data (with enrichment)

        if not self.path.exists():
            if download:
                download_file(
                    spec.url,
                    self.path,
                    expected_sha256=(spec.sha256 if verify else None),
                    force=False,
                )
            else:
                raise FileNotFoundError(
                    f"{self.path} does not exist. Either call with download=True "
                    f"or run the CLI: `desisky-data fetch skyspec --version {version}`"
                )

    def filepath(self) -> Path:
        """Return the path to the FITS file on disk."""
        return self.path

    def load(self, *, as_dataframe: bool = True, enrich: bool = False):
        """
        Load the VAC from disk and return (wavelength, flux, metadata).

        Results are cached after the first call. Enrichment adds computed columns
        (SKY_MAG_V_SPEC, ECLIPSE_FRAC) for v1.0 data only. Future VAC versions
        may include these columns natively.

        Parameters
        ----------
        as_dataframe : bool, default True
            If True, return metadata as a pandas DataFrame. If False, return
            as a structured numpy array.
        enrich : bool, default False
            If True and version is v1.0, add computed columns:
            - SKY_MAG_V_SPEC: V-band magnitude from spectra
            - ECLIPSE_FRAC: Lunar eclipse umbral coverage fraction

        Returns
        -------
        wavelength : np.ndarray
            1D array of wavelengths in Angstroms.
        flux : np.ndarray
            2D array of flux values. Shape: (n_spectra, n_wavelengths)
        metadata : pd.DataFrame or np.ndarray
            Metadata for each spectrum. If enrich=True, includes additional columns.

        Raises
        ------
        ImportError
            If enrichment dependencies (speclite, astropy) are not installed.

        Examples
        --------
        >>> # Load without enrichment (fast)
        >>> wave, flux, meta = vac.load()
        >>>
        >>> # Load with enrichment (adds V-band and ECLIPSE_FRAC for v1.0)
        >>> wave, flux, meta = vac.load(enrich=True)
        """
        # Use appropriate cache
        if enrich:
            if self._loaded_enriched is None:
                wavelength, flux, metadata = self._load_and_enrich(as_dataframe)
                self._loaded_enriched = (wavelength, flux, metadata)
            return self._loaded_enriched
        else:
            if self._loaded is None:
                self._loaded = load_skyspec_vac(self.path, as_dataframe=as_dataframe)
            return self._loaded

    def _load_and_enrich(self, as_dataframe: bool = True):
        """Load data and add enriched columns."""
        from ._enrich import compute_vband_magnitudes, compute_eclipse_fraction

        # Load base data
        wavelength, flux, metadata = load_skyspec_vac(self.path, as_dataframe=as_dataframe)

        # Only enrich v1.0 (future versions may have these columns natively)
        if self.version != "v1.0":
            return wavelength, flux, metadata

        # Only enrich DataFrames
        if not as_dataframe:
            import warnings
            warnings.warn(
                "Enrichment (enrich=True) requires as_dataframe=True. "
                "Returning unenriched data.",
                UserWarning,
                stacklevel=3
            )
            return wavelength, flux, metadata

        # Check if columns already exist (skip if present)
        if 'SKY_MAG_V_SPEC' not in metadata.columns:
            try:
                metadata['SKY_MAG_V_SPEC'] = compute_vband_magnitudes(flux, wavelength)
            except ImportError as e:
                import warnings
                warnings.warn(
                    f"Skipping V-band enrichment: {e}",
                    UserWarning,
                    stacklevel=3
                )

        if 'ECLIPSE_FRAC' not in metadata.columns:
            try:
                metadata['ECLIPSE_FRAC'] = compute_eclipse_fraction(metadata)
            except ImportError as e:
                import warnings
                warnings.warn(
                    f"Skipping ECLIPSE_FRAC enrichment: {e}",
                    UserWarning,
                    stacklevel=3
                )

        return wavelength, flux, metadata

    def load_dark_time(self, *, enrich: bool = True):
        """
        Load subset of non-contaminated (dark time) observations.

        This method filters for observations with minimal contamination from
        moon and sun, suitable for training dark-time sky models:
        - SUNALT < -20 (Sun well below horizon)
        - MOONALT < -5 (Moon below horizon)
        - TRANSPARENCY_GFA > 0 (valid transparency measurements)

        Parameters
        ----------
        enrich : bool, default True
            If True, add computed columns (SKY_MAG_V_SPEC, ECLIPSE_FRAC).

        Returns
        -------
        wavelength : np.ndarray
            1D array of wavelengths (same for all observations).
        flux : np.ndarray
            2D array of flux values for dark-time subset.
        metadata : pd.DataFrame
            Metadata for dark-time subset with reset index.

        Examples
        --------
        >>> vac = SkySpecVAC(download=True)
        >>> wave, flux, meta = vac.load_dark_time()
        >>> print(f"Dark-time: {len(meta)} observations")
        """
        # Load full dataset with enrichment
        wavelength, flux, metadata = self.load(as_dataframe=True, enrich=enrich)

        # Apply dark time filter
        dark_mask = (
            (metadata['SUNALT'] < -20) &
            (metadata['MOONALT'] < -5) &
            (metadata['TRANSPARENCY_GFA'] > 0)
        )

        # Subset data
        flux_subset = flux[dark_mask]
        meta_subset = metadata[dark_mask].reset_index(drop=True)

        return wavelength, flux_subset, meta_subset

    def load_sun_contaminated(self, *, enrich: bool = True):
        """
        Load subset of sun-contaminated (twilight) observations.

        This method filters for observations with significant sun contamination,
        suitable for training twilight sky models:
        - SUNALT > -20 (Sun near or above horizon - twilight)
        - MOONALT <= -5 (Moon below horizon to exclude moon contamination)
        - MOONSEP <= 110 (Sun-Moon separation constraint)
        - TRANSPARENCY_GFA > 0 (valid transparency measurements)

        Parameters
        ----------
        enrich : bool, default True
            If True, add computed columns (SKY_MAG_V_SPEC, ECLIPSE_FRAC).

        Returns
        -------
        wavelength : np.ndarray
            1D array of wavelengths (same for all observations).
        flux : np.ndarray
            2D array of flux values for sun-contaminated subset.
        metadata : pd.DataFrame
            Metadata for sun-contaminated subset with reset index.

        Examples
        --------
        >>> vac = SkySpecVAC(download=True)
        >>> wave, flux, meta = vac.load_sun_contaminated()
        >>> print(f"Sun-contaminated: {len(meta)} observations")
        """
        # Load full dataset with enrichment
        wavelength, flux, metadata = self.load(as_dataframe=True, enrich=enrich)

        # Apply sun contamination filter
        sun_mask = (
            (metadata['SUNALT'] > -20) &
            (metadata['MOONALT'] <= -5) &
            (metadata['MOONSEP'] <= 110) &
            (metadata['TRANSPARENCY_GFA'] > 0)
        )

        # Subset data
        flux_subset = flux[sun_mask]
        meta_subset = metadata[sun_mask].reset_index(drop=True)

        return wavelength, flux_subset, meta_subset

    def load_moon_contaminated(self, *, enrich: bool = True):
        """
        Load subset of observations with significant moon contamination.

        This method filters for observations suitable for training the broadband
        moon sky model, which requires:
        - SUNALT < -20 (nighttime)
        - MOONALT > 5 (Moon above horizon)
        - MOONFRAC > 0.5 (Moon >50% illuminated)
        - MOONSEP <= 90 (Moon within 90 degrees)
        - TRANSPARENCY_GFA > 0 (valid transparency measurements)

        By default, enrichment is enabled to include ECLIPSE_FRAC, which is
        important for modeling moon contamination effects.

        Parameters
        ----------
        enrich : bool, default True
            If True, add computed columns (SKY_MAG_V_SPEC, ECLIPSE_FRAC).

        Returns
        -------
        wavelength : np.ndarray
            1D array of wavelengths (same for all observations).
        flux : np.ndarray
            2D array of flux values for moon-contaminated subset.
        metadata : pd.DataFrame
            Metadata for moon-contaminated subset with reset index.

        Examples
        --------
        >>> vac = SkySpecVAC(download=True)
        >>> wave, flux, meta = vac.load_moon_contaminated()
        >>> print(f"Moon-contaminated: {len(meta)} observations")
        """
        # Load full dataset with enrichment
        wavelength, flux, metadata = self.load(as_dataframe=True, enrich=enrich)

        # Apply moon contamination filter
        moon_mask = (
            (metadata['SUNALT'] < -20) &
            (metadata['MOONALT'] > 5) &
            (metadata['MOONFRAC'] > 0.5) &
            (metadata['MOONSEP'] <= 90) &
            (metadata['TRANSPARENCY_GFA'] > 0)
        )

        # Subset data
        flux_subset = flux[moon_mask]
        meta_subset = metadata[moon_mask].reset_index(drop=True)

        return wavelength, flux_subset, meta_subset
