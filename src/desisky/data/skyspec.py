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

    Examples
    --------
    >>> # PyTorch-like usage
    >>> vac = SkySpecVAC(download=True)
    >>> wave, flux, meta = vac.load()
    >>> print(wave.shape)  # (7781,)
    >>> print(flux.shape)  # (9176, 7781)
    >>> print(meta.columns)  # NIGHT, EXPID, TILEID, AIRMASS, ...

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
        self._loaded: Optional[Tuple] = None  # memoized data

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

    def load(self, *, as_dataframe: bool = True):
        """
        Load the VAC from disk and return (wavelength, flux, metadata).

        Results are cached after the first call.

        Parameters
        ----------
        as_dataframe : bool, default True
            If True, return metadata as a pandas DataFrame. If False, return
            as a structured numpy array.

        Returns
        -------
        wavelength : np.ndarray
            1D array of wavelengths in Angstroms.
        flux : np.ndarray
            2D array of flux values. Shape: (n_spectra, n_wavelengths)
        metadata : pd.DataFrame or np.ndarray
            Metadata for each spectrum.
        """
        if self._loaded is None:
            self._loaded = load_skyspec_vac(self.path, as_dataframe=as_dataframe)
        return self._loaded
