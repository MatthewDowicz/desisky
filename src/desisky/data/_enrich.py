# SPDX-FileCopyrightText: 2025-present MatthewDowicz <mjdowicz@gmail.com>
#
# SPDX-License-Identifier: MIT

"""
Data enrichment utilities for adding computed columns to DESI Sky Spectra VAC.

This module provides functions to compute V-band magnitudes from spectra and
ECLIPSE_FRAC (umbral eclipse coverage fraction) for observations.
"""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

from ._core import default_root, ensure_dir, download_file


# Eclipse catalog specification
ECLIPSE_CATALOG_URL = "https://eclipse.gsfc.nasa.gov/5MCLE/5MKLEcatalog.txt"
ECLIPSE_CATALOG_FILENAME = "5MKLEcatalog.txt"
ECLIPSE_CATALOG_SUBDIR = "eclipse"


def compute_vband_magnitudes(flux: np.ndarray, wavelength: np.ndarray) -> np.ndarray:
    """
    Compute V-band AB magnitudes from sky spectra using speclite.

    Parameters
    ----------
    flux : np.ndarray
        2D array of flux values. Shape: (n_spectra, n_wavelengths)
        Units: assumed to be in 1e-17 erg/s/cm^2/Angstrom (DESI VAC units)
    wavelength : np.ndarray
        1D array of wavelengths in Angstroms. Shape: (n_wavelengths,)

    Returns
    -------
    vband_mags : np.ndarray
        1D array of V-band AB magnitudes. Shape: (n_spectra,)

    Raises
    ------
    ImportError
        If speclite is not installed.
    """
    try:
        from speclite.filters import load_filters
    except ImportError as e:
        raise ImportError(
            "speclite is required for V-band calculation. "
            "Install with: pip install speclite"
        ) from e

    vband_filter = load_filters('bessell-V')
    vband_mags = np.array([
        vband_filter.get_ab_magnitudes(flux[i] * 1e-17, wavelength)['bessell-V'].item()
        for i in range(flux.shape[0])
    ])

    return vband_mags


def load_eclipse_catalog(
    catalog_path: str | Path | None = None,
    download: bool = True,
    root: str | Path | None = None,
) -> "pd.DataFrame":
    """
    Load the Five Millennium Canon of Lunar Eclipses catalog.

    If the catalog is not found locally and download=True, it will be
    downloaded from NASA's eclipse website.

    Parameters
    ----------
    catalog_path : str | Path | None
        Path to the eclipse catalog file. If None, uses the default location
        in the desisky data directory (~/.desisky/data/eclipse/).
    download : bool, default True
        If True, download the catalog if it doesn't exist locally.
    root : str | Path | None
        Root data directory. If None, uses default_root() from _core.

    Returns
    -------
    eclipse_df : pd.DataFrame
        DataFrame containing eclipse data with contact times (P1-P4, U1-U4)

    Raises
    ------
    ImportError
        If pandas or astropy is not installed.
    FileNotFoundError
        If the catalog file cannot be found and download=False.
    """
    try:
        import pandas as pd
        from astropy.time import Time
    except ImportError as e:
        raise ImportError(
            "pandas and astropy are required for eclipse calculations. "
            "Install with: pip install pandas astropy"
        ) from e

    # Determine catalog path
    if catalog_path is None:
        base = Path(root) if root is not None else default_root()
        catalog_dir = ensure_dir(base / ECLIPSE_CATALOG_SUBDIR)
        catalog_path = catalog_dir / ECLIPSE_CATALOG_FILENAME
    else:
        catalog_path = Path(catalog_path)

    # Download if missing
    if not catalog_path.exists():
        if download:
            download_file(ECLIPSE_CATALOG_URL, catalog_path, expected_sha256=None, force=False)
        else:
            raise FileNotFoundError(
                f"Eclipse catalog not found at {catalog_path}. Set download=True to download."
            )

    # Column specifications
    colspecs = [
        (0, 5), (6, 19), (21, 29), (37, 43), (44, 48), (51, 54), (55, 57),
        (59, 66), (68, 74), (75, 82), (84, 89), (91, 96), (99, 103), (106, 109), (111, 115)
    ]
    column_names = [
        'Cat_Num', 'Calendar_Date', 'TD_of_Greatest_Eclipse', 'Luna_Num', 'Saros_Num',
        'Ecl_Type', 'QSE', 'Gamma', 'Mag_Pen', 'Mag_Um', 'Dur_Pen', 'Dur_Par',
        'Dur_Total', 'Lat', 'Long'
    ]

    # Read catalog (DESI DR1 window: 2020-2022, lines 9705+30)
    df = pd.read_fwf(str(catalog_path), colspecs=colspecs, names=column_names,
                      skiprows=9705, nrows=30)

    # Convert numeric columns
    numeric_cols = ["Gamma", "Mag_Pen", "Mag_Um", "Dur_Pen", "Dur_Par", "Dur_Total"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # Parse dates into MJD
    df['MJD'] = df.apply(lambda row: Time.strptime(
        f"{row['Calendar_Date'].strip()} {row['TD_of_Greatest_Eclipse'].strip()}",
        "%Y %b %d %H:%M:%S", scale="tt"
    ).mjd, axis=1)

    # Add NIGHT column (YYYYMMDD)
    df["NIGHT"] = pd.to_datetime(df["Calendar_Date"], format="%Y %b %d",
                                   errors="coerce").dt.strftime("%Y%m%d").astype("int64")

    # Compute contact times
    df = pd.concat([df, _compute_contact_times(df)], axis=1)

    return df


def _compute_contact_times(df: "pd.DataFrame") -> "pd.DataFrame":
    """Compute eclipse contact times (P1-P4, U1-U4) from durations."""
    import pandas as pd

    MIN_PER_DAY = 1440.0  # 60 * 24

    # Convert durations to half-widths in days
    half_pen = df["Dur_Pen"] / (2.0 * MIN_PER_DAY)
    half_par = df["Dur_Par"] / (2.0 * MIN_PER_DAY)
    half_tot = df["Dur_Total"] / (2.0 * MIN_PER_DAY)

    # Penumbral times (always present)
    p1 = df["MJD"] - half_pen
    p4 = df["MJD"] + half_pen

    # Umbral times depend on eclipse type
    u1 = np.where(df["Ecl_Type"].str.startswith("N"), np.nan, df["MJD"] - half_par)
    u4 = np.where(df["Ecl_Type"].str.startswith("N"), np.nan, df["MJD"] + half_par)
    u2 = np.where(df["Ecl_Type"].str.startswith(("N", "P")), np.nan, df["MJD"] - half_tot)
    u3 = np.where(df["Ecl_Type"].str.startswith(("N", "P")), np.nan, df["MJD"] + half_tot)

    return pd.DataFrame({"P1": p1, "U1": u1, "U2": u2, "U3": u3, "U4": u4, "P4": p4})


def compute_eclipse_fraction(
    metadata: "pd.DataFrame",
    eclipse_df: "pd.DataFrame" | None = None,
    catalog_path: str | Path | None = None,
    download: bool = True,
) -> np.ndarray:
    """
    Compute ECLIPSE_FRAC (umbral eclipse coverage) for observations.

    Only assigns non-zero coverage when:
    1. An eclipse is occurring (obs MJD within penumbral window)
    2. Kitt Peak is in nighttime (Sun < -18 deg)
    3. Moon is above horizon (Moon alt > 5 deg)

    Parameters
    ----------
    metadata : pd.DataFrame
        Observation metadata with MJD column
    eclipse_df : pd.DataFrame | None
        Pre-loaded eclipse catalog. If None, loads from catalog_path.
    catalog_path : str | Path | None
        Path to eclipse catalog. If None, uses default location.
    download : bool, default True
        If True, download eclipse catalog if not found locally.

    Returns
    -------
    eclipse_frac : np.ndarray
        Array of eclipse fractions (0 to 1) for each observation
    """
    if eclipse_df is None:
        eclipse_df = load_eclipse_catalog(catalog_path=catalog_path, download=download)

    eclipse_frac = np.zeros(len(metadata))

    for _, ecl_row in eclipse_df.iterrows():
        # Find observations during this eclipse's penumbral window
        sel = (metadata["MJD"] >= ecl_row["P1"]) & (metadata["MJD"] <= ecl_row["P4"])
        if not sel.any():
            continue

        # Compute coverage for each observation
        new_cov = metadata.loc[sel].apply(
            lambda obs: _compute_umbral_coverage(ecl_row, obs["MJD"]), axis=1
        ).values

        eclipse_frac[sel] = np.maximum(eclipse_frac[sel], new_cov)

    return eclipse_frac


def _compute_umbral_coverage(ecl_row: "pd.Series", obs_mjd: float) -> float:
    """Compute umbral coverage fraction for a single observation."""
    import pandas as pd

    # Check observability conditions at Kitt Peak
    if not _check_observability(obs_mjd):
        return 0.0

    # Get eclipse parameters
    ecl_type = str(ecl_row["Ecl_Type"])
    mag_um = ecl_row["Mag_Um"]

    # Convert diameter fraction to area fraction
    x = np.clip(mag_um, 0, 1)
    alpha = 1.0 - 2.0 * x
    area_um = (np.arccos(alpha) - alpha * np.sqrt(1 - alpha**2)) / np.pi

    # No coverage for penumbral-only eclipses
    if ecl_type.startswith("N") or area_um <= 0:
        return 0.0

    # Linear ramp helper
    def ramp(t, t1, t2, y1, y2):
        return y1 if t <= t1 else y2 if t >= t2 else y1 + (t - t1) / (t2 - t1) * (y2 - y1)

    u1, u2, u3, u4 = ecl_row["U1"], ecl_row["U2"], ecl_row["U3"], ecl_row["U4"]

    # Partial eclipse (no U2/U3)
    if pd.isna(u2) or pd.isna(u3):
        return 0.0 if (obs_mjd < u1 or obs_mjd > u4) else ramp(obs_mjd, u1, u4, 0.0, area_um)

    # Total eclipse (U1 -> U2 -> U3 -> U4)
    if obs_mjd < u1:
        return 0.0
    elif obs_mjd < u2:
        return ramp(obs_mjd, u1, u2, 0.0, area_um)
    elif obs_mjd < u3:
        return area_um
    elif obs_mjd < u4:
        return ramp(obs_mjd, u3, u4, area_um, 0.0)
    else:
        return 0.0


def _check_observability(obs_mjd: float) -> bool:
    """Check if eclipse is observable from Kitt Peak (nighttime + Moon above horizon)."""
    try:
        from astropy.coordinates import AltAz, EarthLocation, get_body
        from astropy.time import Time
    except ImportError:
        return True  # Assume observable if astropy not available

    KITT_PEAK = EarthLocation.of_site('Kitt Peak')
    t = Time(obs_mjd, format='mjd', scale='utc')
    altaz_frame = AltAz(obstime=t, location=KITT_PEAK)

    # Check Sun < -18 deg (astronomical twilight)
    sun_alt = get_body(body='sun', time=t).transform_to(altaz_frame).alt.deg
    if sun_alt >= -18.0:
        return False

    # Check Moon > 5 deg above horizon
    moon_alt = get_body('moon', t, location=KITT_PEAK).transform_to(altaz_frame).alt.deg
    return moon_alt > 5.0


# -------------------------
# Solar flux data configuration
# -------------------------

SOLAR_FLUX_URL = "https://huggingface.co/datasets/mjdowicz/desisky/resolve/main/solarflux-2004-2025.csv"
SOLAR_FLUX_SHA256 = "6d68f5e5ca104de5f342a670e298a26e8b8f4031bab9541c9b3474275dc89ac0"


def load_solar_flux(download: bool = True, verify: bool = True) -> "pd.DataFrame":
    """
    Load daily 10.7 cm solar flux measurements (2004-2025).

    Solar flux data is downloaded on first use from HuggingFace and cached locally
    in ``~/.desisky/data/auxiliary/``. The cached file persists across sessions.

    Parameters
    ----------
    download : bool, default True
        If True and data file doesn't exist, download it automatically
    verify : bool, default True
        If True, verify SHA-256 checksum (when available)

    Returns
    -------
    pd.DataFrame
        Solar flux data with columns:
        - datetime: UTC datetime of measurement
        - fluxobsflux: Observed 10.7 cm solar flux (sfu)

    Raises
    ------
    FileNotFoundError
        If file doesn't exist and download=False
    ImportError
        If pandas is not installed

    Examples
    --------
    >>> from desisky.data import load_solar_flux
    >>> solar_df = load_solar_flux()  # Downloads on first use
    >>> print(solar_df.head())

    Notes
    -----
    Solar flux units are in solar flux units (sfu): 10^-22 W·m^-2·Hz^-1
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            "pandas is required for solar flux data. "
            "Install with: pip install pandas"
        ) from e

    # Use auxiliary data directory
    cache_dir = default_root() / "auxiliary"
    cache_dir.mkdir(parents=True, exist_ok=True)
    filepath = cache_dir / "solarflux-2004-2025.csv"

    # Download if needed
    if not filepath.exists():
        if not download:
            raise FileNotFoundError(
                f"Solar flux data not found at {filepath}.\n"
                f"Set download=True to download automatically."
            )
        print(f"Downloading solar flux data (first time only)...")
        print(f"Source: {SOLAR_FLUX_URL}")
        download_file(
            url=SOLAR_FLUX_URL,
            dest=filepath,
            expected_sha256=SOLAR_FLUX_SHA256 if verify else None,
            timeout=120
        )
        print(f"✓ Solar flux data cached at {filepath}")

    # Load CSV
    solar_df = pd.read_csv(filepath)
    solar_df['datetime'] = pd.to_datetime(solar_df['datetime'])

    return solar_df


def attach_solar_flux(
    metadata: "pd.DataFrame",
    solar_flux_df: "pd.DataFrame | None" = None,
    mjd_col: str = "MJD",
    solar_time_col: str = "datetime",
    solar_flux_col: str = "fluxobsflux",
    time_tolerance: str = "12H",
    download: bool = True,
    verbose: bool = True
) -> "pd.DataFrame":
    """
    Add or update SOLFLUX column in metadata using nearest daily solar flux measurements.

    This function performs a nearest-time merge between DESI observation metadata and
    daily 10.7 cm solar flux measurements. The solar flux is a key space weather indicator
    that affects atmospheric properties and can influence sky brightness.

    If ``solar_flux_df`` is not provided, the data will be automatically loaded (and
    downloaded if necessary) using :func:`load_solar_flux`.

    Parameters
    ----------
    metadata : pd.DataFrame
        DESI observation metadata (one row per exposure). Must contain MJD column.
    solar_flux_df : pd.DataFrame | None, default None
        Daily solar flux measurements. If None, data is loaded automatically.
        Must contain datetime and flux columns if provided.
    mjd_col : str, default "MJD"
        Column name for Modified Julian Date in metadata
    solar_time_col : str, default "datetime"
        Column name for datetime in solar_flux_df
    solar_flux_col : str, default "fluxobsflux"
        Column name for solar flux values in solar_flux_df
    time_tolerance : str, default "12H"
        Maximum time separation for valid matches (pandas Timedelta format)
    download : bool, default True
        If True and solar_flux_df is None, download data if not cached
    verbose : bool, default True
        If True, print matching statistics

    Returns
    -------
    pd.DataFrame
        Copy of metadata with updated SOLFLUX column

    Examples
    --------
    >>> from desisky.data import attach_solar_flux
    >>>
    >>> # Simple usage - auto-loads data
    >>> metadata = attach_solar_flux(metadata)
    Downloading solar flux data (first time only)...
    ✓ Solar flux data cached at ~/.desisky/data/auxiliary/solarflux-2004-2025.csv
    Matched solar-flux values for 9170 / 9176 exposures (tolerance = 12H).
    >>>
    >>> # Or provide your own DataFrame
    >>> import pandas as pd
    >>> solar_df = pd.read_csv('custom_solar_flux.csv')
    >>> solar_df['datetime'] = pd.to_datetime(solar_df['datetime'])
    >>> metadata = attach_solar_flux(metadata, solar_df, time_tolerance="6h")

    Notes
    -----
    - Uses pandas merge_asof for efficient nearest-time matching
    - Returns a copy; original metadata is not modified
    - Unmatched exposures will have NaN in SOLFLUX column
    - Solar flux units are in solar flux units (sfu): 10^-22 W·m^-2·Hz^-1
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            "pandas is required for solar flux enrichment. "
            "Install with: pip install pandas"
        ) from e

    # Auto-load solar flux data if not provided
    if solar_flux_df is None:
        solar_flux_df = load_solar_flux(download=download, verify=True)

    # Make copies to avoid mutating originals
    meta = metadata.copy()
    sol = solar_flux_df[[solar_time_col, solar_flux_col]].dropna().copy()

    # Convert MJD to datetime (UTC)
    meta["datetime"] = pd.to_datetime(
        meta[mjd_col],
        origin="1858-11-17",
        unit="D"
    )

    # Sort for merge_asof (required)
    meta.sort_values("datetime", inplace=True)
    sol.sort_values(solar_time_col, inplace=True)

    # Nearest-time merge
    merged = pd.merge_asof(
        meta,
        sol,
        left_on="datetime",
        right_on=solar_time_col,
        direction="nearest",
        tolerance=pd.Timedelta(time_tolerance)
    )

    # Statistics
    n_total = len(merged)
    n_unmatched = merged[solar_flux_col].isna().sum()
    if verbose:
        print(
            f"Matched solar-flux values for {n_total - n_unmatched} / {n_total} "
            f"exposures (tolerance = {time_tolerance})."
        )

    # Update SOLFLUX column and cleanup
    merged["SOLFLUX"] = merged[solar_flux_col]
    merged.drop(columns=[solar_flux_col, "datetime"], inplace=True)

    return merged


def add_galactic_coordinates(
    metadata: "pd.DataFrame",
    ra_col: str = "TILERA",
    dec_col: str = "TILEDEC"
) -> "pd.DataFrame":
    """
    Add Galactic coordinates (GALLON, GALLAT) to metadata.

    Converts ICRS (RA/Dec) coordinates to Galactic coordinates (l, b). Galactic
    coordinates are useful for studying the Integrated Starlight (ISL) contribution
    to sky brightness, which varies with position relative to the Galactic plane.

    Parameters
    ----------
    metadata : pd.DataFrame
        Observation metadata with RA and Dec columns
    ra_col : str, default "TILERA"
        Column name for Right Ascension (degrees)
    dec_col : str, default "TILEDEC"
        Column name for Declination (degrees)

    Returns
    -------
    pd.DataFrame
        Copy of metadata with added GALLON and GALLAT columns

    Examples
    --------
    >>> from desisky.data import add_galactic_coordinates
    >>> metadata = add_galactic_coordinates(metadata)
    >>> print(metadata[['TILERA', 'TILEDEC', 'GALLON', 'GALLAT']].head())

    Notes
    -----
    - GALLON: Galactic longitude ℓ (0-360 degrees)
    - GALLAT: Galactic latitude b (-90 to +90 degrees)
    - Uses astropy.coordinates for transformation
    - Returns a copy; original metadata is not modified
    """
    try:
        from astropy.coordinates import SkyCoord
        from astropy import units as u
    except ImportError as e:
        raise ImportError(
            "astropy is required for coordinate transformations. "
            "Install with: pip install astropy"
        ) from e

    meta = metadata.copy()

    # Convert to Galactic coordinates
    coords_icrs = SkyCoord(
        ra=meta[ra_col].values * u.deg,
        dec=meta[dec_col].values * u.deg,
        frame='icrs'
    )
    gal = coords_icrs.galactic

    meta["GALLON"] = gal.l.deg  # Galactic longitude ℓ
    meta["GALLAT"] = gal.b.deg  # Galactic latitude b

    return meta


def add_ecliptic_coordinates(
    metadata: "pd.DataFrame",
    ra_col: str = "TILERA",
    dec_col: str = "TILEDEC"
) -> "pd.DataFrame":
    """
    Add Ecliptic coordinates (ECLLON, ECLLAT) to metadata.

    Converts ICRS (RA/Dec) coordinates to geocentric ecliptic coordinates (λ, β).
    Ecliptic coordinates are useful for modeling zodiacal light, which is
    concentrated along the ecliptic plane.

    Parameters
    ----------
    metadata : pd.DataFrame
        Observation metadata with RA and Dec columns
    ra_col : str, default "TILERA"
        Column name for Right Ascension (degrees)
    dec_col : str, default "TILEDEC"
        Column name for Declination (degrees)

    Returns
    -------
    pd.DataFrame
        Copy of metadata with added ECLLON and ECLLAT columns

    Examples
    --------
    >>> from desisky.data import add_ecliptic_coordinates
    >>> metadata = add_ecliptic_coordinates(metadata)
    >>> print(metadata[['TILERA', 'TILEDEC', 'ECLLON', 'ECLLAT']].head())

    Notes
    -----
    - ECLLON: Ecliptic longitude λ (0-360 degrees)
    - ECLLAT: Ecliptic latitude β (-90 to +90 degrees)
    - Uses geocentric ecliptic frame (Earth-centered, suitable for zodiacal light)
    - Uses astropy.coordinates for transformation
    - Returns a copy; original metadata is not modified
    """
    try:
        from astropy.coordinates import SkyCoord, GeocentricTrueEcliptic
        from astropy import units as u
    except ImportError as e:
        raise ImportError(
            "astropy is required for coordinate transformations. "
            "Install with: pip install astropy"
        ) from e

    meta = metadata.copy()

    # Convert to ICRS
    coords_icrs = SkyCoord(
        ra=meta[ra_col].values * u.deg,
        dec=meta[dec_col].values * u.deg,
        frame='icrs'
    )

    # Transform to geocentric ecliptic
    ecliptic = coords_icrs.transform_to(GeocentricTrueEcliptic())

    meta["ECLLON"] = ecliptic.lon.deg  # Ecliptic longitude λ
    meta["ECLLAT"] = ecliptic.lat.deg  # Ecliptic latitude β

    return meta
