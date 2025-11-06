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
