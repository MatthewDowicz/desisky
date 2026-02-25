# SPDX-FileCopyrightText: 2025-present MatthewDowicz <mjdowicz@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Spectral feature extraction for DESI sky spectra.

Provides functions for measuring airglow emission line intensities and
computing broadband AB magnitudes from sky spectra.

Typical usage::

    from desisky.data import (
        measure_airglow_intensities,
        compute_broadband_mags,
        AIRGLOW_CDF_NAMES,
    )

    # Airglow line intensities
    ag_df = measure_airglow_intensities(wavelength, flux)

    # Broadband magnitudes (V, g, r, z)
    mags = compute_broadband_mags(wavelength, flux)
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


# ============================================================================
# Airglow Emission Line Definitions
# ============================================================================

#: Default airglow line band definitions.  Each entry specifies the
#: integration band and two flanking continuum windows used for
#: linear continuum subtraction (Noll et al. 2012).
LINE_BANDS: dict[str, dict] = {
    "OI 5577": {"band": (5574, 5585), "cont1": (5555, 5560), "cont2": (5630, 5635)},
    "Na I D":  {"band": (5885, 5902), "cont1": (5880, 5882), "cont2": (5962, 5968)},
    "OI 6300": {"band": (6297, 6305), "cont1": (6282, 6284), "cont2": (6315, 6317)},
    "OI 6364": {"band": (6362, 6370), "cont1": (6353, 6355), "cont2": (6373, 6377)},
    "NI 5200": {"band": (5197, 5206), "cont1": (5195, 5196), "cont2": (5210, 5211)},
    "OH(6-1)": {"band": (6435, 6680), "cont1": (6430, 6432), "cont2": (6694, 6696)},
    "OH(7-2)": {"band": (6810, 7060), "cont1": (6750, 6775), "cont2": (6775, 6800)},
    "OH(8-3)": {"band": (7200, 7450), "cont1": (7197, 7198), "cont2": (7452, 7453)},
    "OH(6-2)": {"band": (8250, 8570), "cont1": (8240, 8243), "cont2": (8583, 8586)},
    "O2(0-1)": {"band": (8605, 8716), "cont1": (8583, 8587), "cont2": (8720, 8725)},
}

#: Default airglow features for CDF comparison plots.  Includes
#: composite features (OH = sum of all OH bands, OI doublet = OI 6300 + OI 6364).
AIRGLOW_CDF_NAMES: list[str] = [
    "OI 5577", "Na I D", "OI doublet", "OH", "O2(0-1)", "NI 5200",
]

#: Default broadband filter names for :func:`compute_broadband_mags`.
BROADBAND_NAMES: list[str] = ["V", "g", "r", "z"]

#: Flux scale factor to convert DESI VAC flux units to physical units
#: (erg s^-1 cm^-2 A^-1).
FLUX_SCALE: float = 1e-17


# ============================================================================
# Airglow Measurement
# ============================================================================


def _integrate_band_with_linear_continuum(
    wl: np.ndarray,
    flux: np.ndarray,
    band: tuple[float, float],
    cont1: tuple[float, float],
    cont2: tuple[float, float],
) -> float:
    """Integrate flux in a band after subtracting a linear continuum.

    Parameters
    ----------
    wl : np.ndarray
        Wavelength grid, shape ``(n_wavelengths,)``.
    flux : np.ndarray
        Flux array for a single spectrum, shape ``(n_wavelengths,)``.
    band : tuple[float, float]
        Wavelength range ``(lo, hi)`` for the emission band.
    cont1, cont2 : tuple[float, float]
        Wavelength ranges for the two flanking continuum windows.

    Returns
    -------
    float
        Continuum-subtracted integrated flux.  ``NaN`` if the continuum
        windows contain fewer than 2 pixels.
    """
    band_mask = (wl >= band[0]) & (wl <= band[1])
    c1_mask = (wl >= cont1[0]) & (wl <= cont1[1])
    c2_mask = (wl >= cont2[0]) & (wl <= cont2[1])

    cont_wl = np.concatenate([wl[c1_mask], wl[c2_mask]])
    cont_fl = np.concatenate([flux[c1_mask], flux[c2_mask]])

    if len(cont_wl) < 2:
        return np.nan

    a, b = np.polyfit(cont_wl, cont_fl, deg=1)
    continuum = a * wl[band_mask] + b
    return float(np.trapezoid(flux[band_mask] - continuum, wl[band_mask]))


def measure_airglow_intensities(
    wavelength: np.ndarray,
    spectra: np.ndarray,
    line_bands: Optional[dict[str, dict]] = None,
) -> pd.DataFrame:
    """Measure airglow emission line intensities from sky spectra.

    For each spectrum, integrates flux within each emission band after
    subtracting a linear continuum fit from two flanking windows
    (Noll et al. 2012).  Composite features are automatically added:

    - ``"OH"`` — sum of all ``OH(*)`` bands
    - ``"OI doublet"`` — ``OI 6300`` + ``OI 6364``

    Parameters
    ----------
    wavelength : np.ndarray
        Wavelength grid, shape ``(n_wavelengths,)``.
    spectra : np.ndarray
        Flux array, shape ``(N, n_wavelengths)``.
    line_bands : dict | None
        Line band definitions.  Defaults to :data:`LINE_BANDS`.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per spectrum and one column per emission
        line (plus composite columns ``"OH"`` and ``"OI doublet"``).
    """
    if line_bands is None:
        line_bands = LINE_BANDS

    rows = []
    for spec in spectra:
        row = {}
        for name, cfg in line_bands.items():
            row[name] = _integrate_band_with_linear_continuum(
                wavelength, spec, cfg["band"], cfg["cont1"], cfg["cont2"],
            )
        rows.append(row)

    df = pd.DataFrame(rows)

    # Composite: total OH
    oh_cols = [c for c in df.columns if c.startswith("OH(")]
    if oh_cols:
        df["OH"] = df[oh_cols].sum(axis=1)

    # Composite: OI doublet
    if {"OI 6300", "OI 6364"}.issubset(df.columns):
        df["OI doublet"] = df["OI 6300"] + df["OI 6364"]

    return df


# ============================================================================
# Broadband Magnitudes
# ============================================================================


def compute_broadband_mags(
    wavelength: np.ndarray,
    spectra: np.ndarray,
    flux_scale: float = FLUX_SCALE,
    filter_names: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Compute broadband AB magnitudes for sky spectra.

    Uses ``speclite`` to convolve spectra with standard filter response
    curves.  Spectra are zero-padded to cover filter wavelength ranges
    that extend beyond the DESI spectral coverage.

    Parameters
    ----------
    wavelength : np.ndarray
        Wavelength grid in Angstrom, shape ``(n_wavelengths,)``.
    spectra : np.ndarray
        Flux array in DESI VAC units, shape ``(N, n_wavelengths)``.
    flux_scale : float
        Multiplicative factor to convert flux to physical units
        (erg s^-1 cm^-2 A^-1).  Default is ``1e-17``.
    filter_names : Sequence[str] | None
        Filter identifiers for ``speclite.filters.load_filter``.
        Defaults to ``["bessell-V", "decam2014-g", "decam2014-r",
        "decam2014-z"]``.

    Returns
    -------
    np.ndarray
        AB magnitudes, shape ``(N, n_filters)``.  Column order matches
        ``filter_names`` (default: V, g, r, z).

    Raises
    ------
    ImportError
        If ``speclite`` or ``astropy`` is not installed.
    """
    try:
        from speclite.filters import load_filter
        import astropy.units as u
    except ImportError:
        raise ImportError(
            "speclite and astropy are required for broadband magnitudes. "
            "Install with: pip install desisky[data]"
        ) from None

    if filter_names is None:
        filter_names = ["bessell-V", "decam2014-g", "decam2014-r", "decam2014-z"]

    filters = [load_filter(name) for name in filter_names]
    n_filters = len(filters)

    mags = np.empty((len(spectra), n_filters))
    for i, spec in enumerate(spectra):
        f_pad = spec.copy()
        w_pad = wavelength.copy()
        for filt in filters:
            f_pad, w_pad = filt.pad_spectrum(f_pad, w_pad, method="zero")
        phys_flux = f_pad * flux_scale * u.erg / (u.cm**2 * u.s * u.angstrom)
        for j, filt in enumerate(filters):
            mags[i, j] = filt.get_ab_magnitude(phys_flux, w_pad * u.angstrom)

    return mags
