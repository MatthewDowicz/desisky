# SPDX-FileCopyrightText: 2025-present MatthewDowicz <mjdowicz@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Known data quality issues and filtering utilities for DESI sky spectra."""

from __future__ import annotations

import numpy as np

# Known contaminated observation periods.
# Format: (start_night_YYYYMMDD, end_night_YYYYMMDD, description)
KNOWN_BAD_PERIODS = [
    (
        20210604,
        20210612,
        "Arizona wildfire aerosol contamination (Telegraph & Mescal fires) — "
        "anomalous blue excess and spectral shape distortion across all sky categories",
    ),
]


def filter_known_contamination(metadata, flux=None, verbose=True):
    """Filter out observations from known contaminated periods.

    Removes spectra taken during time windows with documented environmental
    contamination (e.g., wildfire smoke) that distorts spectral shapes in ways
    not captured by standard quality cuts.

    Parameters
    ----------
    metadata : pd.DataFrame
        Observation metadata. Must contain a ``NIGHT`` column (YYYYMMDD int)
        or an ``MJD`` column from which NIGHT can be computed.
    flux : np.ndarray, optional
        Flux array with the same number of rows as metadata. If provided,
        rows are removed in parallel with metadata.
    verbose : bool, default True
        Print a summary of removed observations.

    Returns
    -------
    metadata : pd.DataFrame
        Filtered metadata with reset index.
    flux : np.ndarray or None
        Filtered flux array, or None if flux was not provided.
    n_removed : int
        Number of observations removed.

    Examples
    --------
    >>> import pandas as pd
    >>> from desisky.data import filter_known_contamination
    >>> meta = pd.read_csv("my_metadata.csv")
    >>> flux = np.load("my_flux.npy")
    >>> meta, flux, n = filter_known_contamination(meta, flux)
    """
    import warnings

    if len(KNOWN_BAD_PERIODS) == 0:
        return metadata, flux, 0

    night = _resolve_night_column(metadata)
    if night is None:
        if verbose:
            warnings.warn(
                "Quality filter skipped: metadata has no 'NIGHT' or 'MJD' column. "
                "Known contaminated observations may be present.",
                UserWarning,
                stacklevel=2,
            )
        return metadata, flux, 0

    bad_mask = np.zeros(len(metadata), dtype=bool)
    details = []
    for start, end, reason in KNOWN_BAD_PERIODS:
        period_mask = (night >= start) & (night <= end)
        n = int(period_mask.sum())
        if n > 0:
            bad_mask |= period_mask
            details.append(f"  {n} spectra from NIGHT {start}-{end}: {reason}")

    n_removed = int(bad_mask.sum())
    if n_removed > 0 and verbose:
        print(
            f"Quality filter: removed {n_removed} spectra from "
            f"known contaminated periods:"
        )
        for d in details:
            print(d)

    good = ~bad_mask
    metadata_out = metadata[good].reset_index(drop=True)
    flux_out = flux[good] if flux is not None else None
    return metadata_out, flux_out, n_removed


def _resolve_night_column(metadata):
    """Get NIGHT as an int array from metadata, computing from MJD if needed."""
    if "NIGHT" in metadata.columns:
        return metadata["NIGHT"].astype(int).values

    if "MJD" in metadata.columns:
        import pandas as pd

        dates = pd.to_datetime(
            metadata["MJD"].values + 2400000.5, origin="julian", unit="D"
        )
        return (dates.year * 10000 + dates.month * 100 + dates.day).values

    return None
