# SPDX-FileCopyrightText: 2025-present MatthewDowicz <mjdowicz@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Utilities for identifying train/validation splits from model metadata."""

from __future__ import annotations

import numpy as np


def get_validation_mask(
    metadata,
    model_meta: dict,
) -> np.ndarray:
    """
    Return a boolean mask identifying validation samples in *metadata*.

    Each LDM checkpoint stores the ``EXPID`` values of the validation set
    used during training (under ``model_meta["training"]["val_expids"]``).
    This function matches those EXPIDs against the ``EXPID`` column of
    *metadata* to produce a per-row boolean mask.

    Parameters
    ----------
    metadata : pd.DataFrame
        Observation metadata containing an ``EXPID`` column.
    model_meta : dict
        Model metadata returned by :func:`desisky.io.load_builtin`,
        expected to contain ``training.val_expids``.

    Returns
    -------
    np.ndarray
        Boolean array of shape ``(len(metadata),)``.
        ``True`` for rows whose ``EXPID`` was in the validation set.

    Raises
    ------
    ValueError
        If *model_meta* does not contain ``training.val_expids``.
    KeyError
        If *metadata* does not have an ``EXPID`` column.

    Examples
    --------
    >>> from desisky.io import load_builtin
    >>> from desisky.data import SkySpecVAC, get_validation_mask
    >>> ldm, ldm_meta = load_builtin("ldm_dark")
    >>> vac = SkySpecVAC(download=True)
    >>> wave, flux, meta = vac.load_dark_time(enrich=True)
    >>> val_mask = get_validation_mask(meta, ldm_meta)
    Found 598 of 672 validation EXPIDs in metadata.
    >>> val_flux, val_meta = flux[val_mask], meta[val_mask]
    """
    val_expids = model_meta.get("training", {}).get("val_expids")
    if val_expids is None:
        raise ValueError(
            "Model metadata does not contain 'training.val_expids'. "
            "Re-save the model checkpoint with validation EXPIDs included."
        )

    if "EXPID" not in metadata.columns:
        raise KeyError(
            "metadata DataFrame does not contain an 'EXPID' column."
        )

    val_set = set(val_expids)
    mask = metadata["EXPID"].isin(val_set).to_numpy()

    n_found = mask.sum()
    n_total = len(val_expids)
    print(f"Found {n_found} of {n_total} validation EXPIDs in metadata.")

    return mask
