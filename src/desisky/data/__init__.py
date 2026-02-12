# SPDX-FileCopyrightText: 2025-present MatthewDowicz <mjdowicz@gmail.com>
#
# SPDX-License-Identifier: MIT

from .skyspec import SkySpecVAC, load_skyspec_vac, REGISTRY, DataSpec
from ._core import default_root, download_file, sha256sum
from ._splits import get_validation_mask
from ._enrich import (
    compute_vband_magnitudes,
    load_eclipse_catalog,
    compute_eclipse_fraction,
    load_solar_flux,
    attach_solar_flux,
    add_galactic_coordinates,
    add_ecliptic_coordinates,
)

__all__ = [
    "SkySpecVAC",
    "load_skyspec_vac",
    "REGISTRY",
    "DataSpec",
    "default_root",
    "download_file",
    "sha256sum",
    "compute_vband_magnitudes",
    "load_eclipse_catalog",
    "compute_eclipse_fraction",
    "load_solar_flux",
    "attach_solar_flux",
    "add_galactic_coordinates",
    "add_ecliptic_coordinates",
    "get_validation_mask",
]
