#!/usr/bin/env python
"""Stack the monthly files written by scripts/extract_sky_spectra.py into the
raw flux (.npy) + metadata (.csv) pair that scripts/prepare_training_data.py
expects, matching the layout of the original jura full_sky_spec.{npy,csv}.

Run in the desisky conda env (needs fitsio, speclite, desisky)::

    python scripts/stack_sky_spectra.py --release loa
    python scripts/prepare_training_data.py --flux-path full_sky_spec_loa.npy \
        --meta-path full_sky_spec_loa.csv --output-dir training_data/

What it does:
  * concatenates META/FLUX of every YYYY/MM/sky-YYYY-MM.fits under --indir
    (refuses if any *.part.fits from an unfinished month is present)
  * drops exposures with NSKY == 0 (no petal had a usable sky model; their
    spectra are all zeros and would give infinite magnitudes)
  * scales flux by 1e17 -> float32 in 1e-17 erg/s/cm2/A/arcsec2, as before
  * adds SKY_MAG_V_SPEC (Bessell V, AB mag/arcsec2) from the spectra; the
    broadband predictor needs it.  g, r, z stay as the pipeline's SKY_MAG_*_SPEC
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import fitsio
import numpy as np
import pandas as pd

from desisky.data import compute_vband_magnitudes

WAVE = np.round(np.arange(3600, 9824 + 0.8, 0.8), 1)
REQUIRED = ["NIGHT", "EXPID", "MJD", "EXPTIME", "TILERA", "TILEDEC",
            "TRANSPARENCY_GFA", "SKY_MAG_G_SPEC", "SKY_MAG_R_SPEC", "SKY_MAG_Z_SPEC",
            "MOONFRAC", "MOONALT", "MOONSEP", "SUNALT", "SUNSEP", "OBSALT", "OBSAZ", "NSKY"]


def read_month(path: pathlib.Path):
    with fitsio.FITS(str(path)) as h:
        meta = h["META"].read()
        flux = h["FLUX"].read()
    if len(meta) != len(flux):
        sys.exit(f"{path}: META rows {len(meta)} != FLUX rows {len(flux)}")
    df = pd.DataFrame(meta)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(lambda v: v.decode().strip() if isinstance(v, bytes) else v)
    return df, flux


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--release", default="loa", help="used for the default --indir/--out-prefix")
    p.add_argument("--indir", default=None, help="monthly FITS directory (default: skydata_<release>)")
    p.add_argument("--out-prefix", default=None,
                   help="output prefix for .npy/.csv (default: full_sky_spec_<release>)")
    args = p.parse_args()
    indir = pathlib.Path(args.indir or f"skydata_{args.release}")
    prefix = args.out_prefix or f"full_sky_spec_{args.release}"

    partial = sorted(indir.glob("????/??/*.part.fits"))
    if partial:
        sys.exit("Unfinished months present, rerun extract_sky_spectra.py first:\n  "
                 + "\n  ".join(str(f) for f in partial))
    files = sorted(indir.glob("????/??/sky-????-??.fits"))
    if not files:
        sys.exit(f"no sky-YYYY-MM.fits files under {indir}")

    metas, fluxes = [], []
    for f in files:
        df, flux = read_month(f)
        print(f"  {f.relative_to(indir)}: {len(df):5d} exposures, {int((df['NSKY'] > 0).sum()):5d} with sky")
        metas.append(df)
        fluxes.append(flux)
    meta = pd.concat(metas, ignore_index=True)
    flux = np.vstack(fluxes)
    if flux.shape[1] != WAVE.size:
        sys.exit(f"flux has {flux.shape[1]} wavelength bins, expected {WAVE.size}")
    print(f"Total: {len(meta)} exposures from {len(files)} months, "
          f"NIGHT {meta.NIGHT.min()}-{meta.NIGHT.max()}")

    missing = [c for c in REQUIRED if c not in meta.columns]
    if missing:
        sys.exit(f"metadata is missing columns: {missing}")

    keep = meta["NSKY"].to_numpy() > 0
    print(f"Dropping {int((~keep).sum())} exposures with NSKY == 0")
    meta = meta[keep].reset_index(drop=True)
    flux = (flux[keep] * 1e17).astype(np.float32)

    finite = np.isfinite(flux).all(axis=1)
    if not finite.all():
        print(f"Dropping {int((~finite).sum())} exposures with non-finite flux")
        meta = meta[finite].reset_index(drop=True)
        flux = flux[finite]

    print("Computing Bessell V magnitudes from the spectra ...", flush=True)
    meta["SKY_MAG_V_SPEC"] = compute_vband_magnitudes(flux, WAVE)

    npy = f"{prefix}.npy"
    csv = f"{prefix}.csv"
    np.save(npy, flux)
    meta.to_csv(csv, index=False)
    print(f"Wrote {npy}  {flux.shape} {flux.dtype}")
    print(f"Wrote {csv}  {len(meta)} rows x {len(meta.columns)} columns")
    print("Median sky mags  V/g/r/z: "
          + "  ".join(f"{meta[c].median():.2f}" for c in
                      ["SKY_MAG_V_SPEC", "SKY_MAG_G_SPEC", "SKY_MAG_R_SPEC", "SKY_MAG_Z_SPEC"]))
    print("Next: python scripts/prepare_training_data.py "
          f"--flux-path {npy} --meta-path {csv} --output-dir training_data/")


if __name__ == "__main__":
    main()
