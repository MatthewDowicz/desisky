#!/usr/bin/env python
"""Extract per-exposure, focal-plane-averaged DESI sky spectra from a spectro
reduction (loa, jura, iron, ...) into monthly FITS files.

Script form of the sky-extraction cells in GetData.ipynb / 0_createVACdata.ipynb
(``load_sky_single`` adapted from desispec.skymag.compute_skymag), parallelised
over months.  Compared with the notebook: month tables are reset_index'ed before
NSKY is assigned (the notebook's ``table.loc[idx]`` wrote to the wrong rows),
all inputs are passed to workers explicitly (Python 3.14 defaults to
forkserver), and output files are written atomically so a resubmitted job
skips finished months.

Run inside the DESI software environment, NOT the desisky conda env (needs
desispec)::

    source /global/common/software/desi/desi_environment.sh main
    python scripts/extract_sky_spectra.py --release loa --nproc 32

or submit ``sbatch jobs/extract_sky_spectra.sh loa``.

Outputs (under --outdir, default skydata_<release>/)::

    sky-meta-<release>.csv      exposure table for --start..--stop plus
                                MOONFRAC MOONALT MOONSEP SUNALT SUNSEP OBSALT OBSAZ
    YYYY/MM/sky-YYYY-MM.fits    META: that month's rows + NSKY (petals averaged)
                                FLUX: (n_exp, 7781) float32,
                                      erg/s/cm2/A/arcsec2 (NOT scaled by 1e17)

Then stack with ``scripts/stack_sky_spectra.py --release <release>``.

Smoke test (3 exposures of one month into a scratch directory)::

    python scripts/extract_sky_spectra.py --release loa --months 2021-01 \
        --limit 3 --nproc 1 --outdir /tmp/skytest
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import pathlib
import sys
import time

import astropy.coordinates
import astropy.time
import astropy.units as u
import fitsio
import numpy as np
import pandas as pd
from astropy.table import Table
from desispec.io import findfile, read_average_flux_calibration

# DESI coadd wavelength grid (3600-9824 A, 0.8 A) and camera stitch points
WMIN, WMAX, WDELTA = 3600, 9824, 0.8
FULLWAVE = np.round(np.arange(WMIN, WMAX + WDELTA, WDELTA), 1)
WSTITCH = {"b": (WMIN, 5790), "r": (5790, 7570), "z": (7570, 9824)}
ISTITCH = {}
for _cam, (_lo, _hi) in WSTITCH.items():
    _ii = np.where((FULLWAVE >= _lo) & (FULLWAVE < _hi))[0]
    ISTITCH[_cam] = (int(_ii[0]), int(_ii[-1]) + 1)  # begin (incl), end (excl)

SPECTRO_CALIB = "/dvs_ro/cfs/cdirs/desi/spectro/desi_spectro_calib/0.6.0"
MIRROR_CLEANING_NIGHT = 20210318               # separate average calibs before/after
FIBER_AREA_ARCSEC2 = np.pi * (1.52 / 2) ** 2   # DESI-6043
DEFAULT_FFRACFLUX = 0.6                        # DESI-6043
KITT_PEAK = dict(lat="31d57m48s", lon="-111d36m0s", height=2120.0 * u.m)
FIRST_SV_NIGHT = 20201214


# ---------------------------------------------------------------- metadata
def load_exposures(release_dir: pathlib.Path, release: str, start: int, stop: int | None) -> pd.DataFrame:
    """Exposure-table rows with start <= NIGHT (<= stop) as a DataFrame."""
    table = Table.read(release_dir / f"exposures-{release}.fits")
    df = table.to_pandas()
    for col in df.columns:  # astropy leaves FITS strings as bytes
        if df[col].dtype == object:
            df[col] = df[col].map(lambda v: v.decode().strip() if isinstance(v, bytes) else v)
    sel = df["NIGHT"] >= start
    if stop is not None:
        sel &= df["NIGHT"] <= stop
    return df[sel].reset_index(drop=True)


def add_sky_geometry(df: pd.DataFrame) -> pd.DataFrame:
    """Sun/Moon/pointing geometry from tile centre + MJD (same as the notebooks)."""
    location = astropy.coordinates.EarthLocation.from_geodetic(**KITT_PEAK)
    obstime = astropy.time.Time(df["MJD"].to_numpy(dtype=float), format="mjd")
    pointing = astropy.coordinates.SkyCoord(
        ra=df["TILERA"].to_numpy(dtype=float) * u.deg,
        dec=df["TILEDEC"].to_numpy(dtype=float) * u.deg,
    )
    sun = astropy.coordinates.get_sun(obstime)
    moon = astropy.coordinates.get_body("moon", obstime)
    elongation = sun.separation(moon)
    phase_angle = np.arctan2(
        sun.distance * np.sin(elongation),
        moon.distance - sun.distance * np.cos(elongation),
    ).to(u.deg).value
    observer = astropy.coordinates.AltAz(location=location, obstime=obstime)  # no refraction
    obs_altaz = pointing.transform_to(observer)
    sun_altaz = sun.transform_to(observer)
    moon_altaz = moon.transform_to(observer)

    df["MOONFRAC"] = (1 + np.cos(np.deg2rad(phase_angle))) / 2.0
    df["MOONALT"] = moon_altaz.alt.to(u.deg).value
    df["MOONSEP"] = moon.separation(pointing).to(u.deg).value
    df["SUNALT"] = sun_altaz.alt.to(u.deg).value
    df["SUNSEP"] = sun.separation(pointing).to(u.deg).value
    df["OBSALT"] = obs_altaz.alt.to(u.deg).value
    df["OBSAZ"] = obs_altaz.az.to(u.deg).value
    return df


# ---------------------------------------------------------------- spectra
_CALIB_CACHE: dict = {}


def _average_calibration(filename: str):
    if filename not in _CALIB_CACHE:
        _CALIB_CACHE[filename] = read_average_flux_calibration(filename)
    return _CALIB_CACHE[filename]


def load_sky_single(night: int, expid: int, release_dir: str):
    """Focal-plane-averaged sky spectrum for one exposure.

    For every petal whose b, r and z sky models exist with non-zero IVAR, the
    fiber-0 sky model is converted to surface brightness with the fixed average
    flux calibration (fiber-acceptance corrected), exposure time and fiber
    area, and the three cameras are stitched at 5790 and 7570 A.

    Returns (nsky, sky): number of petals averaged, and the mean spectrum in
    erg/s/cm2/A/arcsec2 (zeros if nsky == 0).
    """
    cal_tag = "20201214" if night < MIRROR_CLEANING_NIGHT else "20210318"
    spectra = []
    for spec in range(10):
        sky = np.zeros(FULLWAVE.shape)
        ok = True
        for cam in "brz":
            fn = findfile("sky", night=night, expid=expid, camera=f"{cam}{spec}",
                          specprod_dir=release_dir)
            if not os.path.isfile(fn):
                fn += ".gz"
            if not os.path.isfile(fn):
                ok = False
                break
            with fitsio.FITS(fn) as hdus:
                ivar = hdus["IVAR"][0:1, :][0]
                if np.all(ivar == 0):
                    ok = False
                    break
                skyflux = hdus[0][0:1, :][0]
                skywave = hdus["WAVELENGTH"][:]
                exptime = hdus[0].read_header()["EXPTIME"]

            acal = _average_calibration(
                f"{SPECTRO_CALIB}/spec/fluxcalib/fluxcalibaverage-{cam}-{cal_tag}.fits")
            cal = acal.value()
            ffrac = acal.ffracflux_wave if acal.ffracflux_wave is not None else DEFAULT_FFRACFLUX
            cal = cal / ffrac                      # not in place: acal is cached
            begin, end = ISTITCH[cam]
            cal = np.interp(FULLWAVE[begin:end], acal.wave, cal)
            flux = np.interp(FULLWAVE[begin:end], skywave, skyflux)
            sky[begin:end] = flux / exptime / cal / FIBER_AREA_ARCSEC2 * 1e-17
        if ok:
            spectra.append(sky)
    nsky = len(spectra)
    if nsky == 0:
        return 0, np.zeros(FULLWAVE.shape)
    return nsky, np.mean(spectra, axis=0)


def df_to_records(df: pd.DataFrame) -> np.ndarray:
    """DataFrame -> structured array fitsio can write (strings as fixed-width bytes)."""
    cols = {}
    for col in df.columns:
        vals = df[col].to_numpy()
        if vals.dtype.kind in "OUS":
            vals = np.array(
                [v.decode() if isinstance(v, bytes) else ("" if v is None else str(v)) for v in vals],
                dtype="U",
            )
            vals = np.char.encode(vals, "ascii", "replace")
            if vals.dtype.itemsize == 0:
                vals = vals.astype("S1")
        cols[col] = vals
    rec = np.empty(len(df), dtype=[(c, v.dtype) for c, v in cols.items()])
    for c, v in cols.items():
        rec[c] = v
    return rec


def extract_month(task) -> str:
    year, month, table, outdir, release_dir, limit = task
    tag = f"{year}-{month:02d}"
    outfile = pathlib.Path(outdir) / f"{year}" / f"{month:02d}" / f"sky-{tag}.fits"
    if outfile.exists():
        return f"[{tag}] exists, skipped"
    if limit:
        table = table.iloc[:limit]
    n = len(table)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    nsky = np.zeros(n, np.int32)
    flux = np.zeros((n, FULLWAVE.size), np.float32)
    t0 = time.time()
    nfail = 0
    nights = table["NIGHT"].to_numpy()
    expids = table["EXPID"].to_numpy()
    for i in range(n):
        try:
            nsky[i], flux[i] = load_sky_single(int(nights[i]), int(expids[i]), release_dir)
        except Exception as exc:  # keep going; row stays NSKY=0 / zeros
            nfail += 1
            print(f"[{tag}] FAILED {nights[i]} {expids[i]}: {exc!r}", flush=True)
        if (i + 1) % 50 == 0 or i + 1 == n:
            print(f"[{tag}] {i + 1}/{n} exposures, {time.time() - t0:.0f}s", flush=True)

    table = table.copy()
    table["NSKY"] = nsky
    part = outfile.with_name(outfile.stem + ".part.fits")
    with fitsio.FITS(str(part), "rw", clobber=True) as fits:
        fits.write(df_to_records(table), extname="META")
        fits.write(flux, extname="FLUX")
    part.rename(outfile)
    return (f"[{tag}] done: {n} exposures, {int((nsky > 0).sum())} with sky, "
            f"{nfail} failed, {time.time() - t0:.0f}s")


# ---------------------------------------------------------------- driver
def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--release", default="loa",
                   help="spectro reduction name under --redux-root (loa, jura, iron, ...)")
    p.add_argument("--redux-root", default="/dvs_ro/cfs/cdirs/desi/spectro/redux")
    p.add_argument("--start", type=int, default=FIRST_SV_NIGHT,
                   help="first NIGHT to include (default: first night of SV)")
    p.add_argument("--stop", type=int, default=None,
                   help="last NIGHT to include (default: everything in the release)")
    p.add_argument("--outdir", default=None, help="output directory (default: skydata_<release>)")
    p.add_argument("--nproc", type=int, default=32)
    p.add_argument("--months", nargs="*", metavar="YYYY-MM", help="only these months")
    p.add_argument("--limit", type=int, default=0, help="max exposures per month (smoke tests)")
    args = p.parse_args()

    release_dir = pathlib.Path(args.redux_root) / args.release
    if not release_dir.exists():
        sys.exit(f"release directory not found: {release_dir}")
    outdir = pathlib.Path(args.outdir or f"skydata_{args.release}")
    outdir.mkdir(parents=True, exist_ok=True)

    meta_csv = outdir / f"sky-meta-{args.release}.csv"
    if meta_csv.exists():
        print(f"Using existing {meta_csv}", flush=True)
        daily = pd.read_csv(meta_csv)
    else:
        print(f"Reading {release_dir / f'exposures-{args.release}.fits'} ...", flush=True)
        daily = load_exposures(release_dir, args.release, args.start, args.stop)
        print(f"  {len(daily)} exposures, NIGHT {daily.NIGHT.min()}-{daily.NIGHT.max()}", flush=True)
        print("  computing Sun/Moon/pointing geometry ...", flush=True)
        daily = add_sky_geometry(daily)
        daily.to_csv(meta_csv, index=False)
        print(f"  wrote {meta_csv}", flush=True)

    tasks = []
    for (year, month), tbl in daily.groupby([daily["NIGHT"] // 10000, daily["NIGHT"] // 100 % 100]):
        tag = f"{int(year)}-{int(month):02d}"
        if args.months and tag not in args.months:
            continue
        tasks.append((int(year), int(month), tbl.reset_index(drop=True),
                      str(outdir), str(release_dir), args.limit))
    nexp = sum(min(len(t[2]), args.limit) if args.limit else len(t[2]) for t in tasks)
    print(f"{len(tasks)} months, {nexp} exposures, {args.nproc} process(es) -> {outdir}/", flush=True)

    t0 = time.time()
    if args.nproc <= 1:
        for task in tasks:
            print(extract_month(task), flush=True)
    else:
        with mp.get_context("fork").Pool(args.nproc) as pool:
            for msg in pool.imap_unordered(extract_month, tasks):
                print(msg, flush=True)
    print(f"All done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
