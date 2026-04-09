#!/usr/bin/env python
"""Prepare training data by applying quality filters and enrichment.

Takes raw flux (.npy) and metadata (.csv) files and produces clean,
aligned training files with known contaminated observations removed.

Usage:
    python scripts/prepare_training_data.py \
        --flux-path /path/to/updated_flux.npy \
        --meta-path /path/to/updated_sky_meta.csv \
        --output-dir ./training_data/

Outputs:
    training_data/flux_clean.npz        — filtered flux with key "flux"
    training_data/flux_clean.npy        — same as .npz but as raw .npy
    training_data/metadata_clean.csv    — filtered + enriched metadata
    training_data/metadata_moon.csv     — moon subset for broadband training
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Prepare training data with quality filtering and enrichment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--flux-path", type=str, required=True,
                        help="Path to raw flux array (.npy or .npz with 'flux' key)")
    parser.add_argument("--meta-path", type=str, required=True,
                        help="Path to raw metadata CSV")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Directory for output files")
    parser.add_argument("--no-quality-filter", action="store_true",
                        help="Skip removal of known contaminated observations")
    parser.add_argument("--no-enrich", action="store_true",
                        help="Skip metadata enrichment (SOLFLUX, coordinates, etc.)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load raw data ----
    print("Loading raw data...")
    flux_path = Path(args.flux_path)
    if flux_path.suffix == ".npz":
        flux = np.load(flux_path)["flux"]
    else:
        flux = np.load(flux_path)
    metadata = pd.read_csv(args.meta_path)

    print(f"  Flux:     {flux.shape[0]:,} spectra x {flux.shape[1]:,} wavelength bins")
    print(f"  Metadata: {len(metadata):,} rows x {len(metadata.columns)} columns")

    if len(metadata) != len(flux):
        sys.exit(f"Error: metadata rows ({len(metadata)}) != flux rows ({flux.shape[0]})")

    # ---- Quality filter ----
    if not args.no_quality_filter:
        from desisky.data import filter_known_contamination
        metadata, flux, n_removed = filter_known_contamination(
            metadata, flux, verbose=True,
        )
        if n_removed == 0:
            print("  No known contaminated observations found.")
    else:
        print("  Quality filter: SKIPPED (--no-quality-filter)")

    # ---- Filter invalid TRANSPARENCY_GFA ----
    if "TRANSPARENCY_GFA" in metadata.columns:
        bad_transp = (metadata["TRANSPARENCY_GFA"].isna()) | (metadata["TRANSPARENCY_GFA"] <= 0)
        n_bad = bad_transp.sum()
        if n_bad > 0:
            print(f"  Removing {n_bad} rows with NaN/zero TRANSPARENCY_GFA")
            flux = flux[~bad_transp.values]
            metadata = metadata[~bad_transp].reset_index(drop=True)

    # ---- Enrich metadata ----
    if not args.no_enrich:
        print("Enriching metadata...")
        from desisky.data import (
            attach_solar_flux,
            add_galactic_coordinates,
            add_ecliptic_coordinates,
        )
        try:
            if "SOLFLUX" not in metadata.columns:
                metadata = attach_solar_flux(metadata, verbose=False)
                print("  Added SOLFLUX")
        except Exception as e:
            print(f"  Warning: SOLFLUX enrichment failed: {e}")

        try:
            if "GALLON" not in metadata.columns:
                metadata = add_galactic_coordinates(metadata)
                print("  Added GALLON, GALLAT")
        except Exception as e:
            print(f"  Warning: galactic coord enrichment failed: {e}")

        try:
            if "ECLLON" not in metadata.columns:
                metadata = add_ecliptic_coordinates(metadata)
                print("  Added ECLLON, ECLLAT")
        except Exception as e:
            print(f"  Warning: ecliptic coord enrichment failed: {e}")

        # Add broadband column aliases if SKY_MAG columns exist
        mag_aliases = {"V": "SKY_MAG_V_SPEC", "g": "SKY_MAG_G_SPEC",
                       "r": "SKY_MAG_R_SPEC", "z": "SKY_MAG_Z_SPEC"}
        for alias, col in mag_aliases.items():
            if col in metadata.columns and alias not in metadata.columns:
                metadata[alias] = metadata[col]
                print(f"  Added alias: {alias} -> {col}")
    else:
        print("  Enrichment: SKIPPED (--no-enrich)")

    # ---- Save outputs ----
    print(f"\nSaving to {out_dir}/...")

    npz_path = out_dir / "flux_clean.npz"
    np.savez(npz_path, flux=flux)
    print(f"  {npz_path} ({flux.shape[0]:,} x {flux.shape[1]:,})")

    npy_path = out_dir / "flux_clean.npy"
    np.save(npy_path, flux)
    print(f"  {npy_path}")

    csv_path = out_dir / "metadata_clean.csv"
    metadata.to_csv(csv_path, index=False)
    print(f"  {csv_path} ({len(metadata):,} rows)")

    # ---- Moon subset for broadband ----
    moon_mask = (
        (metadata["SUNALT"] < -20) &
        (metadata["MOONALT"] > 5) &
        (metadata["MOONFRAC"] > 0.5) &
        (metadata["MOONSEP"] <= 90) &
        (metadata["TRANSPARENCY_GFA"] > 0)
    )
    meta_moon = metadata[moon_mask].reset_index(drop=True)
    moon_path = out_dir / "metadata_moon.csv"
    meta_moon.to_csv(moon_path, index=False)
    print(f"  {moon_path} ({len(meta_moon):,} moon-contaminated rows)")

    # ---- Summary ----
    print(f"\nDone. Final dataset: {len(metadata):,} spectra")
    print(f"  Moon subset: {len(meta_moon):,} spectra")

    # Quick sanity check on conditioning columns for LDM
    for col in ["SUNALT", "MOONALT", "MOONFRAC", "MOONSEP", "SUNSEP",
                "TRANSPARENCY_GFA", "OBSALT", "SOLFLUX", "ECLLON",
                "ECLLAT", "GALLON", "GALLAT"]:
        if col in metadata.columns:
            n_nan = metadata[col].isna().sum()
            if n_nan > 0:
                print(f"  Warning: {col} has {n_nan} NaN values")


if __name__ == "__main__":
    main()
