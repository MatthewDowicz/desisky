#!/usr/bin/env python
# SPDX-FileCopyrightText: 2025-present MatthewDowicz <mjdowicz@gmail.com>
#
# SPDX-License-Identifier: MIT

"""
Broadband Sky Brightness Inference
====================================

Run the broadband MLP on moon-contaminated data and output predicted
V/g/r/z magnitudes alongside observed values.

Usage:
    desisky-infer-broadband
    desisky-infer-broadband --model-path my_model.eqx --output preds.csv
    desisky-infer-broadband --data-path my_data.fits --output-format npz
"""

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from desisky.io import load_or_builtin

INPUT_FEATURES = [
    "MOONSEP", "MOONFRAC", "MOONALT",
    "OBSALT", "TRANSPARENCY_GFA", "ECLIPSE_FRAC",
]
BAND_NAMES = ["V", "g", "r", "z"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run broadband MLP inference on moon-contaminated data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to custom checkpoint (default: builtin)")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to FITS or CSV with required columns. "
                             "Default: download DESI moon-contaminated subset")
    parser.add_argument("--output", type=str, default="broadband_predictions.csv")
    parser.add_argument("--output-format", choices=["csv", "npz"], default="csv")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Limit number of spectra to process")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Broadband Sky Brightness Inference (Moon-Contaminated)")
    print("=" * 60)

    # [1/3] Load Model
    print("\n[1/3] Loading broadband model...")
    model, meta = load_or_builtin("broadband", path=args.model_path)
    model_source = args.model_path or "builtin"
    print(f"  Model: {model_source}")
    print(f"  Architecture: {meta.get('arch', {})}")

    # [2/3] Load Data
    print("\n[2/3] Loading moon-contaminated data...")
    if args.data_path:
        p = Path(args.data_path)
        ext = p.suffix.lower()
        if ext == ".fits":
            from astropy.table import Table
            metadata = Table.read(p).to_pandas()
        elif ext == ".csv":
            metadata = pd.read_csv(p)
        else:
            sys.exit(f"Unsupported format '{ext}'. Use .fits or .csv")

        required = set(INPUT_FEATURES)
        missing = required - set(metadata.columns)
        if missing:
            sys.exit(f"Missing required columns: {sorted(missing)}")
    else:
        from desisky.data import SkySpecVAC
        vac = SkySpecVAC(version="v1.0", download=True)
        _, _, metadata = vac.load_moon_contaminated(enrich=True)

    if args.n_samples is not None:
        metadata = metadata.iloc[:args.n_samples]
    print(f"  {len(metadata):,} observations")

    # [3/3] Run Inference
    print("\n[3/3] Running inference...")
    inputs = metadata[INPUT_FEATURES].to_numpy().astype(np.float32)
    predictions = np.array(jax.vmap(model)(jnp.asarray(inputs)))
    print(f"  Predictions shape: {predictions.shape}")

    # Build output
    out_path = Path(args.output)
    if args.output_format == "csv":
        df = pd.DataFrame()
        if "EXPID" in metadata.columns:
            df["EXPID"] = metadata["EXPID"].values
        for i, band in enumerate(BAND_NAMES):
            df[f"{band}_pred"] = predictions[:, i]
        # Include observed values if available (e.g., when running on the
        # DESI VAC which has measured magnitudes — lets users compare pred vs obs)
        for band in BAND_NAMES:
            if band in metadata.columns:
                df[f"{band}_obs"] = metadata[band].values
        df.to_csv(out_path, index=False)
        print(f"\n  Saved CSV: {out_path} ({len(df)} rows)")
    else:
        save_dict = {"predictions": predictions}
        if "EXPID" in metadata.columns:
            save_dict["expids"] = metadata["EXPID"].values
        for band in BAND_NAMES:
            if band in metadata.columns:
                save_dict["observed"] = metadata[BAND_NAMES].to_numpy()
                break
        np.savez(out_path, **save_dict)
        print(f"\n  Saved npz: {out_path}")


if __name__ == "__main__":
    main()
