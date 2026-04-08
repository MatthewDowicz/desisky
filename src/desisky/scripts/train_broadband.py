#!/usr/bin/env python
# SPDX-FileCopyrightText: 2025-present MatthewDowicz <mjdowicz@gmail.com>
#
# SPDX-License-Identifier: MIT

"""
Train a Broadband Sky Brightness MLP
=====================================

Predicts V/g/r/z photometric magnitudes from moon-contaminated observing
conditions. This is an updated version of the Krisciunas & Schaefer model.

The model is trained exclusively on DESI moon-contaminated sky spectra
(SUNALT < -20, MOONALT > 5, MOONFRAC > 0.5, MOONSEP <= 90).

Usage:
    desisky-train-broadband --epochs 500
    desisky-train-broadband --epochs 1000 --wandb
    desisky-train-broadband --data-path my_moon_data.fits --epochs 200
"""

import argparse
import sys
from pathlib import Path

import jax
import jax.random as jr
import numpy as np
import torch
from torch.utils.data import random_split

from desisky.models.broadband import make_broadbandMLP
from desisky.training import (
    SkyBrightnessDataset,
    NumpyLoader,
    BroadbandTrainer,
    TrainingConfig,
)

# Feature ordering matters for the MLP — the script extracts columns by name
# from FITS/CSV using metadata[INPUT_FEATURES], which guarantees correct ordering
# regardless of the column order in the user's file.
INPUT_FEATURES = [
    "MOONSEP", "OBSALT", "MOONALT",
    "MOONFRAC", "TRANSPARENCY_GFA", "ECLIPSE_FRAC",
]
IN_SIZE = 6
OUT_SIZE = 4


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a broadband sky brightness MLP on moon-contaminated data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Training
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--loss", choices=["l2", "huber"], default="huber")
    parser.add_argument("--huber-delta", type=float, default=0.25)
    parser.add_argument("--val-split", type=float, default=0.3,
                        help="Fraction of data used for validation")
    parser.add_argument("--validate-every", type=int, default=1,
                        help="Validate every N epochs")
    # Architecture
    parser.add_argument("--width", type=int, default=128, help="Hidden layer width")
    parser.add_argument("--depth", type=int, default=5, help="Number of hidden layers")
    # Data
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to user FITS or CSV file with required columns: "
                             "MOONSEP, OBSALT, MOONALT, MOONFRAC, TRANSPARENCY_GFA, "
                             "ECLIPSE_FRAC, V, g, r, z. Auto-detected by extension. "
                             "Default: download DESI moon-contaminated subset")
    # Checkpointing
    parser.add_argument("--run-name", type=str, default="broadband_moon")
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save checkpoints (useful for sweeps/testing)")
    # wandb
    parser.add_argument("--wandb", action="store_true", help="Enable wandb tracking")
    parser.add_argument("--wandb-project", type=str, default="desisky-broadband")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, default="",
                        help="Comma-separated wandb tags")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--viz-every", type=int, default=50)
    # Quality filtering
    parser.add_argument("--no-quality-filter", action="store_true",
                        help="Disable automatic removal of known contaminated "
                             "observations (e.g., June 2021 wildfire event)")
    # Other
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-every", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("DESI Sky Broadband MLP Training (Moon-Contaminated)")
    print("=" * 80)

    # [1/5] Load Data
    print("\n[1/5] Loading moon-contaminated sky spectra...")
    if args.data_path:
        p = Path(args.data_path)
        ext = p.suffix.lower()
        if ext == ".fits":
            from astropy.table import Table
            metadata = Table.read(p).to_pandas()
        elif ext == ".csv":
            import pandas as pd
            metadata = pd.read_csv(p)
        else:
            sys.exit(f"Unsupported file format '{ext}'. Use .fits or .csv")

        # Validate required columns (ordering doesn't matter —
        # we extract by name via metadata[INPUT_FEATURES] below)
        required = set(INPUT_FEATURES) | {"V", "g", "r", "z"}
        missing = required - set(metadata.columns)
        if missing:
            sys.exit(f"Missing required columns: {sorted(missing)}")

        print(f"  Loaded user data from {args.data_path} ({len(metadata):,} rows)")

        # Remove known contaminated observations
        if not args.no_quality_filter:
            from desisky.data import filter_known_contamination
            metadata, _, _ = filter_known_contamination(
                metadata, verbose=True,
            )

        # User data has pre-computed magnitudes, so we bypass SkyBrightnessDataset
        # (which requires raw spectra + the SKY_MAG_*_SPEC column naming convention).
        # Build inputs/targets arrays directly from the named columns.
        inputs = metadata[INPUT_FEATURES].to_numpy().astype(np.float32)
        targets = metadata[["V", "g", "r", "z"]].to_numpy().astype(np.float32)
        # BroadbandTrainer expects (inputs, targets, spectrum) per batch.
        # User data has no spectra, so we pass a dummy placeholder.
        dummy_spectra = torch.zeros(len(inputs), 1)
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(inputs),
            torch.from_numpy(targets),
            dummy_spectra,
        )
    else:
        from desisky.data import SkySpecVAC
        vac = SkySpecVAC(version="v1.0", download=True,
                         exclude_known_bad=not args.no_quality_filter)
        wavelength, flux, metadata = vac.load_moon_contaminated()
        print(f"  Loaded {len(metadata):,} moon-contaminated observations")

        dataset = SkyBrightnessDataset(
            metadata=metadata, flux=flux, input_features=INPUT_FEATURES,
        )

    # [2/5] Create Train/Test Split
    print("\n[2/5] Creating train/test split...")
    dataset_size = len(dataset)
    test_size = int(args.val_split * dataset_size)
    train_size = dataset_size - test_size

    gen = torch.Generator().manual_seed(args.seed)
    train_set, test_set = random_split(dataset, [train_size, test_size], generator=gen)
    print(f"  Train: {len(train_set):,} | Test: {len(test_set):,}")

    # [3/5] Create DataLoaders
    print("\n[3/5] Creating data loaders...")
    train_loader = NumpyLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = NumpyLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # [4/5] Initialize Model
    print("\n[4/5] Initializing broadband MLP...")
    model = make_broadbandMLP(
        in_size=IN_SIZE, out_size=OUT_SIZE,
        width_size=args.width, depth=args.depth,
        key=jr.PRNGKey(args.seed),
    )
    print(f"  Architecture: {IN_SIZE} -> {args.width}x{args.depth} -> {OUT_SIZE}")

    # [5/5] Configure & Train
    print("\n[5/5] Training...")
    config = TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        loss=args.loss,
        huber_delta=args.huber_delta,
        save_best=not args.no_save,
        save_dir=args.save_dir,
        run_name=args.run_name,
        print_every=args.print_every,
        validate_every=args.validate_every,
    )

    # wandb setup
    wandb_config = None
    on_epoch_end_cb = None

    if args.wandb:
        from desisky.training import WandbConfig
        from desisky.training.wandb_utils import log_figure
        from desisky.training.dataset import gather_full_data
        import matplotlib.pyplot as plt

        wandb_config = WandbConfig(
            project=args.wandb_project,
            entity=args.wandb_entity,
            run_name=None,
            tags=[t.strip() for t in args.wandb_tags.split(",") if t.strip()]
                  + ["broadband", "moon"],
            log_every=args.log_every,
            viz_every=args.viz_every,
        )

        # Gather full train/test data for visualization
        X_train, y_train, _, _ = gather_full_data(train_loader)
        X_test, y_test, _, _ = gather_full_data(test_loader)

        # Scalar metrics (train/loss, val/loss) are logged by the
        # BroadbandTrainer._train_loop(). This callback only does visualizations.
        def on_epoch_end(model, history, epoch):
            """Log per-band 2x2 scatter+residual panels to wandb."""
            if epoch % args.viz_every != 0:
                return
            try:
                from desisky.visualization.plots import plot_broadband_band_panel
                band_names = ["V", "g", "r", "z"]
                for band_idx, band_name in enumerate(band_names):
                    fig = plot_broadband_band_panel(
                        model, X_train, y_train, X_test, y_test,
                        band_idx=band_idx, band_name=band_name,
                    )
                    log_figure(f"viz/{band_name}", fig, epoch)
                    plt.close(fig)
            except Exception as e:
                print(f"  Warning: viz callback failed: {e}")

        on_epoch_end_cb = on_epoch_end

    trainer = BroadbandTrainer(
        model, config,
        wandb_config=wandb_config,
        on_epoch_end=on_epoch_end_cb,
    )
    trained_model, history = trainer.train(train_loader, test_loader)

    print(f"\nBest test loss: {history.best_test_loss:.6f} (epoch {history.best_epoch})")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted.")
        sys.exit(1)
