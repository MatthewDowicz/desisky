#!/usr/bin/env python
# SPDX-FileCopyrightText: 2025-present MatthewDowicz <mjdowicz@gmail.com>
#
# SPDX-License-Identifier: MIT

"""
Train a SkyVAE (InfoVAE-MMD)
==============================

Trains a Variational Autoencoder on DESI sky spectra using the InfoVAE-MMD
objective. The VAE learns to compress 7781-dimensional sky spectra into a
low-dimensional latent representation.

Trained on all sky conditions combined, with train/test split via --val-split.

Usage:
    desisky-train-vae --epochs 100
    desisky-train-vae --epochs 200 --wandb --wandb-project my-vae
    desisky-train-vae --data-path my_spectra.npz --epochs 50
"""

import argparse
import sys
from pathlib import Path

import jax
import jax.random as jr
import jax.numpy as jnp
import numpy as np
import torch

from desisky.models.vae import make_SkyVAE
from desisky.training import (
    VAETrainer,
    VAETrainingConfig,
    NumpyLoader,
)


def classify_sky_condition(metadata):
    """Classify each exposure as dark, moon, twilight, or other."""
    conditions = np.full(len(metadata), "other", dtype=object)
    conditions[(metadata["SUNALT"] < -20) & (metadata["MOONALT"] < -5)] = "dark"
    conditions[
        (metadata["SUNALT"] < -20)
        & (metadata["MOONALT"] > 5)
        & (metadata["MOONFRAC"] > 0.5)
        & (metadata["MOONSEP"] <= 90)
    ] = "moon"
    conditions[(metadata["SUNALT"] > -20) & (metadata["MOONALT"] <= -5)] = "twilight"
    return conditions


def make_vae_epoch_callback(
    model, wavelength, test_flux, test_conditions, viz_every, wconfig,
):
    """Create the on_epoch_end callback for VAE wandb training.

    Logs 4 types of visualizations every viz_every epochs:
    1. Reconstruction plot — original vs reconstructed spectra
    2. Latent corner plot — colored by sky condition (dark/moon/twilight/other)
    3. Broadband CDFs — V, g, r, z (real vs reconstructed) with EMD
    4. Airglow CDFs — per emission line with EMD
    """
    wl_np = np.array(wavelength)

    def on_epoch_end(model, history, epoch):
        if epoch % viz_every != 0:
            return

        from desisky.training.wandb_utils import log_figure
        from desisky.visualization import (
            plot_vae_reconstructions,
            plot_latent_corner,
            plot_broadband_cdfs,
            plot_airglow_cdfs,
        )
        import matplotlib.pyplot as plt

        key = jr.PRNGKey(epoch)
        n_show = 5
        n_cdf = len(test_flux)

        try:
            # 1. Reconstruction plot
            test_batch = jnp.array(test_flux[:n_show])
            result = model(test_batch, key)
            fig_recon = plot_vae_reconstructions(
                np.array(test_batch), np.array(result["output"]), wl_np, n_samples=n_show,
            )
            log_figure("viz/reconstructions", fig_recon, epoch)
            plt.close(fig_recon)

            # 2. Latent corner plot colored by sky condition
            enc_result = model(jnp.array(test_flux), jr.PRNGKey(epoch + 1))
            latents = np.array(enc_result["latent"])
            fig_corner = plot_latent_corner(
                latents,
                labels=[f"z{i}" for i in range(latents.shape[1])],
                sky_conditions=test_conditions,
                condition_names=["dark", "moon", "other", "twilight"],
            )
            log_figure("viz/latent_corner", fig_corner, epoch)
            plt.close(fig_corner)

            # 3. Broadband CDFs (one figure per band: V, g, r, z)
            cdf_batch = jnp.array(test_flux[:n_cdf])
            cdf_result = model(cdf_batch, jr.PRNGKey(epoch + 2))
            real_spectra = np.array(cdf_batch)
            recon_spectra = np.array(cdf_result["output"])

            bb_results = plot_broadband_cdfs(wl_np, real_spectra, recon_spectra)
            for band_name, (fig, emd) in bb_results.items():
                log_figure(f"viz/{band_name}", fig, epoch)
                plt.close(fig)

            # 4. Airglow CDFs (one figure per emission line)
            ag_results = plot_airglow_cdfs(wl_np, real_spectra, recon_spectra)
            for line_name, (fig, emd) in ag_results.items():
                safe_name = line_name.replace(" ", "_").replace("(", "-").replace(")", "-")
                log_figure(f"viz/{safe_name}", fig, epoch)
                plt.close(fig)
        except Exception as e:
            print(f"  Warning: viz callback failed: {e}")

    return on_epoch_end


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a SkyVAE (InfoVAE-MMD) on DESI sky spectra",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Training schedule
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--validate-every", type=int, default=1)
    # InfoVAE hyperparameters
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--beta", type=float, default=1e-3, help="KL weight")
    parser.add_argument("--lam", type=float, default=4.0, help="Total regularization weight")
    parser.add_argument("--kernel-sigma", type=str, default="auto",
                        help="RBF bandwidth for MMD ('auto' or float)")
    parser.add_argument("--clip-gradients", action="store_true")
    # Data
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to user .npz (required key: flux (N,7781)). "
                             "Wavelength grid is hardcoded to the DESI grid. "
                             "Default: full DESI SkySpecVAC")
    # Checkpointing
    parser.add_argument("--run-name", type=str, default="sky_vae")
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save checkpoints (useful for sweeps/testing)")
    # wandb
    parser.add_argument("--wandb", action="store_true", help="Enable wandb tracking")
    parser.add_argument("--wandb-project", type=str, default="desisky-vae")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, default="")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--viz-every", type=int, default=10)
    # Quality filtering
    parser.add_argument("--no-quality-filter", action="store_true",
                        help="Disable automatic removal of known contaminated "
                             "observations (e.g., June 2021 wildfire event)")
    # Other
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-every", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("DESI SkyVAE Training (InfoVAE-MMD)")
    print("=" * 80)

    # [1/5] Load Data
    print("\n[1/5] Loading sky spectra...")
    from desisky.data import SkySpecVAC
    quality_filter = not args.no_quality_filter

    if args.data_path:
        data = np.load(args.data_path)
        flux = data["flux"]
        metadata = None
        # Only need wavelength grid — skip filter (user data should be pre-filtered)
        wavelength, _, _ = SkySpecVAC(version="v1.0", download=True,
                                      exclude_known_bad=False).load()
        if quality_filter:
            import warnings
            warnings.warn(
                "Quality filter cannot be applied to .npz data (no metadata). "
                "Ensure your data was pre-filtered, or use "
                "desisky.data.filter_known_contamination() during data preparation.",
                UserWarning, stacklevel=1,
            )
        print(f"  Loaded user data: {flux.shape[0]:,} spectra from {args.data_path}")
    else:
        vac = SkySpecVAC(version="v1.0", download=True,
                         exclude_known_bad=quality_filter)
        wavelength, flux, metadata = vac.load()
        print(f"  Loaded {len(metadata):,} spectra from DESI SkySpecVAC")

    # Classify sky conditions (for latent corner coloring during wandb)
    test_conditions = None
    if metadata is not None:
        test_conditions = classify_sky_condition(metadata)

    # [2/5] Create Train/Test Split
    print("\n[2/5] Creating train/test split...")
    n_total = len(flux)
    n_test = int(args.val_split * n_total)
    n_train = n_total - n_test

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_total)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    train_flux = flux[train_idx]
    test_flux = flux[test_idx]
    if test_conditions is not None:
        test_conditions = test_conditions[test_idx]
    print(f"  Train: {n_train:,} | Test: {n_test:,}")

    # [3/5] Create DataLoaders
    # Use raw tensors (not TensorDataset) because VAETrainer iterates as
    # `for x in loader:` and expects plain arrays. TensorDataset wraps each
    # batch in a tuple (array,), which causes a shape mismatch in the model.
    print("\n[3/5] Creating data loaders...")
    train_tensor = torch.from_numpy(train_flux.astype(np.float32))
    test_tensor = torch.from_numpy(test_flux.astype(np.float32))
    train_loader = NumpyLoader(train_tensor, batch_size=args.batch_size, shuffle=True)
    test_loader = NumpyLoader(test_tensor, batch_size=args.batch_size, shuffle=False)

    # [4/5] Initialize Model
    print("\n[4/5] Initializing SkyVAE...")
    model = make_SkyVAE(
        in_channels=flux.shape[1],
        latent_dim=args.latent_dim,
        key=jr.PRNGKey(args.seed),
    )
    print(f"  Architecture: in_channels={flux.shape[1]}, latent_dim={args.latent_dim}")

    # [5/5] Configure & Train
    print("\n[5/5] Training...")
    kernel_sigma = args.kernel_sigma
    if kernel_sigma != "auto":
        kernel_sigma = float(kernel_sigma)

    config = VAETrainingConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        beta=args.beta,
        lam=args.lam,
        kernel_sigma=kernel_sigma,
        clip_gradients=args.clip_gradients,
        save_best=not args.no_save,
        save_dir=args.save_dir,
        run_name=args.run_name,
        print_every=args.print_every,
        validate_every=args.validate_every,
        random_seed=args.seed,
    )

    # wandb setup
    wandb_config = None
    on_epoch_end_cb = None

    if args.wandb:
        from desisky.training import WandbConfig
        wandb_config = WandbConfig(
            project=args.wandb_project,
            entity=args.wandb_entity,
            run_name=None,
            tags=[t.strip() for t in args.wandb_tags.split(",") if t.strip()]
                  + ["vae"],
            log_every=args.log_every,
            viz_every=args.viz_every,
        )

        on_epoch_end_cb = make_vae_epoch_callback(
            model=model,
            wavelength=wavelength,
            test_flux=test_flux,
            test_conditions=test_conditions,
            viz_every=args.viz_every,
            wconfig=wandb_config,
        )

    trainer = VAETrainer(
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
