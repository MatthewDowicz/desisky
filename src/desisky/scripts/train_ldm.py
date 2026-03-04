#!/usr/bin/env python
# SPDX-FileCopyrightText: 2025-present MatthewDowicz <mjdowicz@gmail.com>
#
# SPDX-License-Identifier: MIT

"""
Train a Latent Diffusion Model (LDM)
=====================================

Trains a conditional 1D U-Net latent diffusion model using the EDM framework
(Karras et al. 2022). Supports dark, moon, and twilight variants with
variant-specific conditioning columns.

Usage:
    desisky-train-ldm --variant dark --epochs 200
    desisky-train-ldm --variant moon --epochs 300 --wandb
    desisky-train-ldm --variant twilight --vae-path my_vae.eqx
"""

import argparse
import sys
from pathlib import Path

import jax
import jax.random as jr
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import torch

from desisky.io import load_or_builtin
from desisky.training import (
    NumpyLoader,
    LatentDiffusionTrainer,
    LDMTrainingConfig,
)
from desisky.training.ldm_trainer import fit_conditioning_scaler, normalize_conditioning
from desisky.models.ldm import make_UNet1D_cond, compute_sigma_data

# Variant-specific conditioning columns
CONDITIONING_COLS = {
    "dark": [
        "OBSALT", "TRANSPARENCY_GFA", "SUNALT", "SOLFLUX",
        "ECLLON", "ECLLAT", "GALLON", "GALLAT",
    ],
    "moon": [
        "OBSALT", "TRANSPARENCY_GFA", "SUNALT",
        "MOONALT", "MOONSEP", "MOONFRAC",
    ],
    "twilight": [
        "OBSALT", "TRANSPARENCY_GFA", "SUNALT", "SUNSEP",
    ],
}


def make_ldm_epoch_callback(
    vae, wavelength, flux, conditioning, val_idx,
    scaler, config, conditioning_cols, viz_every,
):
    """Create on_epoch_end callback for LDM wandb training.

    Logs 4 types of visualizations:
    1. Broadband CDFs — V, g, r, z (real vs generated) with EMD
    2. Airglow CDFs — per emission line with EMD
    3. Latent corner comparison — real vs generated with per-dim EMD
    4. Conditional validation grids — one grid per conditioning variable
       (e.g., dark produces 8 grids, moon 6, twilight 4)
    """
    wl_np = np.array(wavelength)
    val_conditioning_raw = conditioning[val_idx]

    vae_inference = eqx.nn.inference_mode(vae)

    def on_epoch_end(model, ema_model, history, epoch):
        if epoch % viz_every != 0:
            return
        if ema_model is None:
            return

        from desisky.training.wandb_utils import log_figure
        from desisky.inference import LatentDiffusionSampler
        from desisky.visualization import (
            plot_cdf_comparison,
            plot_latent_corner_comparison,
            plot_conditional_validation_grid,
        )
        from desisky.data import (
            compute_broadband_mags, measure_airglow_intensities,
            BROADBAND_NAMES, AIRGLOW_CDF_NAMES,
        )
        import matplotlib.pyplot as plt

        try:
            import warnings

            sampler = LatentDiffusionSampler(
                ldm_model=ema_model, vae_model=vae,
                sigma_data=config.sigma_data,
                conditioning_scaler=scaler,
                num_steps=250, latent_channels=1, latent_dim=8,
            )

            # Use all validation data
            n_gen = len(val_idx)
            gen_cond = jnp.array(val_conditioning_raw[:n_gen])
            generated = np.array(sampler.sample(
                key=jr.PRNGKey(epoch), conditioning=gen_cond, guidance_scale=1.0,
            ))
            real = np.array(flux[val_idx[:n_gen]])

            # Suppress speclite log10 warnings from negative flux in
            # early-training generated spectra (expected and harmless)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="invalid value.*log10")
                real_mags = compute_broadband_mags(wl_np, real)
                gen_mags = compute_broadband_mags(wl_np, generated)
                real_ag = measure_airglow_intensities(wl_np, real)
                gen_ag = measure_airglow_intensities(wl_np, generated)

            # 1. Broadband CDFs (one per band)
            for j, band_name in enumerate(BROADBAND_NAMES):
                fig, _ = plot_cdf_comparison(
                    real_mags[:, j:j+1], gen_mags[:, j:j+1], [band_name],
                )
                log_figure(f"viz/{band_name}", fig, epoch)
                plt.close(fig)

            # 2. Airglow CDFs (one per line)
            ag_names = [n for n in AIRGLOW_CDF_NAMES if n in real_ag.columns]
            for line_name in ag_names:
                fig, _ = plot_cdf_comparison(
                    real_ag[[line_name]].to_numpy(),
                    gen_ag[[line_name]].to_numpy(),
                    [line_name],
                )
                safe_name = line_name.replace(" ", "_").replace("(", "-").replace(")", "-")
                log_figure(f"viz/{safe_name}", fig, epoch)
                plt.close(fig)

            # 3. Latent corner comparison (real vs generated)
            real_latents = np.array(vae_inference(real, jr.PRNGKey(0))["latent"])
            gen_latents = np.array(vae_inference(generated, jr.PRNGKey(0))["latent"])
            fig_corner = plot_latent_corner_comparison(
                real_latents, gen_latents,
                labels=[f"z{i}" for i in range(real_latents.shape[1])],
            )
            log_figure("viz/latent_corner", fig_corner, epoch)
            plt.close(fig_corner)

            # 4. Conditional validation grids — one per conditioning variable
            real_features = np.column_stack([real_mags, real_ag[ag_names].to_numpy()])
            gen_features = np.column_stack([gen_mags, gen_ag[ag_names].to_numpy()])
            feature_names = list(BROADBAND_NAMES) + ag_names

            for col_idx, col_name in enumerate(conditioning_cols):
                fig_grid = plot_conditional_validation_grid(
                    real_features, gen_features,
                    cond_values=val_conditioning_raw[:n_gen, col_idx],
                    cond_name=col_name,
                    feature_names=feature_names,
                )
                log_figure(f"viz/cond_grid_{col_name}", fig_grid, epoch)
                plt.close(fig_grid)
        except Exception as e:
            print(f"  Warning: viz callback failed: {e}")

    return on_epoch_end


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Latent Diffusion Model on DESI sky spectra latents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Variant
    parser.add_argument("--variant", choices=["dark", "moon", "twilight"], default="dark")
    # Architecture
    parser.add_argument("--hidden", type=int, default=32, help="Base hidden channels")
    parser.add_argument("--levels", type=int, default=3, help="U-Net depth")
    parser.add_argument("--emb-dim", type=int, default=32, help="Embedding dimension")
    # Training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dropout-p", type=float, default=0.1, help="CFG dropout probability")
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--validate-every", type=int, default=1)
    # Models
    parser.add_argument("--vae-path", type=str, default=None,
                        help="Path to custom VAE checkpoint (default: builtin)")
    # Data
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to user data. Supports: "
                             ".npz (pre-processed: flux, conditioning, wavelength keys); "
                             ".fits (SkySpecVAC format, auto-filtered by --variant); "
                             ".csv (metadata, requires --flux-path for spectra). "
                             "Default: download DESI subset for variant")
    parser.add_argument("--flux-path", type=str, default=None,
                        help="Path to .npy flux array. Required when --data-path is .csv")
    # Checkpointing
    parser.add_argument("--run-name", type=str, default=None,
                        help="Run name (default: ldm_<variant>)")
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save checkpoints (useful for sweeps/testing)")
    # wandb
    parser.add_argument("--wandb", action="store_true", help="Enable wandb tracking")
    parser.add_argument("--wandb-project", type=str, default="desisky-ldm")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, default="")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--viz-every", type=int, default=20)
    # Other
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-every", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    variant = args.variant
    run_name = args.run_name or f"ldm_{variant}"
    cond_cols = CONDITIONING_COLS[variant]
    meta_dim = len(cond_cols)

    print("=" * 80)
    print(f"DESI LDM Training — {variant} (meta_dim={meta_dim})")
    print("=" * 80)

    # [1/7] Load VAE
    print("\n[1/7] Loading VAE encoder...")
    vae, vae_meta = load_or_builtin("vae", path=args.vae_path)
    vae_inference = eqx.nn.inference_mode(vae)
    vae_source = args.vae_path or "builtin"
    print(f"  VAE: {vae_source} (latent_dim={vae.latent_dim})")

    # Variant-specific quality filters (same as SkySpecVAC subset methods)
    VARIANT_FILTERS = {
        "dark": lambda m: (
            (m["SUNALT"] < -20) & (m["MOONALT"] < -5)
            & (m["TRANSPARENCY_GFA"] > 0)
        ),
        "moon": lambda m: (
            (m["SUNALT"] < -20) & (m["MOONALT"] > 5)
            & (m["MOONFRAC"] > 0.5) & (m["MOONSEP"] <= 90)
            & (m["TRANSPARENCY_GFA"] > 0)
        ),
        "twilight": lambda m: (
            (m["SUNALT"] > -20) & (m["MOONALT"] <= -5)
            & (m["SUNSEP"] <= 110) & (m["TRANSPARENCY_GFA"] > 0)
        ),
    }

    # [2/7] Load Data
    print(f"\n[2/7] Loading {variant} sky spectra...")
    if args.data_path:
        p = Path(args.data_path)
        ext = p.suffix.lower()

        if ext == ".npz":
            # Pre-processed data: expects flux, conditioning, wavelength keys
            data = np.load(p)
            flux = data["flux"]
            conditioning = data["conditioning"]
            wavelength = data["wavelength"]
            print(f"  Loaded pre-processed data: {flux.shape[0]:,} spectra")
        elif ext in (".fits", ".csv"):
            # Raw data with metadata columns — auto-filter by variant
            if ext == ".fits":
                from desisky.data.skyspec import load_skyspec_vac
                wavelength, flux, metadata = load_skyspec_vac(
                    p, as_dataframe=True,
                )
            else:
                import pandas as pd
                if not args.flux_path:
                    sys.exit("--flux-path is required when --data-path is .csv")
                metadata = pd.read_csv(p)
                flux = np.load(args.flux_path)
                # Use DESI wavelength grid
                from desisky.data import SkySpecVAC
                wavelength, _, _ = SkySpecVAC(version="v1.0", download=True).load()

            # Apply variant-specific filter
            variant_mask = VARIANT_FILTERS[variant](metadata)
            n_before = len(metadata)
            flux = flux[variant_mask]
            metadata = metadata[variant_mask].reset_index(drop=True)
            print(f"  Filtered {variant}: {len(metadata):,} / {n_before:,} spectra")

            # Enrich and extract conditioning
            from desisky.data import (
                attach_solar_flux, add_galactic_coordinates,
                add_ecliptic_coordinates,
            )
            metadata = attach_solar_flux(metadata, time_tolerance="12h")
            metadata = add_galactic_coordinates(metadata)
            metadata = add_ecliptic_coordinates(metadata)
            conditioning = metadata[cond_cols].to_numpy().astype(np.float32)

            # Remove NaN/Inf rows
            finite_mask = np.isfinite(conditioning).all(axis=1)
            n_bad = (~finite_mask).sum()
            if n_bad > 0:
                print(f"  Removing {n_bad} rows with NaN/Inf values")
                flux = flux[finite_mask]
                conditioning = conditioning[finite_mask]
        else:
            sys.exit(f"Unsupported file format '{ext}'. Use .npz, .fits, or .csv")

        print(f"  Loaded {len(flux):,} {variant} spectra from {p.name}")
    else:
        from desisky.data import SkySpecVAC
        vac = SkySpecVAC(version="v1.0", download=True)
        # Map CLI variant names to SkySpecVAC method names
        loader_names = {
            "dark": "load_dark_time",
            "moon": "load_moon_contaminated",
            "twilight": "load_sun_contaminated",
        }
        loader = getattr(vac, loader_names[variant])
        wavelength, flux, metadata = loader(enrich=True)

        # Attach derived columns (solar flux, galactic/ecliptic coords).
        # Applied unconditionally so any variant can use any column.
        from desisky.data import (
            attach_solar_flux, add_galactic_coordinates,
            add_ecliptic_coordinates,
        )
        metadata = attach_solar_flux(metadata, time_tolerance="12h")
        metadata = add_galactic_coordinates(metadata)
        metadata = add_ecliptic_coordinates(metadata)

        conditioning = metadata[cond_cols].to_numpy().astype(np.float32)

        # Remove rows with NaN/Inf in conditioning columns
        finite_mask = np.isfinite(conditioning).all(axis=1)
        n_bad = (~finite_mask).sum()
        if n_bad > 0:
            print(f"  Removing {n_bad} rows with NaN/Inf values")
            flux = flux[finite_mask]
            conditioning = conditioning[finite_mask]
            metadata = metadata.loc[finite_mask].reset_index(drop=True)

        print(f"  Loaded {len(metadata):,} {variant} spectra")

    # [3/7] Encode to Latent Space
    # Use VAE's full forward pass (encode → reparameterize → sample) so
    # the diffusion model sees the same latent distribution the decoder expects.
    print("\n[3/7] Encoding spectra to latent space...")
    flux_jax = jnp.array(flux.astype(np.float32))
    key = jr.PRNGKey(args.seed)
    # Encode in batches to avoid OOM
    batch_size = 512
    all_latents = []
    for i in range(0, len(flux_jax), batch_size):
        batch = flux_jax[i:i+batch_size]
        key, subkey = jr.split(key)
        result = vae_inference(batch, subkey)
        all_latents.append(np.array(result["latent"]))
    latents = np.concatenate(all_latents, axis=0)  # (N, latent_dim)
    # Reshape for U-Net: (N, 1, latent_dim)
    latents = latents[:, None, :]
    print(f"  Latent shape: {latents.shape}")

    # [4/7] Compute sigma_data and fit conditioning scaler
    print("\n[4/7] Computing sigma_data and conditioning scaler...")
    sigma_data = float(compute_sigma_data(latents))
    print(f"  sigma_data = {sigma_data:.4f}")

    # Train/val split
    rng = np.random.default_rng(args.seed)
    n_total = len(latents)
    n_val = int(args.val_split * n_total)
    perm = rng.permutation(n_total)
    train_idx = perm[n_val:]
    val_idx = perm[:n_val]
    print(f"  Train: {len(train_idx):,} | Val: {len(val_idx):,}")

    # Fit scaler on training conditioning only
    scaler = fit_conditioning_scaler(conditioning[train_idx], cond_cols)
    cond_norm = normalize_conditioning(conditioning, scaler)

    # [5/7] Create DataLoaders
    print("\n[5/7] Creating data loaders...")
    train_latents = torch.from_numpy(latents[train_idx].astype(np.float32))
    train_cond = torch.from_numpy(cond_norm[train_idx].astype(np.float32))
    val_latents = torch.from_numpy(latents[val_idx].astype(np.float32))
    val_cond = torch.from_numpy(cond_norm[val_idx].astype(np.float32))

    train_ds = torch.utils.data.TensorDataset(train_latents, train_cond)
    val_ds = torch.utils.data.TensorDataset(val_latents, val_cond)
    train_loader = NumpyLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = NumpyLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # [6/7] Initialize U-Net Model
    print("\n[6/7] Initializing UNet1D_cond...")
    model = make_UNet1D_cond(
        in_ch=1, out_ch=1, meta_dim=meta_dim,
        hidden=args.hidden, levels=args.levels, emb_dim=args.emb_dim,
        key=jr.PRNGKey(args.seed),
    )
    print(f"  hidden={args.hidden}, levels={args.levels}, emb_dim={args.emb_dim}")

    # [7/7] Configure & Train
    print("\n[7/7] Training...")
    config = LDMTrainingConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        meta_dim=meta_dim,
        sigma_data=sigma_data,
        dropout_p=args.dropout_p,
        ema_decay=args.ema_decay,
        save_best=not args.no_save,
        run_name=run_name,
        save_dir=args.save_dir,
        random_seed=args.seed,
        print_every=args.print_every,
        validate_every=args.validate_every,
        conditioning_scaler=scaler,
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
                  + ["ldm", variant],
            log_every=args.log_every,
            viz_every=args.viz_every,
        )

        on_epoch_end_cb = make_ldm_epoch_callback(
            vae=vae,
            wavelength=wavelength,
            flux=flux,
            conditioning=conditioning,
            val_idx=val_idx,
            scaler=scaler,
            config=config,
            conditioning_cols=cond_cols,
            viz_every=args.viz_every,
        )

    trainer = LatentDiffusionTrainer(
        model, config,
        wandb_config=wandb_config,
        on_epoch_end=on_epoch_end_cb,
    )
    trained_model, ema_model, history = trainer.train(train_loader, val_loader)

    print(f"\nBest val loss: {history.best_val_loss:.6f} (epoch {history.best_epoch})")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted.")
        sys.exit(1)
