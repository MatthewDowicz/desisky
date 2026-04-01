#!/usr/bin/env python
"""Standalone VAE training script for NERSC Perlmutter A100s.

Works around a CUDA graph capture bug on Perlmutter A100s where
re-execution of a cached JIT'd program with different input buffers
fails.  The workaround:
  1. Single GPU (CUDA_VISIBLE_DEVICES=0)
  2. Disable XLA autotuner (--xla_gpu_autotune_level=0)
  3. Pre-load all data to GPU as one JAX array
  4. Use jax.lax.dynamic_slice inside the JIT'd function for batching
  5. Pass batch index as jnp.array (dynamic, not static) to avoid recompilation

This script mirrors the full feature set of ``desisky-train-vae`` (wandb
visualizations, loss component tracking, sky condition classification,
custom data paths, etc.) but uses the dynamic_slice pattern instead of
DataLoader batching.

Usage:
    CUDA_VISIBLE_DEVICES=0 XLA_FLAGS="--xla_gpu_autotune_level=0" \\
        python scripts/nersc_train_vae.py --epochs 100

    CUDA_VISIBLE_DEVICES=0 XLA_FLAGS="--xla_gpu_autotune_level=0" \\
        python scripts/nersc_train_vae.py --epochs 100 --wandb
"""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax

from desisky.models.vae import make_SkyVAE
from desisky.training.vae_losses import vae_loss_infovae, default_kernel_sigma
from desisky.data import SkySpecVAC
from desisky.io import save

# ---------------------------------------------------------------------------
# Module-level globals captured by make_step (set by main via `global`).
# Using module globals avoids recompilation on A100s.
# ---------------------------------------------------------------------------
_opt = None
_ks = None
_beta = None
_lam = None
_bs = None


@eqx.filter_jit
def make_step(model, opt_state, data, batch_idx, key):
    x = jax.lax.dynamic_slice(data, (batch_idx * _bs, 0), (_bs, data.shape[1]))
    (loss, aux), grads = eqx.filter_value_and_grad(vae_loss_infovae, has_aux=True)(
        model, x=x, key=key, beta=_beta, lam=_lam, kernel_sigma=_ks,
    )
    updates, opt_state = _opt.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss, aux


# ---------------------------------------------------------------------------
# Sky condition classification (same as CLI)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model, test_data, n_batches, bs, in_channels, key):
    """Evaluate model on test data. Returns total loss and component breakdown."""
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_mmd = 0.0
    n = 0
    for i in range(n_batches):
        key, subkey = jr.split(key)
        x = jax.lax.dynamic_slice(test_data, (jnp.array(i) * bs, 0), (bs, in_channels))
        loss_i, aux_i = vae_loss_infovae(
            model, x, subkey, beta=_beta, lam=_lam, kernel_sigma=_ks,
        )
        total_loss += float(loss_i) * bs
        total_recon += float(aux_i["recon"]) * bs
        total_kl += float(aux_i["kl_weighted"]) * bs
        total_mmd += float(aux_i["mmd_weighted"]) * bs
        n += bs
    return {
        "loss": total_loss / n,
        "recon": total_recon / n,
        "kl": total_kl / n,
        "mmd": total_mmd / n,
    }


# ---------------------------------------------------------------------------
# Wandb visualization callback
# ---------------------------------------------------------------------------
def run_visualizations(model, wavelength, test_flux, test_conditions, epoch):
    """Log reconstruction plots, latent corners, broadband & airglow CDFs to wandb."""
    try:
        from desisky.training.wandb_utils import log_figure
        from desisky.visualization import (
            plot_vae_reconstructions,
            plot_latent_corner,
            plot_broadband_cdfs,
            plot_airglow_cdfs,
        )
        import matplotlib.pyplot as plt

        wl_np = np.array(wavelength)
        key = jr.PRNGKey(epoch)
        n_show = 5

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

        # 3. Broadband CDFs
        cdf_result = model(jnp.array(test_flux), jr.PRNGKey(epoch + 2))
        real_spectra = np.array(test_flux)
        recon_spectra = np.array(cdf_result["output"])

        bb_results = plot_broadband_cdfs(wl_np, real_spectra, recon_spectra)
        for band_name, (fig, emd) in bb_results.items():
            log_figure(f"viz/{band_name}", fig, epoch)
            plt.close(fig)

        # 4. Airglow CDFs
        ag_results = plot_airglow_cdfs(wl_np, real_spectra, recon_spectra)
        for line_name, (fig, emd) in ag_results.items():
            safe_name = line_name.replace(" ", "_").replace("(", "-").replace(")", "-")
            log_figure(f"viz/{safe_name}", fig, epoch)
            plt.close(fig)

    except Exception as e:
        print(f"  Warning: viz callback failed at epoch {epoch}: {e}")


# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Train VAE on NERSC Perlmutter (CUDA graph workaround)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Training schedule
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--validate-every", type=int, default=1)
    # InfoVAE hyperparameters
    p.add_argument("--latent-dim", type=int, default=8)
    p.add_argument("--beta", type=float, default=1e-3, help="KL weight")
    p.add_argument("--lam", type=float, default=4.0, help="Total regularization weight")
    p.add_argument("--kernel-sigma", type=str, default="auto",
                    help="RBF bandwidth for MMD ('auto' or float)")
    p.add_argument("--clip-gradients", action="store_true")
    # Data
    p.add_argument("--data-path", type=str, default=None,
                    help="Path to user .npz (required key: flux (N,7781)). "
                         "Default: full DESI SkySpecVAC")
    p.add_argument("--metadata-path", type=str, default=None,
                    help="Path to CSV with metadata (SUNALT, MOONALT, etc.) for "
                         "sky condition classification in wandb visualizations. "
                         "Rows must align with --data-path flux.")
    # Checkpointing
    p.add_argument("--run-name", type=str, default="vae_nersc")
    p.add_argument("--save-dir", type=str, default=None)
    p.add_argument("--no-save", action="store_true",
                    help="Don't save checkpoints (useful for sweeps/testing)")
    # wandb
    p.add_argument("--wandb", action="store_true", help="Enable wandb tracking")
    p.add_argument("--wandb-project", type=str, default="desisky-vae")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-tags", type=str, default="")
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--viz-every", type=int, default=10)
    # Other
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--print-every", type=int, default=10)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global _opt, _ks, _beta, _lam, _bs
    args = parse_args()

    print("=" * 80)
    print("DESI SkyVAE Training — NERSC Perlmutter")
    print("=" * 80)
    print(f"JAX devices: {jax.devices()}")
    print(f"XLA_FLAGS: {os.environ.get('XLA_FLAGS', 'NOT SET')}")

    # [1/5] Load Data
    print("\n[1/5] Loading sky spectra...")
    vac = SkySpecVAC(version="v1.0", download=True)

    if args.data_path:
        data = np.load(args.data_path)
        flux = data["flux"]
        wavelength, _, _ = vac.load()
        print(f"  Loaded user data: {flux.shape[0]:,} spectra from {args.data_path}")

        if args.metadata_path:
            import pandas as pd
            metadata = pd.read_csv(args.metadata_path)
            print(f"  Loaded metadata: {len(metadata):,} rows from {args.metadata_path}")
            if len(metadata) != len(flux):
                print(f"  Warning: metadata rows ({len(metadata)}) != flux rows ({len(flux)}), disabling sky conditions")
                metadata = None
        else:
            metadata = None
    else:
        wavelength, flux, metadata = vac.load()
        print(f"  Loaded {len(flux):,} spectra from DESI SkySpecVAC")

    in_channels = flux.shape[1]

    # Classify sky conditions (for latent corner coloring during wandb)
    test_conditions = None
    if metadata is not None:
        test_conditions = classify_sky_condition(metadata)

    # [2/5] Create Train/Test Split
    print("\n[2/5] Creating train/test split...")
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(flux))
    n_test = int(args.val_split * len(flux))
    train_idx = perm[: len(flux) - n_test]
    test_idx = perm[len(flux) - n_test :]

    train_flux = flux[train_idx].astype(np.float32)
    test_flux = flux[test_idx].astype(np.float32)
    if test_conditions is not None:
        test_conditions = test_conditions[test_idx]

    # Trim to multiples of batch_size for dynamic_slice
    bs = args.batch_size
    n_train_trimmed = (len(train_flux) // bs) * bs
    n_test_trimmed = (len(test_flux) // bs) * bs
    rng.shuffle(train_flux)
    train_data = jnp.asarray(train_flux[:n_train_trimmed])
    test_data = jnp.asarray(test_flux[:n_test_trimmed])
    n_train_batches = n_train_trimmed // bs
    n_test_batches = n_test_trimmed // bs
    print(f"  Train: {n_train_trimmed:,} ({n_train_batches} batches) | "
          f"Test: {n_test_trimmed:,} ({n_test_batches} batches)")

    # [3/5] Initialize Model
    print("\n[3/5] Initializing SkyVAE...")
    model = make_SkyVAE(
        in_channels=in_channels, latent_dim=args.latent_dim, key=jr.PRNGKey(args.seed),
    )
    print(f"  Architecture: in_channels={in_channels}, latent_dim={args.latent_dim}")

    # [4/5] Set up optimizer and globals
    print("\n[4/5] Configuring optimizer...")
    kernel_sigma = args.kernel_sigma
    if kernel_sigma == "auto":
        kernel_sigma = default_kernel_sigma(args.latent_dim)
    else:
        kernel_sigma = float(kernel_sigma)

    if args.clip_gradients:
        _opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(args.learning_rate))
    else:
        _opt = optax.adam(args.learning_rate)
    _ks = kernel_sigma
    _beta = args.beta
    _lam = args.lam
    _bs = bs
    opt_state = _opt.init(eqx.filter(model, eqx.is_array))

    # [5/5] wandb setup
    use_wandb = args.wandb
    if use_wandb:
        import wandb
        tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()] + ["vae", "nersc"]
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            tags=tags,
        )
        run_name = args.run_name if args.run_name != "vae_nersc" else wandb.run.name
        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")
        print(f"  wandb run: {wandb.run.url}")
    else:
        run_name = args.run_name

    # Training loop
    print(f"\n[5/5] Training for {args.epochs} epochs...")
    key = jr.PRNGKey(args.seed)
    best_test_loss = float("inf")
    best_epoch = -1

    try:
        for epoch in range(args.epochs):
            # Shuffle training data each epoch (CPU → GPU)
            train_np = np.array(train_data)
            rng.shuffle(train_np)
            train_data = jnp.asarray(train_np)

            # --- Train ---
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_kl = 0.0
            epoch_mmd = 0.0
            for i in range(n_train_batches):
                key, subkey = jr.split(key)
                model, opt_state, loss_val, aux = make_step(
                    model, opt_state, train_data, jnp.array(i), subkey,
                )
                epoch_loss += float(loss_val) * bs
                epoch_recon += float(aux["recon"]) * bs
                epoch_kl += float(aux["kl_weighted"]) * bs
                epoch_mmd += float(aux["mmd_weighted"]) * bs
            epoch_loss /= n_train_trimmed
            epoch_recon /= n_train_trimmed
            epoch_kl /= n_train_trimmed
            epoch_mmd /= n_train_trimmed

            # --- Validate ---
            test_metrics = None
            if epoch % args.validate_every == 0:
                key, eval_key = jr.split(key)
                test_metrics = evaluate(model, test_data, n_test_batches, bs, in_channels, eval_key)

                improved = ""
                if test_metrics["loss"] < best_test_loss:
                    best_test_loss = test_metrics["loss"]
                    best_epoch = epoch
                    improved = " *"

                    # Save checkpoint
                    if not args.no_save:
                        save_dir = Path(args.save_dir) if args.save_dir else (
                            Path.home() / ".cache" / "desisky" / "saved_models" / "vae"
                        )
                        save_dir.mkdir(parents=True, exist_ok=True)
                        save_path = save_dir / f"{run_name}.eqx"
                        save(save_path, model, {
                            "schema": 1,
                            "arch": {"in_channels": in_channels, "latent_dim": args.latent_dim},
                            "training": {
                                "date": datetime.now().isoformat(),
                                "epoch": epoch,
                                "train_loss": epoch_loss,
                                "test_loss": test_metrics["loss"],
                                "test_recon": test_metrics["recon"],
                                "test_kl": test_metrics["kl"],
                                "test_mmd": test_metrics["mmd"],
                                "beta": args.beta,
                                "lam": args.lam,
                                "learning_rate": args.learning_rate,
                                "kernel_sigma": float(_ks),
                                "seed": args.seed,
                            },
                        })

                # Print progress
                if epoch % args.print_every == 0:
                    print(
                        f"Epoch {epoch:4d}/{args.epochs} | "
                        f"Train: {epoch_loss:.6f} (R:{epoch_recon:.4f} KL:{epoch_kl:.4f} MMD:{epoch_mmd:.4f}) | "
                        f"Test: {test_metrics['loss']:.6f} (R:{test_metrics['recon']:.4f}) | "
                        f"Best: {best_test_loss:.6f} (ep {best_epoch}){improved}"
                    )

            elif epoch % args.print_every == 0:
                print(
                    f"Epoch {epoch:4d}/{args.epochs} | "
                    f"Train: {epoch_loss:.6f} (R:{epoch_recon:.4f} KL:{epoch_kl:.4f} MMD:{epoch_mmd:.4f})"
                )

            # --- wandb logging ---
            if use_wandb:
                import wandb
                log_dict = {
                    "train/loss": epoch_loss,
                    "train/recon": epoch_recon,
                    "train/kl": epoch_kl,
                    "train/mmd": epoch_mmd,
                    "train/loss_z": epoch_kl + epoch_mmd,
                    "epoch": epoch,
                }
                if test_metrics is not None:
                    log_dict.update({
                        "val/loss": test_metrics["loss"],
                        "val/recon": test_metrics["recon"],
                        "val/kl": test_metrics["kl"],
                        "val/mmd": test_metrics["mmd"],
                        "val/loss_z": test_metrics["kl"] + test_metrics["mmd"],
                    })
                if epoch % args.log_every == 0:
                    wandb.log(log_dict)

                # Visualizations
                if epoch % args.viz_every == 0:
                    run_visualizations(model, wavelength, test_flux, test_conditions, epoch)

        print(f"\nDone. Best test loss: {best_test_loss:.6f} (epoch {best_epoch})")

    except KeyboardInterrupt:
        print(f"\n\nTraining interrupted at epoch {epoch}.")
        print(f"Best test loss so far: {best_test_loss:.6f} (epoch {best_epoch})")

    finally:
        if use_wandb:
            import wandb
            if wandb.run is not None:
                print(f"  wandb run: {wandb.run.url}")
            wandb.finish()


if __name__ == "__main__":
    main()
