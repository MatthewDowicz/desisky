#!/usr/bin/env python
"""Standalone VAE training script for NERSC Perlmutter A100s.

Works around a CUDA graph capture bug on Perlmutter A100s where any
second XLA compilation in a process fails.  The workaround:
  1. Single GPU (CUDA_VISIBLE_DEVICES=0)
  2. Disable XLA autotuner (--xla_gpu_autotune_level=0)
  3. Pre-load all data to GPU as one JAX array
  4. Use jax.lax.dynamic_slice inside the JIT'd function for batching
  5. Pass batch index as jnp.array (dynamic, not static) to avoid recompilation

Usage:
    CUDA_VISIBLE_DEVICES=0 XLA_FLAGS="--xla_gpu_autotune_level=0" \\
        python scripts/nersc_train_vae.py --epochs 100

    CUDA_VISIBLE_DEVICES=0 XLA_FLAGS="--xla_gpu_autotune_level=0" \\
        python scripts/nersc_train_vae.py --epochs 100 --wandb
"""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")

import argparse
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
from pathlib import Path

from desisky.models.vae import make_SkyVAE
from desisky.training.vae_losses import vae_loss_infovae, default_kernel_sigma
from desisky.data import SkySpecVAC
from desisky.io import save

# Module-level globals captured by make_step (set by main via `global`).
# Using module globals avoids recompilation on A100s.
opt = None
ks = None


@eqx.filter_jit
def make_step(model, opt_state, data, batch_idx, key):
    x = jax.lax.dynamic_slice(data, (batch_idx * 64, 0), (64, data.shape[1]))
    (loss, aux), grads = eqx.filter_value_and_grad(vae_loss_infovae, has_aux=True)(
        model, x=x, key=key, beta=1e-3, lam=4.0, kernel_sigma=ks,
    )
    updates, opt_state = opt.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss, aux


def parse_args():
    p = argparse.ArgumentParser(description="Train VAE on NERSC Perlmutter")
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--beta", type=float, default=1e-3)
    p.add_argument("--lam", type=float, default=4.0)
    p.add_argument("--latent-dim", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--save-dir", type=str, default=None)
    p.add_argument("--run-name", type=str, default="vae_nersc")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="desisky-vae")
    return p.parse_args()


def main():
    global opt, ks
    args = parse_args()

    print(f"JAX devices: {jax.devices()}")
    print(f"XLA_FLAGS: {os.environ.get('XLA_FLAGS', 'NOT SET')}")

    # Load data
    print("\nLoading data...")
    vac = SkySpecVAC(version="v1.0", download=True)
    wavelength, flux, metadata = vac.load()
    in_channels = flux.shape[1]
    print(f"  {len(flux):,} spectra, {in_channels} wavelength bins")

    # Train/test split
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(flux))
    n_test = int(args.val_split * len(flux))
    train_flux = flux[perm[: len(flux) - n_test]].astype(np.float32)
    test_flux = flux[perm[len(flux) - n_test :]].astype(np.float32)

    # Trim to multiple of batch_size and shuffle
    bs = args.batch_size
    n_train_trimmed = (len(train_flux) // bs) * bs
    n_test_trimmed = (len(test_flux) // bs) * bs
    rng.shuffle(train_flux)
    train_data = jnp.asarray(train_flux[:n_train_trimmed])
    test_data = jnp.asarray(test_flux[:n_test_trimmed])
    n_train_batches = n_train_trimmed // bs
    n_test_batches = n_test_trimmed // bs
    print(f"  Train: {n_train_trimmed:,} ({n_train_batches} batches) | Test: {n_test_trimmed:,} ({n_test_batches} batches)")

    # Model + optimizer
    model = make_SkyVAE(
        in_channels=in_channels, latent_dim=args.latent_dim, key=jr.PRNGKey(args.seed)
    )
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(args.learning_rate))
    opt_state = opt.init(eqx.filter(model, eqx.is_array))
    ks = default_kernel_sigma(args.latent_dim)

    # Update make_step's hardcoded beta/lam if non-default
    # (For now, beta=1e-3 and lam=4.0 are hardcoded in make_step above.
    #  If you need different values, edit the function directly.)

    # wandb
    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, config=vars(args))

    # Training loop
    key = jr.PRNGKey(args.seed)
    best_test_loss = float("inf")
    best_epoch = -1

    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        # Shuffle training data each epoch (on CPU, then upload)
        train_np = np.array(train_data)
        rng.shuffle(train_np)
        train_data = jnp.asarray(train_np)

        # Train
        epoch_loss = 0.0
        for i in range(n_train_batches):
            key, subkey = jr.split(key)
            model, opt_state, loss_val, aux = make_step(
                model, opt_state, train_data, jnp.array(i), subkey,
            )
            epoch_loss += float(loss_val) * bs
        epoch_loss /= n_train_trimmed

        # Evaluate (using same dynamic_slice pattern, no separate compilation)
        key, eval_key = jr.split(key)
        test_loss = 0.0
        for i in range(n_test_batches):
            eval_key, subkey = jr.split(eval_key)
            x = jax.lax.dynamic_slice(test_data, (jnp.array(i) * bs, 0), (bs, in_channels))
            loss_i, _ = vae_loss_infovae(
                model, x, subkey, beta=args.beta, lam=args.lam, kernel_sigma=ks,
            )
            test_loss += float(loss_i) * bs
        test_loss /= n_test_trimmed

        improved = ""
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_epoch = epoch

            save_dir = Path(args.save_dir) if args.save_dir else Path.home() / ".cache" / "desisky" / "saved_models" / "vae"
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"{args.run_name}.eqx"
            save(save_path, model, {
                "schema": 1,
                "arch": {"in_channels": in_channels, "latent_dim": args.latent_dim},
                "training": {
                    "epoch": epoch,
                    "test_loss": test_loss,
                    "train_loss": epoch_loss,
                    "beta": args.beta,
                    "lam": args.lam,
                    "learning_rate": args.learning_rate,
                    "seed": args.seed,
                },
            })
            improved = " *"

        print(
            f"Epoch {epoch:4d}/{args.epochs} | "
            f"Train: {epoch_loss:.6f} | Test: {test_loss:.6f} | "
            f"Best: {best_test_loss:.6f} (ep {best_epoch}){improved}"
        )

        if args.wandb:
            import wandb
            wandb.log({
                "train/loss": epoch_loss,
                "val/loss": test_loss,
                "epoch": epoch,
            })

    print(f"\nDone. Best test loss: {best_test_loss:.6f} (epoch {best_epoch})")
    if args.wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
