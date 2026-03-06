#!/usr/bin/env python
# SPDX-FileCopyrightText: 2025-present MatthewDowicz <mjdowicz@gmail.com>
#
# SPDX-License-Identifier: MIT

"""
VAE Encode + Reconstruct
=========================

Run the SkyVAE on DESI sky spectra to produce latent representations
and reconstructions. Supports subset selection (dark, moon, twilight).

Usage:
    desisky-infer-vae
    desisky-infer-vae --subset dark --output dark_latents.npz
    desisky-infer-vae --model-path my_vae.eqx --n-samples 1000
"""

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import equinox as eqx

from desisky.io import load_or_builtin


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run VAE encode + reconstruct on DESI sky spectra",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to custom VAE checkpoint (default: builtin)")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to user .npz (required key: flux (N,7781)). "
                             "Wavelength hardcoded to DESI grid. Default: DESI SkySpecVAC")
    parser.add_argument("--subset", choices=["full", "dark", "moon", "twilight"],
                        default="full",
                        help="Which sky condition subset to load")
    parser.add_argument("--output", type=str, default="vae_output.npz")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Limit number of spectra")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("SkyVAE Encode + Reconstruct")
    print("=" * 60)

    # [1/3] Load Model
    print("\n[1/3] Loading VAE model...")
    model, meta = load_or_builtin("vae", path=args.model_path)
    model = eqx.nn.inference_mode(model)
    model_source = args.model_path or "builtin"
    print(f"  Model: {model_source} (latent_dim={model.latent_dim})")

    # [2/3] Load Data
    print("\n[2/3] Loading data...")
    from desisky.data import SkySpecVAC
    vac = SkySpecVAC(version="v1.0", download=True)

    if args.data_path:
        data = np.load(args.data_path)
        flux = data["flux"]
        # DESI wavelength grid (hardcoded)
        wavelength, _, _ = vac.load()
        print(f"  Loaded user data: {flux.shape[0]:,} spectra")
    else:
        if args.subset == "full":
            wavelength, flux, metadata = vac.load()
        elif args.subset == "dark":
            wavelength, flux, metadata = vac.load_dark_time()
        elif args.subset == "moon":
            wavelength, flux, metadata = vac.load_moon_contaminated()
        elif args.subset == "twilight":
            wavelength, flux, metadata = vac.load_sun_contaminated()
        print(f"  Loaded {len(flux):,} {args.subset} spectra")

    if args.n_samples is not None:
        flux = flux[:args.n_samples]

    # [3/3] Encode + Reconstruct
    print(f"\n[3/3] Processing {len(flux):,} spectra...")
    key = jr.PRNGKey(args.seed)
    flux_jax = jnp.array(flux.astype(np.float32))

    # Process in batches (future-proofs for larger datasets)
    batch_size = 1024
    all_latents, all_means, all_logvars, all_recon = [], [], [], []

    for i in range(0, len(flux_jax), batch_size):
        batch = flux_jax[i:i+batch_size]
        key, subkey = jr.split(key)
        result = jax.vmap(lambda x: model(x, subkey))(batch)

        all_latents.append(np.array(result["latent"]))
        all_means.append(np.array(result["mean"]))
        all_logvars.append(np.array(result["logvar"]))
        all_recon.append(np.array(result["output"]))

    latents = np.concatenate(all_latents)
    means = np.concatenate(all_means)
    logvars = np.concatenate(all_logvars)
    reconstructed = np.concatenate(all_recon)

    # Per-spectrum reconstruction error
    recon_error = np.mean((flux[:len(reconstructed)] - reconstructed) ** 2, axis=1)
    print(f"  Mean reconstruction error: {recon_error.mean():.6f}")

    # Save
    out_path = Path(args.output)
    np.savez(
        out_path,
        wavelength=np.array(wavelength),
        latents=latents,
        means=means,
        logvars=logvars,
        reconstructed=reconstructed,
        recon_error=recon_error,
    )
    print(f"\n  Saved: {out_path}")
    print(f"  Keys: wavelength {wavelength.shape}, latents {latents.shape}, "
          f"reconstructed {reconstructed.shape}")


if __name__ == "__main__":
    main()
