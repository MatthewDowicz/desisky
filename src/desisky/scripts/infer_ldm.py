#!/usr/bin/env python
# SPDX-FileCopyrightText: 2025-present MatthewDowicz <mjdowicz@gmail.com>
#
# SPDX-License-Identifier: MIT

"""
LDM Spectral Generation
=========================

Generate sky spectra using a trained Latent Diffusion Model.
Conditioning can come from real DESI data, a user file, or a JSON string.

Usage:
    desisky-infer-ldm --variant dark --n-samples 100
    desisky-infer-ldm --variant moon --guidance-scale 2.0 --n-samples 500
    desisky-infer-ldm --cond-path my_conditions.csv --n-samples 50
    desisky-infer-ldm --conditioning '[[60,0.9,-30,150,45,10,120,5]]'
"""

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from desisky.io import load_or_builtin
from desisky.inference import LatentDiffusionSampler

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate sky spectra using a trained LDM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--variant", choices=["dark", "moon", "twilight"], default="dark")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to custom LDM checkpoint (default: builtin)")
    parser.add_argument("--vae-path", type=str, default=None,
                        help="Path to custom VAE checkpoint (default: builtin)")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--guidance-scale", type=float, default=1.0,
                        help="CFG guidance scale (1.0=conditional, >1=amplified)")
    parser.add_argument("--num-steps", type=int, default=250,
                        help="Heun solver steps (more=higher quality)")
    parser.add_argument("--output", type=str, default="ldm_generated.npz")
    parser.add_argument("--seed", type=int, default=42)

    # Conditioning source (mutually exclusive)
    cond_group = parser.add_mutually_exclusive_group()
    cond_group.add_argument("--from-data", action="store_true", default=True,
                            help="Sample conditioning from real DESI data (default)")
    cond_group.add_argument("--cond-path", type=str, default=None,
                            help="Path to .npz or .csv with conditioning vectors")
    cond_group.add_argument("--conditioning", type=str, default=None,
                            help="Manual conditioning as JSON array, e.g. "
                                 "'[[60,0.9,-30,150,45,10,120,5]]'")
    return parser.parse_args()


def load_conditioning(args, variant, meta_dim):
    """Load conditioning from the specified source."""
    cond_cols = CONDITIONING_COLS[variant]

    if args.conditioning is not None:
        # Manual JSON input
        cond = np.array(json.loads(args.conditioning), dtype=np.float32)
        if cond.ndim == 1:
            cond = cond[None, :]
        if cond.shape[1] != meta_dim:
            sys.exit(f"Conditioning dim {cond.shape[1]} != expected {meta_dim}")
        # Repeat to match n_samples if only one row provided
        if len(cond) == 1 and args.n_samples > 1:
            cond = np.tile(cond, (args.n_samples, 1))
        return cond

    if args.cond_path is not None:
        p = Path(args.cond_path)
        ext = p.suffix.lower()
        if ext == ".npz":
            data = np.load(p)
            return data["conditioning"].astype(np.float32)
        elif ext == ".csv":
            import pandas as pd
            df = pd.read_csv(p)
            return df[cond_cols].to_numpy().astype(np.float32)
        else:
            sys.exit(f"Unsupported conditioning format '{ext}'. Use .npz or .csv")

    # Default: sample from real DESI data
    from desisky.data import SkySpecVAC
    vac = SkySpecVAC(version="v1.0", download=True)
    # Map CLI variant names to SkySpecVAC method names
    loader_names = {
        "dark": "load_dark_time",
        "moon": "load_moon_contaminated",
        "twilight": "load_sun_contaminated",
    }
    loader = getattr(vac, loader_names[variant])
    _, _, metadata = loader()

    cond = metadata[cond_cols].to_numpy().astype(np.float32)

    # Remove rows with NaN/Inf in conditioning columns
    finite_mask = np.isfinite(cond).all(axis=1)
    if (~finite_mask).sum() > 0:
        cond = cond[finite_mask]

    # Randomly sample n_samples rows
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(cond), size=min(args.n_samples, len(cond)), replace=False)
    return cond[idx]


def main():
    args = parse_args()
    variant = args.variant
    cond_cols = CONDITIONING_COLS[variant]
    meta_dim = len(cond_cols)

    print("=" * 60)
    print(f"LDM Spectral Generation — {variant}")
    print("=" * 60)

    # [1/4] Load Models
    print("\n[1/4] Loading models...")
    ldm_kind = f"ldm_{variant}"
    ldm, ldm_meta = load_or_builtin(ldm_kind, path=args.model_path)
    vae, _ = load_or_builtin("vae", path=args.vae_path)
    ldm_source = args.model_path or f"builtin ({ldm_kind})"
    vae_source = args.vae_path or "builtin"
    print(f"  LDM: {ldm_source}")
    print(f"  VAE: {vae_source} (latent_dim={vae.latent_dim})")

    # Extract training metadata
    sigma_data = ldm_meta["training"]["sigma_data"]
    conditioning_scaler = ldm_meta["training"].get("conditioning_scaler")

    # [2/4] Load Conditioning
    print(f"\n[2/4] Loading conditioning (meta_dim={meta_dim})...")
    conditioning = load_conditioning(args, variant, meta_dim)
    n_samples = len(conditioning)
    print(f"  {n_samples} conditioning vectors loaded")

    # [3/4] Generate Spectra
    print(f"\n[3/4] Generating {n_samples} spectra ({args.num_steps} Heun steps)...")
    sampler = LatentDiffusionSampler(
        ldm_model=ldm,
        vae_model=vae,
        sigma_data=sigma_data,
        conditioning_scaler=conditioning_scaler,
        num_steps=args.num_steps,
    )

    cond_jax = jnp.array(conditioning)
    spectra, latents = sampler.sample(
        key=jr.PRNGKey(args.seed),
        conditioning=cond_jax,
        guidance_scale=args.guidance_scale,
        return_latents=True,
    )
    spectra = np.array(spectra)
    latents = np.array(latents)
    print(f"  Spectra shape: {spectra.shape}")

    # [4/4] Save Output
    print("\n[4/4] Saving output...")
    from desisky.data import SkySpecVAC
    vac = SkySpecVAC(version="v1.0", download=True)
    wavelength, _, _ = vac.load()

    out_path = Path(args.output)
    np.savez(
        out_path,
        spectra=spectra,
        latents=latents,
        conditioning=conditioning,
        wavelength=np.array(wavelength),
    )
    print(f"  Saved: {out_path}")
    print(f"  Keys: spectra {spectra.shape}, latents {latents.shape}, "
          f"conditioning {conditioning.shape}")


if __name__ == "__main__":
    main()
