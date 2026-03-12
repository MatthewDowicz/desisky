#!/usr/bin/env python
"""
Benchmark script for desisky inference and training timings.

Reports two timings for each operation:
  - First eval: includes JAX JIT compilation overhead
  - Avg eval:   average over N repeated runs (post-JIT)

Usage
-----
    python benchmarks/timing.py --all                  # everything
    python benchmarks/timing.py --inference             # inference only
    python benchmarks/timing.py --training              # training only
    python benchmarks/timing.py --inference --models ldm_dark vae  # subset

Results are printed as a markdown table and optionally saved to a JSON file.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

@dataclass
class TimingResult:
    operation: str
    first_ms: float
    avg_ms: float
    n_repeats: int


def time_fn(fn, n_repeats: int = 10, warmup: bool = True) -> tuple[float, float]:
    """
    Time a function, returning (first_call_ms, avg_ms_over_repeats).

    The first call includes JIT compilation. Subsequent calls measure
    post-JIT performance. All calls use block_until_ready() to ensure
    GPU work is complete.
    """
    # First call (includes JIT)
    t0 = time.perf_counter()
    result = fn()
    if hasattr(result, "block_until_ready"):
        result.block_until_ready()
    elif isinstance(result, tuple) and hasattr(result[0], "block_until_ready"):
        result[0].block_until_ready()
    first_ms = (time.perf_counter() - t0) * 1000

    if not warmup:
        return first_ms, first_ms

    # Repeated calls (post-JIT)
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        result = fn()
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()
        elif isinstance(result, tuple) and hasattr(result[0], "block_until_ready"):
            result[0].block_until_ready()
        times.append((time.perf_counter() - t0) * 1000)

    avg_ms = sum(times) / len(times)
    return first_ms, avg_ms


# ---------------------------------------------------------------------------
# Inference benchmarks
# ---------------------------------------------------------------------------

def bench_broadband(n_rows: int = 1000, n_repeats: int = 100) -> TimingResult:
    """Benchmark broadband MLP prediction."""
    from desisky.io import load_builtin

    model, meta = load_builtin("broadband")
    key = jr.PRNGKey(0)
    # Random conditioning inputs: shape (n_rows, 6)
    conditions = jr.normal(key, (n_rows, 6))

    predict_batch = jax.jit(jax.vmap(model))

    first_ms, avg_ms = time_fn(lambda: predict_batch(conditions), n_repeats=n_repeats)
    return TimingResult(f"Broadband MLP ({n_rows} rows)", first_ms, avg_ms, n_repeats)


def bench_vae(n_spectra: int = 1000, n_repeats: int = 100) -> TimingResult:
    """Benchmark VAE encode + decode."""
    from desisky.io import load_builtin

    vae, meta = load_builtin("vae")
    key = jr.PRNGKey(0)
    spectra = jr.normal(key, (n_spectra, 7781))

    @jax.jit
    def encode_decode(spectra):
        means, logvars = jax.vmap(vae.encode)(spectra)
        reconstructed = jax.vmap(vae.decode)(means)
        return reconstructed

    first_ms, avg_ms = time_fn(lambda: encode_decode(spectra), n_repeats=n_repeats)
    return TimingResult(f"VAE encode+decode ({n_spectra})", first_ms, avg_ms, n_repeats)


def _make_sampler(variant: str):
    """Load models and create a LatentDiffusionSampler for the given variant."""
    from desisky.inference import LatentDiffusionSampler
    from desisky.io import load_builtin

    ldm, ldm_meta = load_builtin(f"ldm_{variant}")
    vae, _ = load_builtin("vae")

    sampler = LatentDiffusionSampler(
        ldm, vae,
        sigma_data=ldm_meta["training"]["sigma_data"],
        conditioning_scaler=ldm_meta["training"].get("conditioning_scaler"),
    )
    return sampler, ldm_meta


# Conditioning dimensions per variant
_COND_DIMS = {"dark": 8, "moon": 6, "twilight": 4}


def bench_ldm(
    variant: str = "dark",
    n_samples: int = 100,
    stochastic: bool = False,
    n_repeats: int = 10,
) -> TimingResult:
    """Benchmark LDM sampling (deterministic or stochastic)."""
    sampler, ldm_meta = _make_sampler(variant)
    cond_dim = _COND_DIMS[variant]

    key = jr.PRNGKey(42)
    # Use random conditioning (already auto-normalized by sampler)
    conditioning = jr.normal(key, (n_samples, cond_dim))

    mode = "stochastic" if stochastic else "deterministic"

    def run():
        return sampler.sample(
            key=jr.PRNGKey(0),
            conditioning=conditioning,
            guidance_scale=2.0,
            stochastic=stochastic,
        )

    first_ms, avg_ms = time_fn(run, n_repeats=n_repeats)
    return TimingResult(
        f"LDM {variant} ({n_samples}, {mode})", first_ms, avg_ms, n_repeats
    )


# ---------------------------------------------------------------------------
# Training benchmarks
# ---------------------------------------------------------------------------

def bench_train_broadband(n_epochs: int = 10, n_repeats: int = 3) -> TimingResult:
    """Benchmark broadband training for N epochs on synthetic data."""
    from desisky.models.broadband import make_broadbandMLP
    from desisky.training.losses import loss_func


    key = jr.PRNGKey(0)
    model = make_broadbandMLP(in_size=6, out_size=4, width_size=128, depth=5, key=key)
    optimizer = optax.adam(1e-4)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # Synthetic data
    n_train = 2000
    X = jr.normal(jr.PRNGKey(1), (n_train, 6))
    Y = jr.normal(jr.PRNGKey(2), (n_train, 4))
    batch_size = 256

    @eqx.filter_jit
    def train_step(model, opt_state, x_batch, y_batch):
        def loss_fn(m):
            preds = jax.vmap(m)(x_batch)
            return jnp.mean((preds - y_batch) ** 2)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state_new = optimizer.update(grads, opt_state, model)
        model_new = eqx.apply_updates(model, updates)
        return model_new, opt_state_new, loss

    def run_epochs():
        m, os_ = model, opt_state
        n_batches = n_train // batch_size
        for epoch in range(n_epochs):
            for i in range(n_batches):
                x_b = X[i * batch_size:(i + 1) * batch_size]
                y_b = Y[i * batch_size:(i + 1) * batch_size]
                m, os_, loss = train_step(m, os_, x_b, y_b)
        # Block until complete
        loss.block_until_ready()
        return loss

    first_ms, avg_ms = time_fn(run_epochs, n_repeats=n_repeats)
    return TimingResult(
        f"Broadband train ({n_epochs} epochs)", first_ms, avg_ms, n_repeats
    )


def bench_train_vae(n_epochs: int = 10, n_repeats: int = 3) -> TimingResult:
    """Benchmark VAE training for N epochs on synthetic data."""

    from desisky.models.vae import make_SkyVAE

    key = jr.PRNGKey(0)
    model = make_SkyVAE(in_channels=7781, latent_dim=8, key=key)
    optimizer = optax.adam(1e-4)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # Synthetic data
    n_train = 500
    spectra = jr.normal(jr.PRNGKey(1), (n_train, 7781))
    batch_size = 64

    @eqx.filter_jit
    def train_step(model, opt_state, batch, rng_key):
        def loss_fn(m):
            keys = jr.split(rng_key, batch.shape[0])
            results = jax.vmap(m)(batch, keys)
            recon_loss = jnp.mean((results["output"] - batch) ** 2)
            kl_loss = -0.5 * jnp.mean(1 + results["logvar"] - results["mean"] ** 2 - jnp.exp(results["logvar"]))
            return recon_loss + 1e-3 * kl_loss
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state_new = optimizer.update(grads, opt_state, model)
        model_new = eqx.apply_updates(model, updates)
        return model_new, opt_state_new, loss

    def run_epochs():
        m, os_ = model, opt_state
        n_batches = n_train // batch_size
        rng = jr.PRNGKey(42)
        for epoch in range(n_epochs):
            for i in range(n_batches):
                rng, step_key = jr.split(rng)
                batch = spectra[i * batch_size:(i + 1) * batch_size]
                m, os_, loss = train_step(m, os_, batch, step_key)
        loss.block_until_ready()
        return loss

    first_ms, avg_ms = time_fn(run_epochs, n_repeats=n_repeats)
    return TimingResult(
        f"VAE train ({n_epochs} epochs)", first_ms, avg_ms, n_repeats
    )


def bench_train_ldm(n_epochs: int = 10, n_repeats: int = 3) -> TimingResult:
    """Benchmark LDM training for N epochs on synthetic latent data."""

    from desisky.models.ldm import make_UNet1D_cond, edm_denoiser, compute_sigma_data

    key = jr.PRNGKey(0)
    meta_dim = 8  # dark variant
    model = make_UNet1D_cond(
        in_ch=1, out_ch=1, meta_dim=meta_dim,
        hidden=32, levels=3, emb_dim=32, key=key,
    )
    optimizer = optax.adam(1e-4)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # Synthetic latent data
    n_train = 500
    latents = jr.normal(jr.PRNGKey(1), (n_train, 1, 8))
    conditioning = jr.normal(jr.PRNGKey(2), (n_train, meta_dim))
    sigma_data = float(jnp.std(latents))
    batch_size = 64

    # EDM training constants
    P_mean = -1.2
    P_std = 1.2

    @eqx.filter_jit
    def train_step(model, opt_state, x_batch, cond_batch, rng_key):
        def loss_fn(m):
            bsz = x_batch.shape[0]
            # Sample sigma from log-normal
            rng1, rng2 = jr.split(rng_key)
            log_sigma = P_mean + P_std * jr.normal(rng1, (bsz,))
            sigma = jnp.exp(log_sigma)
            # Add noise
            noise = jr.normal(rng2, x_batch.shape)
            x_noisy = x_batch + sigma[:, None, None] * noise
            # EDM loss
            weight = (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2
            D_x = jax.vmap(edm_denoiser, in_axes=(None, 0, 0, 0, None, None, None))(
                m, x_noisy, sigma, cond_batch, sigma_data, None, 0.0
            )
            return jnp.mean(weight[:, None, None] * (D_x - x_batch) ** 2)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state_new = optimizer.update(grads, opt_state, model)
        model_new = eqx.apply_updates(model, updates)
        return model_new, opt_state_new, loss

    def run_epochs():
        m, os_ = model, opt_state
        n_batches = n_train // batch_size
        rng = jr.PRNGKey(99)
        for epoch in range(n_epochs):
            for i in range(n_batches):
                rng, step_key = jr.split(rng)
                x_b = latents[i * batch_size:(i + 1) * batch_size]
                c_b = conditioning[i * batch_size:(i + 1) * batch_size]
                m, os_, loss = train_step(m, os_, x_b, c_b, step_key)
        loss.block_until_ready()
        return loss

    first_ms, avg_ms = time_fn(run_epochs, n_repeats=n_repeats)
    return TimingResult(
        f"LDM dark train ({n_epochs} epochs)", first_ms, avg_ms, n_repeats
    )


# ---------------------------------------------------------------------------
# Registry and runner
# ---------------------------------------------------------------------------

INFERENCE_BENCHMARKS = {
    "broadband": bench_broadband,
    "vae": bench_vae,
    "ldm_dark_det": lambda: bench_ldm("dark", stochastic=False),
    "ldm_dark_sto": lambda: bench_ldm("dark", stochastic=True),
    "ldm_moon": lambda: bench_ldm("moon", stochastic=False),
    "ldm_twilight": lambda: bench_ldm("twilight", stochastic=False),
}

TRAINING_BENCHMARKS = {
    "broadband": bench_train_broadband,
    "vae": bench_train_vae,
    "ldm_dark": bench_train_ldm,
}


def get_platform_info() -> dict:
    """Collect platform information."""
    devices = jax.devices()
    device = devices[0]
    return {
        "platform": device.platform,  # "cpu" or "gpu"
        "device_kind": getattr(device, "device_kind", str(device)),
        "hostname": platform.node(),
        "python_version": platform.python_version(),
        "jax_version": jax.__version__,
        "n_devices": len(devices),
    }


def print_results(results: list[TimingResult], platform_info: dict):
    """Print results as a formatted markdown table."""
    print()
    print(f"## Platform: {platform_info['hostname']} "
          f"({platform_info['device_kind']}, {platform_info['platform'].upper()})")
    print(f"JAX {platform_info['jax_version']} | "
          f"Python {platform_info['python_version']} | "
          f"{platform_info['n_devices']} device(s)")
    print()
    print(f"| {'Operation':<40} | {'First (ms)':>12} | {'Avg (ms)':>12} | {'N':>4} |")
    print(f"|{'-' * 42}|{'-' * 14}|{'-' * 14}|{'-' * 6}|")
    for r in results:
        print(f"| {r.operation:<40} | {r.first_ms:>12.2f} | {r.avg_ms:>12.2f} | {r.n_repeats:>4} |")
    print()


def main():
    parser = argparse.ArgumentParser(description="Benchmark desisky operations")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="Run all benchmarks")
    group.add_argument("--inference", action="store_true", help="Run inference benchmarks only")
    group.add_argument("--training", action="store_true", help="Run training benchmarks only")
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Specific benchmarks to run (e.g., broadband vae ldm_dark_det)",
    )
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON file")
    parser.add_argument("--n-samples", type=int, default=100, help="Number of LDM samples (default: 100)")
    parser.add_argument("--n-rows", type=int, default=1000, help="Number of broadband rows (default: 1000)")
    parser.add_argument("--n-spectra", type=int, default=1000, help="Number of VAE spectra (default: 1000)")
    args = parser.parse_args()

    # Default to --all if nothing specified
    if not args.all and not args.inference and not args.training:
        args.all = True

    platform_info = get_platform_info()
    print(f"Platform: {platform_info['hostname']} ({platform_info['device_kind']})")
    print(f"JAX backend: {platform_info['platform'].upper()}")
    print()

    results = []

    if args.all or args.inference:
        print("=" * 60)
        print("INFERENCE BENCHMARKS")
        print("=" * 60)
        benchmarks = INFERENCE_BENCHMARKS
        if args.models:
            benchmarks = {k: v for k, v in benchmarks.items() if k in args.models}

        for name, fn in benchmarks.items():
            print(f"  Running {name}...", end=" ", flush=True)
            # Override defaults for configurable benchmarks
            if name == "broadband":
                r = bench_broadband(n_rows=args.n_rows)
            elif name == "vae":
                r = bench_vae(n_spectra=args.n_spectra)
            elif "ldm" in name:
                variant = name.replace("ldm_", "").replace("_det", "").replace("_sto", "")
                stochastic = name.endswith("_sto")
                r = bench_ldm(variant, n_samples=args.n_samples, stochastic=stochastic)
            else:
                r = fn()
            print(f"first={r.first_ms:.1f}ms, avg={r.avg_ms:.1f}ms")
            results.append(r)

    if args.all or args.training:
        print()
        print("=" * 60)
        print("TRAINING BENCHMARKS")
        print("=" * 60)
        benchmarks = TRAINING_BENCHMARKS
        if args.models:
            benchmarks = {k: v for k, v in benchmarks.items() if k in args.models}

        for name, fn in benchmarks.items():
            print(f"  Running {name}...", end=" ", flush=True)
            r = fn()
            print(f"first={r.first_ms:.1f}ms, avg={r.avg_ms:.1f}ms")
            results.append(r)

    print_results(results, platform_info)

    if args.output:
        output = {
            "platform": platform_info,
            "results": [asdict(r) for r in results],
        }
        Path(args.output).write_text(json.dumps(output, indent=2))
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
