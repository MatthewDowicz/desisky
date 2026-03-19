# LDM Training for DESI Sky Spectra

This guide explains how to train a Latent Diffusion Model (LDM) on DESI sky spectra using the `desisky` package.

## Overview

The LDM generates realistic night-sky emission spectra by learning the distribution of VAE latent vectors conditioned on observational metadata. It operates in the compressed 8-dimensional latent space rather than the full 7781-dimensional spectral space, which is more computationally efficient and produces higher quality results.

Training uses the **EDM framework** (Karras et al. 2022):
- Log-normal noise distribution for continuous sigma sampling
- Preconditioned denoiser: `D(x; sigma) = c_skip * x + c_out * F_theta(c_in * x; c_noise)`
- EDM-weighted loss: `lambda(sigma) * ||D(x + sigma*n; sigma) - x||^2`
- Exponential Moving Average (EMA) of model weights for stable sampling

The trained LDM, combined with the pre-trained VAE decoder, can generate full sky spectra conditioned on observational parameters like sun altitude, moon position, atmospheric transparency, and more.

## Prerequisites

Before training the LDM, you need:

1. **A trained VAE** — to encode spectra into latent vectors. Use the pre-trained VAE or train your own (see [VAE_TRAINING.md](VAE_TRAINING.md))
2. **DESI sky spectra** — downloaded via `SkySpecVAC`
3. **GPU recommended** — LDM training benefits significantly from CUDA acceleration

```bash
pip install desisky[cuda12,data,viz]
```

## Quick Start

```python
import jax
import jax.random as jr
import jax.numpy as jnp
import numpy as np
from torch.utils.data import TensorDataset, random_split
import torch

from desisky.data import SkySpecVAC
from desisky.io import load_model
from desisky.models.ldm import make_UNet1D_cond, compute_sigma_data
from desisky.training import (
    LatentDiffusionTrainer,
    LDMTrainingConfig,
    NumpyLoader,
    fit_conditioning_scaler,
    normalize_conditioning,
)

# 1. Load data and encode to latent space
vac = SkySpecVAC(version="v1.0", download=True)
wave, flux, meta = vac.load_dark_time()
flux = flux.astype(np.float32)

vae, _ = load_model("vae")
latents = np.array(jax.vmap(vae.encode)(flux)[0])  # Use means, shape: (N, 8)

# 2. Extract conditioning features
conditioning_features = [
    "OBSALT", "TRANSPARENCY_GFA", "SUNALT", "SOLFLUX",
    "ECLLON", "ECLLAT", "GALLON", "GALLAT",
]
conditioning = meta[conditioning_features].values.astype(np.float32)

# 3. Train/val split
n = len(latents)
train_size = int(0.9 * n)
val_size = n - train_size

dataset = TensorDataset(
    torch.from_numpy(latents),
    torch.from_numpy(conditioning),
)
train_set, val_set = random_split(
    dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42),
)

train_loader = NumpyLoader(train_set, batch_size=64, shuffle=True)
val_loader = NumpyLoader(val_set, batch_size=64, shuffle=False)

# 4. Fit conditioning scaler on training data only
cond_train = np.array([conditioning[i] for i in train_set.indices])
scaler = fit_conditioning_scaler(cond_train, conditioning_features)

# 5. Compute sigma_data from training latents
latent_train = np.array([latents[i] for i in train_set.indices])
sigma_data = float(compute_sigma_data(jnp.array(latent_train)))

# 6. Initialize model
ldm = make_UNet1D_cond(
    latent_dim=8,
    meta_dim=len(conditioning_features),
    key=jr.PRNGKey(42),
)

# 7. Configure and train
config = LDMTrainingConfig(
    epochs=200,
    learning_rate=1e-4,
    meta_dim=len(conditioning_features),
    sigma_data=sigma_data,
    dropout_p=0.1,               # CFG conditioning dropout
    ema_decay=0.9999,
    early_stop_on_ema=True,
    conditioning_scaler=scaler,  # Saved in checkpoint for inference
    run_name="ldm_dark",
)

trainer = LatentDiffusionTrainer(ldm, config)
model, ema_model, history = trainer.train(train_loader, val_loader)

print(f"Best val loss: {history.best_val_loss:.6f} (epoch {history.best_epoch})")
```

## EDM Framework

### Noise Schedule

Training samples noise levels from a log-normal distribution:

```
ln(sigma) ~ N(P_mean, P_std^2)
```

Default values (`P_mean = -1.2`, `P_std = 1.2`) cover the range from near-zero noise to heavy corruption.

### Preconditioned Denoiser

The raw network output `F_theta` is wrapped in preconditioning functions that improve training stability:

```
D(x; sigma) = c_skip(sigma) * x + c_out(sigma) * F_theta(c_in(sigma) * x; c_noise(sigma))
```

Where:
- `c_skip` — skip connection weight (high at low noise, passes input through)
- `c_out` — output scaling (low at low noise, prevents large gradients)
- `c_in` — input scaling (normalizes noisy input)
- `c_noise` — noise level embedding (log-transformed)

### Loss Weighting

The EDM loss is weighted by `lambda(sigma) = (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2`, which gives roughly equal importance to all noise levels.

### sigma_data

`sigma_data` is the standard deviation of the training latent vectors. It's used by the preconditioning functions to properly scale the network. Compute it from training data:

```python
from desisky.models.ldm import compute_sigma_data
sigma_data = compute_sigma_data(training_latents)
```

## Conditioning Normalization

Raw conditioning features (e.g., SUNALT in degrees, TRANSPARENCY_GFA as a fraction) have vastly different scales. The conditioning scaler normalizes them to zero-mean, unit-variance:

```python
from desisky.training import fit_conditioning_scaler, normalize_conditioning

# Fit on training data only (to avoid data leakage)
scaler = fit_conditioning_scaler(cond_train, conditioning_features)
# Returns: {"mean": [...], "scale": [...], "columns": [...]}

# Normalize
cond_train_norm = normalize_conditioning(cond_train, scaler)
cond_val_norm = normalize_conditioning(cond_val, scaler)
```

The scaler is stored in the checkpoint metadata when you pass it to `LDMTrainingConfig.conditioning_scaler`. At inference time, `LatentDiffusionSampler` reads it from the checkpoint and auto-normalizes raw conditioning inputs.

## Classifier-Free Guidance (CFG)

CFG training drops conditioning information with probability `dropout_p` (default 0.1), replacing it with zeros. At inference time, the guided prediction is:

```
D_guided = D_uncond + guidance_scale * (D_cond - D_uncond)
```

Higher `guidance_scale` (typically 1.5-3.0) makes the model follow conditioning more strongly.

## Exponential Moving Average (EMA)

The EMA model is a smoothed copy of training weights:

```
theta_ema = decay * theta_ema + (1 - decay) * theta
```

With `decay = 0.9999`, the EMA model averages over ~10,000 gradient steps. The EMA model typically produces higher quality samples and is the model used for inference.

When `early_stop_on_ema=True` (default), validation loss is computed on the EMA model and used for early stopping / best-model checkpointing.

## Training Configuration

```python
config = LDMTrainingConfig(
    # Required
    epochs=200,                  # Training epochs
    learning_rate=1e-4,          # Adam learning rate
    meta_dim=8,                  # Number of conditioning features
    sigma_data=0.85,             # Training data std (from compute_sigma_data)

    # CFG / noise
    dropout_p=0.1,               # Conditioning dropout for CFG training
    p_mean=-1.2,                 # Log-normal noise mean (EDM default)
    p_std=1.2,                   # Log-normal noise std (EDM default)

    # EMA
    ema_decay=0.9999,            # EMA smoothing factor (0.0 to disable)
    early_stop_on_ema=True,      # Gate early stopping on EMA val loss

    # Checkpointing
    save_best=True,
    run_name="ldm_dark",
    save_dir=None,               # Default: ~/.cache/desisky/saved_models/ldm

    # Logging
    print_every=50,
    validate_every=1,
    random_seed=42,

    # Metadata (stored in checkpoint)
    conditioning_scaler=scaler,  # For auto-normalization at inference
    val_expids=[...],            # Validation exposure IDs for reproducibility
)
```

## Training History

```python
history = LDMTrainingHistory(
    train_losses=[...],       # EDM training loss per epoch
    val_losses=[...],         # EDM val loss per epoch (base model)
    ema_val_losses=[...],     # EDM val loss per epoch (EMA model)
    best_val_loss=0.042,      # Best validation loss
    best_epoch=156,           # Epoch of best validation loss
)
```

## W&B Experiment Tracking

Add Weights & Biases logging:

```python
from desisky.training import WandbConfig

wandb_config = WandbConfig(project="desisky-ldm", tags=["dark-time"])

trainer = LatentDiffusionTrainer(ldm, config, wandb_config=wandb_config)
model, ema_model, history = trainer.train(train_loader, val_loader)
```

This logs training and validation losses (base + EMA) every epoch.

### Custom Visualization with on_epoch_end

Use the callback to log CDF comparisons, latent corner plots, or conditional validation grids:

```python
from desisky.data import measure_airglow_intensities, compute_broadband_mags
from desisky.visualization import (
    plot_latent_corner_comparison,
    plot_broadband_cdfs,
    plot_airglow_cdfs,
    plot_conditional_validation_grid,
)
from desisky.training import log_figure

def on_epoch_end(model, history, epoch):
    # Generate latent samples from EMA model
    generated_latents = sample_from_model(ema_model, conditioning, ...)

    # Corner plot: real vs generated latent distributions
    fig = plot_latent_corner_comparison(real_latents, generated_latents)
    log_figure("viz/latents", fig, epoch)

    # Decode to spectra and compute physical features
    gen_spectra = jax.vmap(vae.decode)(generated_latents)
    gen_mags = compute_broadband_mags(wavelength, gen_spectra)

    # Broadband CDF comparison
    fig = plot_broadband_cdfs(real_mags, gen_mags)
    log_figure("viz/broadband_cdfs", fig, epoch)

trainer = LatentDiffusionTrainer(
    ldm, config,
    wandb_config=wandb_config,
    on_epoch_end=on_epoch_end,
)
```

### Hyperparameter Sweeps

```python
import wandb

sweep_config = {
    "method": "bayes",
    "metric": {"name": "val/edm_loss_ema", "goal": "minimize"},
    "parameters": {
        "learning_rate": {"min": 1e-5, "max": 1e-3},
        "dropout_p": {"values": [0.05, 0.1, 0.15]},
        "ema_decay": {"values": [0.999, 0.9995, 0.9999]},
    },
}

sweep_id = wandb.sweep(sweep_config, project="desisky-ldm")

def train_sweep():
    config = LDMTrainingConfig(
        epochs=100,
        learning_rate=wandb.config.learning_rate,
        meta_dim=8,
        sigma_data=sigma_data,
        dropout_p=wandb.config.dropout_p,
        ema_decay=wandb.config.ema_decay,
        conditioning_scaler=scaler,
    )
    trainer = LatentDiffusionTrainer(
        ldm, config,
        wandb_config=WandbConfig(project="desisky-ldm"),
    )
    trainer.train(train_loader, val_loader)

wandb.agent(sweep_id, function=train_sweep, count=20)
```

See `examples/08_ldm_wandb_training.ipynb` for a complete working example.

## Using the Trained Model

### Sampling

```python
from desisky.inference import LatentDiffusionSampler

sampler = LatentDiffusionSampler(
    ldm=ema_model,           # Use EMA model for best quality
    vae_model=vae,
    sigma_data=sigma_data,
    conditioning_scaler=scaler,
    num_steps=100,
)

# Raw conditioning values — auto-normalized by the scaler
conditioning = jnp.array([
    [2100.0, 0.9, -30.0, 150.0, 45.0, 10.0, 120.0, 5.0],
])

generated = sampler.sample(
    key=jr.PRNGKey(42),
    conditioning=conditioning,
    guidance_scale=2.0,
)
# generated shape: (1, 7781)
```

### Loading from Checkpoint

```python
from desisky.io import load_model

ldm, meta = load_model("ldm_dark", path="path/to/checkpoint.eqx")

# Checkpoint metadata includes everything needed for inference
sigma_data = meta["training"]["sigma_data"]
scaler = meta["training"]["conditioning_scaler"]
```

## Model Variants

| Variant | Subset | Conditioning Features | meta_dim |
|---------|--------|----------------------|----------|
| `ldm_dark` | Dark-time | OBSALT, TRANSPARENCY_GFA, SUNALT, SOLFLUX, ECLLON, ECLLAT, GALLON, GALLAT | 8 |
| `ldm_moon` | Moon-contaminated | OBSALT, TRANSPARENCY_GFA, SUNALT, MOONALT, MOONSEP, MOONFRAC | 6 |
| `ldm_twilight` | Twilight | OBSALT, TRANSPARENCY_GFA, SUNALT, SUNSEP | 4 |

Each variant is trained on the corresponding data subset with its own conditioning scaler.

## Troubleshooting

**Loss not decreasing**
- Verify `sigma_data` is computed from the actual training latents
- Check that conditioning is properly normalized
- Try reducing learning rate

**Generated spectra look unrealistic**
- Increase `num_steps` in the sampler (100-200 is typical)
- Adjust `guidance_scale` (try 1.5-3.0)
- Ensure you're using the EMA model, not the base model
- Train for more epochs

**EMA val loss diverges from base val loss**
- This is normal — the EMA model lags behind the base model
- EMA val loss should eventually be lower and more stable
- If EMA loss is consistently higher, try lower `ema_decay` (e.g., 0.999)

**Out of memory**
- Reduce batch size
- Reduce model size (fewer channels/blocks)

## References

- **EDM Paper**: Karras et al., "Elucidating the Design Space of Diffusion-Based Generative Models" (NeurIPS 2022)
- **CFG Paper**: Ho & Salimans, "Classifier-Free Diffusion Guidance" (NeurIPS Workshop 2022)
- **Latent Diffusion**: Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models" (CVPR 2022)

## API Reference

- `desisky.training.LatentDiffusionTrainer` — Main training class
- `desisky.training.LDMTrainingConfig` — Training configuration
- `desisky.training.LDMTrainingHistory` — Training history
- `desisky.training.fit_conditioning_scaler` — Fit normalization scaler
- `desisky.training.normalize_conditioning` — Apply normalization
- `desisky.models.ldm.UNet1D_cond` — 1D U-Net architecture
- `desisky.models.ldm.make_UNet1D_cond` — Model constructor
- `desisky.models.ldm.edm_denoiser` — EDM preconditioned denoiser
- `desisky.models.ldm.compute_sigma_data` — Compute training data std
- `desisky.inference.LatentDiffusionSampler` — EDM Heun sampler
