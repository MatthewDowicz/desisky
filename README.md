# desisky

[![PyPI - Version](https://img.shields.io/pypi/v/desisky.svg)](https://pypi.org/project/desisky)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/desisky.svg)](https://pypi.org/project/desisky)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/MatthewDowicz/desisky/actions/workflows/tests.yml/badge.svg)](https://github.com/MatthewDowicz/desisky/actions/workflows/tests.yml)

-----

## About

`desisky` provides machine learning models and tools for DESI sky modeling:

1. **Predictive broadband model** — Predicts surface brightness in V, g, r, and z photometric bands from observational metadata (moon position, transparency, eclipse fraction)
2. **Variational Autoencoder (VAE)** — Compresses sky spectra (7,781 wavelength bins → 8-dimensional latent space) for analysis, anomaly detection, and dimensionality reduction. Trained with InfoVAE-MMD objective
3. **Latent Diffusion Models (LDM)** — Generates realistic night-sky emission spectra using EDM preconditioning (Karras et al. 2022), conditioned on observational parameters:
   - **LDM Dark** — Dark-time spectra conditioned on 8 features: sun position, transparency, galactic/ecliptic coordinates, and solar flux
   - **LDM Moon** — Moon-contaminated spectra conditioned on 6 features: moon position, separation, and illumination fraction
   - **LDM Twilight** — Twilight spectra conditioned on 4 features: observation altitude, transparency, sun altitude, and sun separation
4. **Data utilities** — Download and load the DESI DR1 Sky Spectra Value-Added Catalog (VAC) with automatic SHA-256 integrity verification, subset filtering, and enrichment (V-band magnitudes, eclipse fractions, solar flux, coordinate transforms)
5. **Spectral analysis** — Measure airglow emission line intensities and compute broadband magnitudes directly from spectra
6. **CLI tools** — Train and run inference on all models from the command line with optional W&B experiment tracking
7. **Experiment tracking** — Optional Weights & Biases integration with visualization callbacks and hyperparameter sweeps

Built with **JAX/Equinox** for high-performance model inference and designed to integrate with SpecSim and survey forecasting workflows. This repository hosts the code and notebooks supporting the forthcoming paper by Dowicz et al. (20XX).

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data](#data)
  - [Data Subsets](#data-subsets)
  - [Data Enrichment](#data-enrichment)
  - [Spectral Analysis](#spectral-analysis)
  - [Data Download CLI](#data-download-cli)
- [Models](#models)
  - [Available Pre-trained Models](#available-pre-trained-models)
  - [Loading and Saving Models](#loading-and-saving-models)
- [CLI Tools](#cli-tools)
  - [CLI Training](#cli-training)
  - [CLI Inference](#cli-inference)
- [Training (Python API)](#training-python-api)
  - [VAE Training](#vae-training)
  - [LDM Training](#ldm-training)
- [Experiment Tracking (W&B)](#experiment-tracking-wb)
- [Visualization](#visualization)
- [Examples](#examples)
- [Project Structure](#project-structure)
- [Development](#development)
- [License](#license)

## Installation

```bash
# Default: inference-ready (CPU)
pip install desisky

# With data loading (FITS files, enrichment)
pip install desisky[data]

# GPU training + data + visualization
pip install desisky[cuda12,data,viz]

# Everything (CPU) including W&B experiment tracking
pip install desisky[all]

# Everything with GPU
pip install desisky[all,cuda12]
```

> **Note:** CUDA wheels may require manual installation. See the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) for details.

### Optional dependency groups

| Extra | Packages |
|-------|----------|
| `cuda12` | jax[cuda12], equinox, optax, torch, tqdm |
| `data` | fitsio, pandas, speclite, astropy |
| `viz` | matplotlib, umap-learn |
| `wandb` | wandb, matplotlib, pandas |
| `all` | All of the above (CPU JAX) |

Core dependencies (always installed): numpy, scipy, requests, jax, equinox

## Quick Start

### Download and load DESI sky spectra

```python
from desisky.data import SkySpecVAC

# Download DR1 VAC (~274 MB, with SHA-256 verification)
vac = SkySpecVAC(version="v1.0", download=True)

# Load wavelength, flux, and metadata
wavelength, flux, metadata = vac.load()
print(f"Wavelength: {wavelength.shape}")  # (7781,)
print(f"Flux: {flux.shape}")              # (9176, 7781)
print(f"Metadata columns: {list(metadata.columns)}")
# ['NIGHT', 'EXPID', 'TILEID', 'AIRMASS', 'EBV', 'MOONFRAC', 'MOONALT', ...]

# Enriched columns are added automatically for v1.0
print('SKY_MAG_V_SPEC' in metadata.columns)  # True
print('ECLIPSE_FRAC' in metadata.columns)    # True
```

### Predict sky brightness with broadband model

```python
import desisky
import jax.numpy as jnp

model, meta = desisky.io.load_model("broadband")

# Input: [MOONSEP, OBSALT, MOONALT, MOONFRAC, TRANSPARENCY_GFA, ECLIPSE_FRAC]
x = jnp.array([45.0, 80.0, 30.0, 0.8, 0.95, 0.0])

# Predict surface brightness in V, g, r, z bands
y = model(x)  # Shape: (4,)
print(f"Predicted magnitudes: {y}")
```

### Encode sky spectra with VAE

```python
from desisky.io import load_model
from desisky.data import SkySpecVAC
import jax
import jax.random as jr

vac = SkySpecVAC(version="v1.0", download=True)
wavelength, flux, metadata = vac.load()

vae, meta = load_model("vae")

# Encode a single spectrum to latent representation
mean, logvar = vae.encode(flux[0])
print(f"Latent mean: {mean}")  # Shape: (8,)

# Sample and decode
latent = vae.sample(mean, logvar, jr.PRNGKey(0))
reconstructed = vae.decode(latent)
print(f"Reconstructed shape: {reconstructed.shape}")  # (7781,)

# Batch encoding with vmap
batch_means, batch_logvars = jax.vmap(vae.encode)(flux)
print(f"Batch latents: {batch_means.shape}")  # (9176, 8)
```

### Generate sky spectra with Latent Diffusion Model

```python
from desisky.inference import LatentDiffusionSampler
import jax.random as jr
import jax.numpy as jnp

# Create sampler (auto-loads model + VAE from pretrained checkpoints)
sampler = LatentDiffusionSampler("ldm_dark")

# Conditioning: [OBSALT, transparency, SUNALT, SOLFLUX, ECLLON, ECLLAT, GALLON, GALLAT]
conditioning = jnp.array([
    [2100.0, 0.9, -30.0, 150.0, 45.0, 10.0, 120.0, 5.0],  # Dark sky
])

generated = sampler.sample(
    key=jr.PRNGKey(42),
    conditioning=conditioning,
    guidance_scale=2.0,
)
print(f"Generated spectrum shape: {generated.shape}")  # (1, 7781)
```

## Data

### Data subsets

The VAC provides subset methods for filtering observations by sky conditions:

**Dark time** (non-contaminated):
```python
wave, flux, meta = vac.load_dark_time()
# SUNALT < -20  |  MOONALT < -5  |  TRANSPARENCY_GFA > 0
```

**Twilight** (sun-contaminated):
```python
wave, flux, meta = vac.load_sun_contaminated()
# SUNALT > -20  |  MOONALT <= -5  |  SUNSEP <= 110  |  TRANSPARENCY_GFA > 0
```

**Moon-contaminated**:
```python
wave, flux, meta = vac.load_moon_contaminated()
# SUNALT < -20  |  MOONALT > 5  |  MOONFRAC > 0.5  |  MOONSEP <= 90  |  TRANSPARENCY_GFA > 0
```

All data is automatically enriched with computed columns on load (v1.0 only):

### Enriched columns

| Column | Description |
|--------|-------------|
| `SKY_MAG_V_SPEC` | V-band AB magnitude computed from the spectrum via speclite |
| `ECLIPSE_FRAC` | Lunar eclipse umbral coverage fraction (0-1) |
| `SOLFLUX` | Daily 10.7 cm solar flux (sfu) |
| `GALLON`, `GALLAT` | Galactic coordinates (from TILERA/TILEDEC) |
| `ECLLON`, `ECLLAT` | Ecliptic coordinates (from TILERA/TILEDEC) |

The individual enrichment functions are also available for standalone use:

```python
from desisky.data import (
    compute_vband_magnitudes,    # V-band magnitudes from spectra
    load_eclipse_catalog,        # NASA lunar eclipse catalog
    compute_eclipse_fraction,    # Eclipse umbral coverage
    load_solar_flux,             # F10.7 solar radio flux
    attach_solar_flux,           # Add SOLFLUX column to metadata
    add_galactic_coordinates,    # Add GALLON, GALLAT columns
    add_ecliptic_coordinates,    # Add ECLLON, ECLLAT columns
)
```

### Spectral analysis

Extract physical features from spectra (require `desisky[data]`):

```python
from desisky.data import measure_airglow_intensities, compute_broadband_mags

# Measure 10 airglow emission line intensities via continuum-subtracted integration
# Returns DataFrame: OI_5577, OI_6300, OI_6364, OH_1, OH_2, ..., OH_7
airglow = measure_airglow_intensities(wavelength, flux)

# Compute broadband magnitudes via speclite (V, g, r, z)
mags = compute_broadband_mags(wavelength, flux)  # Shape: (n_spectra, 4)
```

The airglow measurement follows the method of Noll et al. (2012), using two flanking continuum windows for background subtraction. Composite lines are also computed: OH (sum of all OH bands) and OI doublet (OI 6300 + OI 6364).

Related constants:
- `LINE_BANDS` — Dictionary of airglow line wavelength windows
- `AIRGLOW_CDF_NAMES` — Standard names for the 10 + 2 composite airglow features
- `BROADBAND_NAMES` — Standard names for the 4 broadband magnitudes (`["V", "g", "r", "z"]`)
- `FLUX_SCALE` — Default flux scaling factor (1e-17 erg/s/cm^2/A)

### Data download CLI

```bash
# Show default data directory
desisky-data dir

# Download DESI DR1 sky spectra VAC
desisky-data fetch --version v1.0

# Download to custom location
desisky-data fetch --root /path/to/data

# Skip checksum verification (not recommended)
desisky-data fetch --no-verify
```

Override the default data directory with an environment variable:

```bash
export DESISKY_DATA_DIR=/path/to/data
```

## Models

### Available pre-trained models

| Model | Architecture | Description |
|-------|-------------|-------------|
| `broadband` | MLP (6 → 128 × 5 → 4) | Predicts V, g, r, z magnitudes from observational metadata |
| `vae` | Encoder-Decoder (7781 → 8 → 7781) | Compresses sky spectra to 8D latent space |
| `ldm_dark` | 1D U-Net + EDM | Generates dark-time spectra (8 conditioning features) |
| `ldm_moon` | 1D U-Net + EDM | Generates moon-contaminated spectra (6 conditioning features) |
| `ldm_twilight` | 1D U-Net + EDM | Generates twilight spectra (4 conditioning features) |

**LDM conditioning features:**

- **`ldm_dark`**: `[OBSALT, TRANSPARENCY_GFA, SUNALT, SOLFLUX, ECLLON, ECLLAT, GALLON, GALLAT]`
- **`ldm_moon`**: `[OBSALT, TRANSPARENCY_GFA, SUNALT, MOONALT, MOONSEP, MOONFRAC]`
- **`ldm_twilight`**: `[OBSALT, TRANSPARENCY_GFA, SUNALT, SUNSEP]`

### Loading and saving models

All pre-trained weights are hosted on [HuggingFace](https://huggingface.co/datasets/mjdowicz/desisky) and downloaded automatically on first use:

```python
import desisky

# Load pre-trained weights (downloads from HuggingFace on first use)
model, meta = desisky.io.load_model("broadband")

# Load from a user checkpoint
model, meta = desisky.io.load_model("vae", path="path/to/checkpoint.eqx")

# Save a trained model
desisky.io.save(
    "my_model.eqx",
    model,
    meta={
        "schema": 1,
        "arch": {"in_channels": 7781, "latent_dim": 8},
        "training": {"date": "2025-01-15", "epoch": 100},
    },
)
```

Checkpoints use a JSON header (architecture + training metadata) followed by binary Equinox-serialized weights.

By default, downloaded weights are cached in `~/.desisky/models/<kind>/`. Override with:

```bash
export DESISKY_CACHE_DIR=/path/to/cache   # shell
```

```python
import os
os.environ["DESISKY_CACHE_DIR"] = "/path/to/cache"  # Python / notebook
```

## CLI Tools

All CLI commands are registered as console entry points and available after installation.

- **Inference** commands work with the base install (`pip install desisky`)
- **Training** commands require training dependencies: `pip install desisky[all]` (CPU) or `pip install desisky[cuda12]` (GPU)
- **Training with W&B visualization** requires the wandb extra: `pip install desisky[all,wandb]` or `pip install desisky[cuda12,wandb]`

For the full reference including data formats and wandb integration, see [`docs/CLI_GUIDE.md`](docs/CLI_GUIDE.md).

### CLI Training

```bash
# Broadband MLP (moon-contaminated data)
desisky-train-broadband --epochs 500
desisky-train-broadband --epochs 500 --wandb

# VAE (full dataset)
desisky-train-vae --epochs 100
desisky-train-vae --epochs 100 --wandb

# LDM (per-variant: dark, moon, twilight)
desisky-train-ldm --variant dark --epochs 200
desisky-train-ldm --variant moon --epochs 300 --wandb --vae-path my_vae.eqx
```

All training scripts support:
- `--wandb` for optional W&B experiment tracking with automatic visualization callbacks
- `--data-path` for user-provided data (`.fits`, `.csv`, `.npz` depending on model)
- `--no-save` to skip checkpointing (useful for testing or sweeps)
- `--vae-path` / `--model-path` for custom pretrained weights

### CLI Inference

```bash
# Broadband predictions (CSV or npz output)
desisky-infer-broadband --output predictions.csv
desisky-infer-broadband --output predictions.npz --output-format npz

# VAE encode + reconstruct
desisky-infer-vae --subset dark --output dark_latents.npz

# LDM spectral generation
desisky-infer-ldm --variant dark --n-samples 500
desisky-infer-ldm --variant moon --n-samples 100 --guidance-scale 2.0
desisky-infer-ldm --conditioning '[[60,0.9,-30,150,45,10,120,5]]'
```

## Training (Python API)

### VAE training

The VAE is trained with the **InfoVAE-MMD** objective, which provides better control over the trade-off between reconstruction quality and latent space regularization compared to standard beta-VAE. The total loss is:

```
L = Reconstruction + beta * KL + (lam - beta) * MMD
```

```python
from desisky.training import VAETrainer, VAETrainingConfig, NumpyLoader
from desisky.models.vae import make_SkyVAE
import jax.random as jr

model = make_SkyVAE(in_channels=7781, latent_dim=8, key=jr.PRNGKey(42))

config = VAETrainingConfig(
    epochs=100,
    learning_rate=1e-4,
    beta=1e-3,           # KL divergence weight
    lam=4.0,             # Total regularization weight (MMD weight = lam - beta)
    kernel_sigma="auto", # RBF kernel bandwidth for MMD
)

trainer = VAETrainer(model, config)
trained_model, history = trainer.train(train_loader, test_loader)
```

### LDM training

The LDM is trained with the **EDM framework** (Karras et al. 2022) using continuous log-normal noise sampling, preconditioned denoiser, and EDM-weighted loss. Exponential Moving Average (EMA) of model weights is maintained for stable inference.

```python
from desisky.training import (
    LatentDiffusionTrainer, LDMTrainingConfig,
    fit_conditioning_scaler, normalize_conditioning,
)
from desisky.models.ldm import compute_sigma_data

# 1. Compute sigma_data from training latents
sigma_data = compute_sigma_data(latent_train)

# 2. Fit conditioning scaler on training data (stored in checkpoint for inference)
scaler = fit_conditioning_scaler(cond_train, ["OBSALT", "TRANSPARENCY_GFA", "SUNALT", ...])

# 3. Normalize conditioning with the scaler
cond_train_norm = normalize_conditioning(cond_train, scaler)
cond_val_norm = normalize_conditioning(cond_val, scaler)

# 4. Configure training — scaler is passed here so it gets saved in checkpoint metadata
config = LDMTrainingConfig(
    epochs=200,
    learning_rate=1e-4,
    meta_dim=8,                       # Number of conditioning features
    sigma_data=sigma_data,
    ema_decay=0.9999,
    early_stop_on_ema=True,           # Gate early stopping on EMA validation loss
    conditioning_scaler=scaler,       # Saved in checkpoint for auto-normalization at inference
)

trainer = LatentDiffusionTrainer(model, config)
model, ema_model, history = trainer.train(train_loader, val_loader)
```

Both trainers support:
- Automatic best-model checkpointing
- Optional `on_epoch_end(model, history, epoch)` callback for custom per-epoch logging
- `tqdm` progress bars (auto-detected; falls back to `print_every` when unavailable)
- Training without validation (`test_loader=None` / `val_loader=None`) for final training after hyperparameters are validated

## Experiment Tracking (W&B)

Optionally integrate with [Weights & Biases](https://wandb.ai/) for real-time experiment tracking and hyperparameter sweeps:

```bash
pip install desisky[wandb]
```

```python
from desisky.training import VAETrainer, VAETrainingConfig, WandbConfig

config = VAETrainingConfig(epochs=100, learning_rate=1e-4)
wandb_config = WandbConfig(project="desisky-vae", tags=["experiment-1"])

trainer = VAETrainer(model, config, wandb_config=wandb_config)
model, history = trainer.train(train_loader, test_loader)
```

This logs all loss components (train/val) to your W&B dashboard every epoch. Add an `on_epoch_end` callback for custom visualization logging:

```python
from desisky.training import log_figure

def on_epoch_end(model, history, epoch):
    fig = plot_vae_reconstructions(originals, reconstructions, wavelength)
    log_figure("viz/reconstructions", fig, epoch)

trainer = VAETrainer(
    model, config,
    wandb_config=wandb_config,
    on_epoch_end=on_epoch_end,
)
```

W&B hyperparameter sweeps are demonstrated in notebooks 07 and 08.

## Visualization

All visualization functions return plain matplotlib `Figure` objects and are usable with or without W&B:

```python
from desisky.visualization import (
    # Experiment tracking plots
    plot_vae_reconstructions,          # Original vs reconstructed spectra
    plot_latent_corner,                # Corner plot of latent dims, colored by sky condition
    plot_latent_corner_comparison,     # Corner plot comparing two latent distributions (e.g. real vs generated)
    plot_cdf_comparison,               # CDF + histogram with Wasserstein-1 (EMD) annotation
    plot_conditional_validation_grid,  # Feature statistics vs conditioning variable with 16-84% CI bands
    plot_broadband_cdfs,               # Broadband magnitude CDF comparison
    plot_airglow_cdfs,                 # Airglow line intensity CDF comparison

    # General diagnostics
    plot_loss_curve,                   # Training/validation loss curves
    plot_nn_outlier_analysis,          # 2x3 diagnostic panel for MLP models
    plot_latent_umap,                  # UMAP projection of latent space (requires umap-learn)
)
```

## Performance Benchmarks

Key inference timings (after JIT compilation):

| Operation | CPU | GPU (RTX 3090) | Speedup |
|-----------|-----|----------------|---------|
| LDM inference (100 spectra, deterministic) | ~4,500 ms | ~130 ms | **~35x** |
| VAE encode+decode (1,000 spectra) | ~27 ms | ~1.8 ms | **~15x** |
| Broadband MLP (1,000 rows) | ~0.06 ms | ~0.06 ms | ~1x |

For practical inference runs (thousands of spectra), GPU is clearly worth it. See [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) for full results across platforms.

For running on NERSC Perlmutter with GPU support, see [`docs/NERSC_SETUP.md`](docs/NERSC_SETUP.md).

## Examples

### User-facing tutorials (`examples/`)

| Notebook | Description |
|----------|-------------|
| [00_tutorial.ipynb](examples/00_tutorial.ipynb) | **Start here** -- End-to-end walkthrough of every major capability |
| [01_broadband_training.ipynb](examples/01_broadband_training.ipynb) | Train broadband model on moon-contaminated subset |
| [02_vae_inference.ipynb](examples/02_vae_inference.ipynb) | VAE encoding/decoding and latent space visualization |
| [03_vae_analysis.ipynb](examples/03_vae_analysis.ipynb) | Latent space interpolation and anomaly detection |
| [04_vae_training.ipynb](examples/04_vae_training.ipynb) | Train VAE from scratch with InfoVAE-MMD objective |
| [05_ldm_inference.ipynb](examples/05_ldm_inference.ipynb) | Generate dark/moon/twilight spectra with EDM sampler |
| [06_ldm_training.ipynb](examples/06_ldm_training.ipynb) | Train LDM from scratch with EDM framework and EMA |
| [07_vae_wandb_training.ipynb](examples/07_vae_wandb_training.ipynb) | VAE + W&B: reconstruction plots, latent corners, sweeps |
| [08_ldm_wandb_training.ipynb](examples/08_ldm_wandb_training.ipynb) | LDM + W&B: CDF comparisons, validation grids, sweeps |

### Analysis & research notebooks (`notebooks/`)

| Notebook | Description |
|----------|-------------|
| [investigate_outlier_spectra.ipynb](notebooks/investigate_outlier_spectra.ipynb) | Investigation of anomalous LDM-generated spectra (dark samples looking moon-like, twilight outliers) |
| [specsim_polar_comparison.ipynb](notebooks/specsim_polar_comparison.ipynb) | Side-by-side comparison of desisky vs specsim sky predictions in polar coordinates |
| [desi_data_pres.ipynb](notebooks/desi_data_pres.ipynb) | Companion notebook for the DESI data systems presentation (slide figures, validation plots) |

## Project Structure

```
desisky/
├── src/desisky/
│   ├── data/                   # Data loading, enrichment, spectral analysis
│   │   ├── skyspec.py          #   SkySpecVAC class with subset filtering
│   │   ├── _core.py            #   Download utilities with SHA-256 verification
│   │   ├── _enrich.py          #   V-band, eclipse, solar flux, coordinates
│   │   ├── _spectral.py        #   Airglow line intensities, broadband magnitudes
│   │   └── _splits.py          #   Validation mask utilities
│   ├── models/                 # Model architectures (JAX/Equinox)
│   │   ├── broadband.py        #   Broadband MLP
│   │   ├── vae.py              #   SkyVAE encoder-decoder
│   │   └── ldm.py              #   1D U-Net + EDM preconditioning
│   ├── io/                     # Model I/O and checkpoint handling
│   │   └── model_io.py         #   Save/load with JSON header + binary weights
│   ├── inference/              # Sampling algorithms
│   │   └── sampling.py         #   EDM Heun ODE solver, classifier-free guidance
│   ├── training/               # Training infrastructure
│   │   ├── trainer.py          #   BroadbandTrainer
│   │   ├── vae_trainer.py      #   VAETrainer (InfoVAE-MMD)
│   │   ├── ldm_trainer.py      #   LatentDiffusionTrainer (EDM)
│   │   ├── dataset.py          #   PyTorch Dataset/DataLoader wrappers
│   │   ├── losses.py           #   L2, Huber loss functions
│   │   ├── vae_losses.py       #   InfoVAE-MMD loss with RBF kernel
│   │   └── wandb_utils.py      #   W&B logging utilities
│   ├── visualization/          # Plotting
│   │   ├── plots.py            #   Loss curves, outlier analysis, broadband band panels
│   │   └── wandb_plots.py      #   Reconstructions, corner plots, CDFs, validation grids
│   └── scripts/                # CLI tools
│       ├── download_data.py    #   desisky-data command
│       ├── train_broadband.py  #   desisky-train-broadband
│       ├── train_vae.py        #   desisky-train-vae
│       ├── train_ldm.py        #   desisky-train-ldm
│       ├── infer_broadband.py  #   desisky-infer-broadband
│       ├── infer_vae.py        #   desisky-infer-vae
│       └── infer_ldm.py        #   desisky-infer-ldm
├── tests/                      # 382 unit tests
├── examples/                   # 9 user-facing tutorial notebooks
├── notebooks/                  # Analysis & presentation notebooks
├── scripts/                    # Standalone scripts (NERSC VAE training, data prep)
├── jobs/                       # SLURM job scripts for NERSC training
├── benchmarks/                 # Inference/training timing benchmarks
├── docs/
│   ├── CLI_GUIDE.md            # CLI data formats, output formats, wandb reference
│   ├── BENCHMARKS.md           # Cross-platform timing results (Entropy + NERSC Perlmutter)
│   └── NERSC_SETUP.md          # NERSC environment setup and GPU training recipes
├── pyproject.toml
├── CHANGELOG.md
└── LICENSE.txt
```

## Development

```bash
git clone https://github.com/MatthewDowicz/desisky.git
cd desisky
pip install -e ".[all]"
pip install pytest pytest-cov

# Run all tests
pytest

# Run with coverage
pytest --cov=desisky --cov-report=html

# Run specific test file
pytest tests/test_model_io.py -v
```

## License

`desisky` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
