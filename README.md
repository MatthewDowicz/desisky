# desisky

[![PyPI - Version](https://img.shields.io/pypi/v/desisky.svg)](https://pypi.org/project/desisky)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/desisky.svg)](https://pypi.org/project/desisky)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/MatthewDowicz/desisky/actions/workflows/tests.yml/badge.svg)](https://github.com/MatthewDowicz/desisky/actions/workflows/tests.yml)

-----

## About

`desisky` provides machine learning models and tools for DESI sky modeling:

1. **Predictive broadband model** - Predicts surface brightness in V, g, r, and z photometric bands from observational metadata (moon position, transparency, eclipse fraction)
2. **Variational Autoencoder (VAE)** - Compresses sky spectra (7,781 wavelength points → 8-dimensional latent space) for analysis, anomaly detection, and dimensionality reduction
3. **Latent Diffusion Models (LDM)** - Generates realistic night-sky emission spectra conditioned on observational parameters:
   - **LDM Dark** - Dark-time spectra conditioned on sun position, transparency, galactic/ecliptic coordinates, and solar flux
   - **LDM Moon** - Moon-contaminated spectra conditioned on moon position, separation, and illumination fraction
   - **LDM Twilight** - Twilight spectra conditioned on observation altitude, transparency, sun altitude, and sun separation
4. **Data utilities** - Download and load DESI DR1 Sky Spectra Value-Added Catalog (VAC) with automatic integrity verification and subset filtering

Built with **JAX/Equinox** for high-performance model inference and designed to integrate with SpecSim and survey forecasting workflows. This repository hosts the code and notebooks supporting the forthcoming paper by Dowicz et al. (20XX).

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Broadband Model](#load-pre-trained-broadband-model-and-run-inference)
  - [VAE](#load-pre-trained-vae-and-encode-sky-spectra)
  - [Latent Diffusion Model](#generate-sky-spectra-with-latent-diffusion-model)
  - [Data Loading](#download-and-load-desi-sky-spectra-data)
- [Data Subsets](#data-subsets)
- [Loading Pre-trained Models](#loading-pre-trained-models)
- [Data Download](#data-download)
- [Examples](#examples)
- [Development](#development)
- [License](#license)

## Installation

### Basic installation (model inference only)

```bash
pip install desisky[cpu]
```

### With data utilities (includes FITS file reading)

```bash
pip install desisky[cpu,data]
```

### For GPU support

```bash
pip install desisky[cuda12,data]
```

**Note:** CUDA wheels require manual installation. See [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) for details.

## Quick Start

### Load pre-trained broadband model and run inference

```python
import desisky
import jax.numpy as jnp

# Load the pre-trained broadband model
model, meta = desisky.io.load_model("broadband")

# Example input: [placeholder for actual feature names]
x = jnp.array([...])  # Shape: (6,)

# Predict surface brightness in V, g, r, z bands
y = model(x)  # Shape: (4,)
print(f"Predicted magnitudes: {y}")
```

### Load pre-trained VAE and encode sky spectra

```python
from desisky.io import load_builtin
from desisky.data import SkySpecVAC
import jax.random as jr

# Load DESI sky spectra
vac = SkySpecVAC(version="v1.0", download=True)
wavelength, flux, metadata = vac.load()

# Load pre-trained VAE
vae, meta = load_builtin("vae")

# Encode a sky spectrum to latent representation
spectrum = flux[0].squeeze()
mean, logvar = vae.encode(spectrum)
print(f"Latent mean: {mean}")  # Shape: (8,)

# Sample and decode
latent = vae.sample(mean, logvar, jr.PRNGKey(0))
reconstructed = vae.decode(latent)
print(f"Reconstructed shape: {reconstructed.shape}")  # Shape: (7781,)

# Batch processing with vmap
import jax
batch_means, batch_logvars = jax.vmap(vae.encode)(flux.squeeze())
print(f"Batch latents shape: {batch_means.shape}")  # Shape: (9176, 8)
```

### Generate sky spectra with Latent Diffusion Model

```python
from desisky.io import load_builtin
from desisky.inference import LatentDiffusionSampler
import jax.random as jr
import jax.numpy as jnp

# Load pre-trained VAE and LDM
vae, _ = load_builtin("vae")
ldm, _ = load_builtin("ldm_dark")

# Create sampler (Heun method recommended for quality)
sampler = LatentDiffusionSampler(
    ldm_model=ldm,
    vae_model=vae,
    method="heun",
    num_steps=1000
)

# Define conditioning: [OBSALT, TRANSP, SUNALT, SOLFLUX, ECLLON, ECLLAT, GALLON, GALLAT]
conditioning = jnp.array([
    [2100.0, 0.9, -30.0, 150.0, 45.0, 10.0, 120.0, 5.0],  # Dark sky conditions
])

# Generate spectrum
generated_spectra = sampler.sample(
    key=jr.PRNGKey(42),
    conditioning=conditioning,
    guidance_scale=2.0
)

print(f"Generated spectrum shape: {generated_spectra.shape}")  # (1, 7781)
```

### Download and load DESI sky spectra data

```python
from desisky.data import SkySpecVAC

# Download DR1 VAC (274 MB, with SHA-256 verification)
vac = SkySpecVAC(version="v1.0", download=True)

# Load wavelength, flux, and metadata
wavelength, flux, metadata = vac.load()

print(f"Wavelength shape: {wavelength.shape}")  # (7781,)
print(f"Flux shape: {flux.shape}")              # (9176, 7781)
print(f"Metadata columns: {list(metadata.columns)}")
# ['NIGHT', 'EXPID', 'TILEID', 'AIRMASS', 'EBV', 'MOONFRAC', 'MOONALT', ...]

# Load with enrichment (adds V-band magnitudes and eclipse fraction)
wavelength, flux, metadata = vac.load(enrich=True)
print('SKY_MAG_V_SPEC' in metadata.columns)  # True
print('ECLIPSE_FRAC' in metadata.columns)    # True
```

## Data Subsets

The VAC provides three subset methods for filtering observations by sky conditions:

### Dark Time (Non-contaminated)

```python
# Load observations with minimal sun/moon contamination
wave, flux, meta = vac.load_dark_time()

# Filtering criteria:
# - SUNALT < -20° (Sun well below horizon)
# - MOONALT < -5° (Moon below horizon)
# - TRANSPARENCY_GFA > 0 (valid measurements)
```

### Sun Contaminated (Twilight)

```python
# Load twilight observations
wave, flux, meta = vac.load_sun_contaminated()

# Filtering criteria:
# - SUNALT > -20° (Sun near or above horizon)
# - MOONALT <= -5° (Moon below horizon)
# - MOONSEP <= 110° (Sun-Moon separation)
# - TRANSPARENCY_GFA > 0
```

### Moon Contaminated

```python
# Load moon-bright observations
wave, flux, meta = vac.load_moon_contaminated()

# Filtering criteria:
# - SUNALT < -20° (nighttime)
# - MOONALT > 5° (Moon above horizon)
# - MOONFRAC > 0.5 (Moon >50% illuminated)
# - MOONSEP <= 90° (Moon within 90°)
# - TRANSPARENCY_GFA > 0
```

All subset methods include enrichment by default (`enrich=True`), adding computed columns for V-band magnitude and lunar eclipse fraction.

## Loading Pre-trained Models

The `desisky.io.load_model()` function provides a unified interface for loading models:

```python
import desisky

# Load packaged pre-trained weights
model, meta = desisky.io.load_model("broadband")

# Load from a custom checkpoint
model, meta = desisky.io.load_model("broadband", path="path/to/checkpoint.eqx")

# Save your own trained model
desisky.io.save(
    "my_model.eqx",
    model,
    meta={
        "schema": 1,
        "arch": {"in_size": 6, "out_size": 4, "width_size": 128, "depth": 5},
        "training": {"date": "2025-01-15", "commit": "abc123"},
    }
)
```

**Available models:**
- `"broadband"` - Multi-layer perceptron (6 inputs → 4 outputs) for V, g, r, z magnitude prediction from moon/transparency conditions
- `"vae"` - Variational autoencoder (7781 → 8 → 7781) for sky spectra compression, reconstruction, and latent space analysis
- `"ldm_dark"` - Latent diffusion model (1D U-Net) for generating dark-time sky spectra conditioned on 8 observational parameters:
  - Conditioning: `[OBSALT, TRANSPARENCY_GFA, SUNALT, SOLFLUX, ECLLON, ECLLAT, GALLON, GALLAT]`
- `"ldm_moon"` - Latent diffusion model (1D U-Net) for generating moon-contaminated sky spectra conditioned on 6 observational parameters:
  - Conditioning: `[OBSALT, TRANSPARENCY_GFA, SUNALT, MOONALT, MOONSEP, MOONFRAC]`
- `"ldm_twilight"` - Latent diffusion model (1D U-Net) for generating twilight sky spectra conditioned on 4 observational parameters:
  - Conditioning: `[OBSALT, TRANSPARENCY_GFA, SUNALT, SUNSEP]`

## Data Download

### Python API

```python
from desisky.data import SkySpecVAC

# Download to default location (~/.desisky/data)
vac = SkySpecVAC(download=True)

# Download to custom location
vac = SkySpecVAC(root="/path/to/data", download=True)

# Skip SHA-256 verification (not recommended)
vac = SkySpecVAC(download=True, verify=False)

# Get path to downloaded file
print(vac.filepath())
```

### Command-line interface

```bash
# Show default data directory
desisky-data dir

# Download DESI DR1 sky spectra VAC
desisky-data fetch --version v1.0

# Download to custom location
desisky-data fetch --root /path/to/data

# Skip checksum verification
desisky-data fetch --no-verify
```

### Environment variable

Override the default data directory:

```bash
export DESISKY_DATA_DIR=/path/to/data
desisky-data dir  # Shows /path/to/data
```

## Examples

See [examples/](examples/) directory for Jupyter notebooks demonstrating:

- **[00_quickstart.ipynb](examples/00_quickstart.ipynb)** - Quick introduction: loading models, data subsets, and running inference
- **[01_broadband_training.ipynb](examples/01_broadband_training.ipynb)** - Train the broadband model on moon-contaminated subset
- **[02_vae_inference.ipynb](examples/02_vae_inference.ipynb)** - VAE inference: encoding/decoding sky spectra and latent space visualization
- **[03_vae_analysis.ipynb](examples/03_vae_analysis.ipynb)** - Advanced VAE analysis: latent space interpolation and anomaly detection
- **[04_vae_training.ipynb](examples/04_vae_training.ipynb)** - Train a VAE from scratch with InfoVAE-MMD objective
- **[05_ldm_inference.ipynb](examples/05_ldm_inference.ipynb)** - Generate dark-time and moon-contaminated sky spectra using the latent diffusion models with custom conditioning

## Development

### Setting up development environment

```bash
git clone https://github.com/MatthewDowicz/desisky.git
cd desisky
pip install -e ".[cpu,data]"
pip install pytest pytest-cov
```

### Running tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=desisky --cov-report=html

# Run specific test file
pytest tests/test_model_io.py -v
```

### Project Structure

```
desisky/
├── src/desisky/
│   ├── io/              # Model I/O (save/load checkpoints with metadata)
│   ├── models/          # Model architectures
│   │   ├── broadband.py # Broadband MLP for magnitude prediction
│   │   ├── vae.py       # Variational autoencoder (InfoVAE-MMD)
│   │   └── ldm.py       # Latent diffusion model (1D U-Net)
│   ├── data/            # Data downloading, loading, and enrichment
│   │   ├── skyspec.py   # SkySpecVAC class with subset filtering
│   │   ├── _enrich.py   # V-band, eclipse, solar flux, coordinates
│   │   └── _core.py     # Download utilities with SHA-256 verification
│   ├── training/        # Training infrastructure
│   │   ├── dataset.py   # PyTorch Dataset wrappers
│   │   ├── vae_trainer.py    # VAE training loop
│   │   ├── losses.py         # Loss functions
│   │   └── vae_losses.py     # InfoVAE-MMD loss
│   ├── inference/       # Sampling algorithms
│   │   └── sampling.py  # DDPM, DDIM, Heun samplers for LDM
│   ├── visualization/   # Plotting utilities
│   ├── scripts/         # CLI tools (desisky-data)
│   └── weights/         # Pre-trained model weights (small models)
├── tests/               # Comprehensive test suite (123+ tests)
│   ├── test_vae.py           # VAE unit tests
│   ├── test_model_io.py      # Model I/O tests
│   ├── test_enrichment.py    # Data enrichment tests
│   ├── test_ldm_sampling.py  # LDM sampling tests
│   └── ...                   # Other test modules
├── examples/            # Jupyter notebook tutorials
│   ├── 00_quickstart.ipynb
│   ├── 01_broadband_training.ipynb
│   ├── 02_vae_inference.ipynb
│   ├── 03_vae_analysis.ipynb
│   ├── 04_vae_training.ipynb
│   └── 05_ldm_inference.ipynb
└── pyproject.toml       # Package configuration
```

### Key Features

- **JAX/Equinox models**: High-performance, functional ML models with automatic differentiation
- **Production-ready I/O**: Checkpoint format with JSON metadata + binary weights
- **Automatic caching**: Downloaded data and models cached locally for fast re-use
- **Integrity verification**: SHA-256 checksums for all downloaded files
- **Subset filtering**: Easy access to dark-time, twilight, and moon-contaminated observations
- **Data enrichment**: Automatic computation of V-band magnitudes, eclipse fractions, solar flux, and coordinate transformations
- **Multiple sampling methods**: DDPM, DDIM, and Heun (probability-flow ODE) for LDM inference
- **Comprehensive tests**: 123+ unit tests ensuring reliability

## License

`desisky` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
