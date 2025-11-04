# desisky

[![PyPI - Version](https://img.shields.io/pypi/v/desisky.svg)](https://pypi.org/project/desisky)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/desisky.svg)](https://pypi.org/project/desisky)

-----

## About

`desisky` provides machine learning models and tools for DESI sky modeling:

1. **Predictive broadband model** - Returns surface brightness in V, g, r, and z photometric bands from observational metadata
2. **Generative sky-spectrum model** *(coming soon)* - Synthesizes realistic night-sky emission spectra using latent diffusion models
3. **Data utilities** - Download and load DESI sky spectra Value-Added Catalog (VAC) with automatic integrity verification

Built with **JAX/Equinox** and designed to integrate with SpecSim and survey forecasting workflows. This repository hosts the code and notebooks supporting the forthcoming paper by Dowicz et al. (20XX).

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
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
# ['NIGHT', 'EXPID', 'TILEID', 'AIRMASS', 'EBV', ...]
```

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
- `"broadband"` - Multi-layer perceptron for V, g, r, z magnitude prediction

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

- **Model inference** - Loading models and running predictions
- **Data visualization** - Plotting sky spectra and observing conditions
- **Integration examples** - Using desisky with SpecSim

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
│   ├── io/           # Model I/O (save/load checkpoints)
│   ├── models/       # Model architectures (broadband MLP, VAE, diffusion)
│   ├── data/         # Data downloading and loading utilities
│   ├── scripts/      # CLI tools (desisky-data)
│   └── weights/      # Pre-trained model weights
├── tests/            # Comprehensive test suite (36 tests)
├── examples/         # Jupyter notebook examples
└── pyproject.toml    # Package configuration
```

## License

`desisky` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
