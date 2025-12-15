# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2025-12-12

### Added

- Pre-trained Latent Diffusion Model for moon-contaminated sky spectra (`ldm_moon`)
  - Trained on observations with moon altitude > 5°, moon fraction > 0.5, and moon separation ≤ 90°
  - Conditioned on 6 observational parameters: `[OBSALT, TRANSPARENCY_GFA, SUNALT, MOONALT, MOONSEP, MOONFRAC]`
  - Enables generation of realistic moon-contaminated sky spectra for different lunar conditions
- Pre-trained Latent Diffusion Model for twilight sky spectra (`ldm_twilight`)
  - Trained on observations with sun altitude > -20° (twilight conditions)
  - Conditioned on 4 observational parameters: `[OBSALT, TRANSPARENCY_GFA, SUNALT, SUNSEP]`
  - Enables generation of realistic twilight sky spectra for different solar elevation angles
- Updated `05_ldm_inference.ipynb` with examples comparing dark-time, moon-contaminated, and twilight models
- Updated `06_ldm_training.ipynb` with twilight model training example
- Integration tests for both `ldm_moon` and `ldm_twilight` model loading and inference
- Documentation updates for all three LDM variants in README.md

## [0.1.0] - 2025-12-03

### Added

- Initial release of `desisky` package
- Pre-trained broadband model for V, g, r, z magnitude prediction from observational metadata (moon position, transparency, eclipse fraction)
- Variational Autoencoder (VAE) for sky spectra compression (7,781 wavelength points → 8-dimensional latent space)
- Latent Diffusion Model (LDM) for generating realistic dark-time night-sky emission spectra conditioned on 8 observational parameters
- Data utilities for downloading and loading DESI DR1 Sky Spectra Value-Added Catalog (VAC) with automatic SHA-256 integrity verification
- Subset filtering methods for different observing conditions:
  - `load_dark_time()` - Non-contaminated observations (sun/moon below horizon)
  - `load_sun_contaminated()` - Twilight observations
  - `load_moon_contaminated()` - Moon-bright observations
- Data enrichment features:
  - V-band magnitude computation from spectra
  - Lunar eclipse fraction calculation
  - Solar flux integration
  - Galactic and ecliptic coordinate transformations
- Command-line interface `desisky-data` for data management (download, verify, locate)
- Multiple sampling methods for latent diffusion inference:
  - DDPM (Denoising Diffusion Probabilistic Models)
  - DDIM (Denoising Diffusion Implicit Models)
  - Heun (probability-flow ODE solver)
- Production-ready model I/O system with JSON metadata + binary weights
- Automatic caching for downloaded data and pre-trained models
- Comprehensive test suite with 123+ unit tests covering all major functionality
- Example Jupyter notebooks:
  - `00_quickstart.ipynb` - Quick introduction to loading models and data
  - `01_broadband_training.ipynb` - Train broadband model
  - `02_vae_inference.ipynb` - VAE encoding/decoding
  - `03_vae_analysis.ipynb` - Latent space analysis
  - `04_vae_training.ipynb` - Train VAE from scratch
  - `05_ldm_inference.ipynb` - Generate sky spectra with LDM
  - `06_ldm_training.ipynb` - Train LDM from scratch
- JAX/Equinox-based models with automatic differentiation for high-performance inference
- PyTorch DataLoader integration for training workflows
- Support for CPU and CUDA (GPU) installations
- MIT License

[unreleased]: https://github.com/MatthewDowicz/desisky/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/MatthewDowicz/desisky/compare/v0.1.0...v0.3.0
[0.1.0]: https://github.com/MatthewDowicz/desisky/releases/tag/v0.1.0
