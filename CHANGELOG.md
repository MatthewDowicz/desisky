# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.8.0] - 2026-03-18

### Added

- **String-based sampler construction** — `LatentDiffusionSampler("ldm_dark")` auto-loads the pretrained model, VAE, and training metadata (sigma_data, conditioning_scaler) in one line
- `ldm_path` / `vae_path` parameters for loading custom checkpoints with auto meta-extraction
- `ValueError` with clear message when `sigma_data` is missing in Module mode
- **UMAP latent-space visualization** — `plot_latent_umap()` function in `desisky.visualization`; `umap-learn` added as optional dependency under `viz` extras
- `plot_latent_umap(fit_on=...)` parameter — fit UMAP on a reference dataset (e.g. real data) and project all points through the same mapping, for fair real-vs-generated comparisons
- **NERSC setup guide** — `docs/NERSC_SETUP.md` with environment setup, GPU inference recipe, and cache configuration for Perlmutter
- **Benchmarks** — `docs/BENCHMARKS.md` with Entropy CPU/GPU timing results; `benchmarks/timing.py` script for reproducible benchmarking

### Changed

- `LatentDiffusionSampler` first parameter renamed from `ldm_model` to `ldm`; accepts `str` (auto-load) or `eqx.Module` (manual). Positional callers `(module, vae, sigma_data)` are unaffected.
- `vae_model` parameter now defaults to `None` (auto-loads pretrained VAE) instead of being required
- `sigma_data` parameter now defaults to `None` (auto-extracted from checkpoint metadata in string mode) instead of being required
- Default `num_steps` changed from 100 to 50 (fast option)
- Renamed `TRANSPARENCY_GFA` to `transparency` in user-facing documentation and comments (programmatic column names unchanged)
- **Tutorial notebook restructured** — reordered for user-facing workflows first (quick start → data → validation → internals), added table of contents, simplified framing for users without ML background
- Latent corner plot histograms cleaned up for better readability

### Breaking

- Code using `ldm_model=` as a keyword argument must change to `ldm=`
- Code relying on `num_steps` defaulting to 100 must now pass `num_steps=100` explicitly

## [0.7.0] - 2026-03-05

### Added

- **Tutorial notebook** — `examples/00_tutorial.ipynb` covers every major capability end-to-end (data loading, enrichment, spectral analysis, broadband MLP, VAE, LDM dark/moon/twilight, validation, model I/O, training, CLI)
- **Stochastic sampler** — `sample_edm_stochastic()` implements Algorithm 2 from Karras et al. 2022 (Langevin-corrected reverse SDE). Use `sampler.sample(stochastic=True)` for diverse ensemble generation.

### Changed

- `SkySpecVAC.load()` now auto-enriches with all 5 columns (V-band, eclipse fraction, solar flux, galactic coords, ecliptic coords); the `enrich` parameter has been removed
- `load_dark_time()`, `load_sun_contaminated()`, `load_moon_contaminated()` no longer accept an `enrich` parameter

### Removed

- `00_quickstart.ipynb` — superseded by the tutorial notebook

## [0.6.0] - 2026-03-04

### Added

- **CLI training scripts** for all three model types, each with optional W&B integration:
  - `desisky-train-broadband` — Train broadband MLP on moon-contaminated data with per-band scatter+residual visualization callbacks
  - `desisky-train-vae` — Train VAE (InfoVAE-MMD) with reconstruction, latent corner, broadband CDF, and airglow CDF visualization callbacks
  - `desisky-train-ldm` — Train LDM (EDM framework) for dark/moon/twilight variants with broadband CDF, airglow CDF, latent corner comparison, and conditional validation grid callbacks
- **CLI inference scripts** for all three model types:
  - `desisky-infer-broadband` — Run broadband MLP inference, output CSV or npz
  - `desisky-infer-vae` — VAE encode+reconstruct with subset selection (full/dark/moon/twilight)
  - `desisky-infer-ldm` — Generate spectra with LDM; conditioning from real data, user file, or inline JSON
- **Multi-format data loading** for LDM training (`--data-path`):
  - `.npz` — Pre-processed data (flux, conditioning, wavelength keys)
  - `.fits` — SkySpecVAC format with automatic variant-specific quality filtering
  - `.csv` + `--flux-path` — Metadata CSV with separate `.npy` flux array
- **`plot_broadband_band_panel()`** — New 2x2 scatter+residual diagnostic panel for broadband training visualization
- `BroadbandTrainer` now supports `wandb_config` and `on_epoch_end` callback parameters (backward-compatible, both default to `None`)
- 6 new console entry points registered in `pyproject.toml`
- `docs/CLI_GUIDE.md` — Documents all CLI commands, data formats, output formats, and wandb integration
- Integration tests for all 6 CLI scripts (3 training + 3 inference)

### Changed

- `train_vae.py` — Full rewrite to use `VAETrainer` with wandb callbacks; uses raw tensors instead of `TensorDataset` for correct `VAETrainer` iteration
- `BroadbandTrainer` restructured with `train()` / `_train_loop()` pattern matching `VAETrainer` / `LatentDiffusionTrainer`

### Fixed

- LDM training encoding now uses VAE full forward pass (`vae(batch, key)["latent"]`) with reparameterization, matching the notebook pattern, instead of `vae.encode()` which returns a raw tuple
- SkySpecVAC method name mapping: `load_dark` → `load_dark_time`, `load_moon` → `load_moon_contaminated`, `load_twilight` → `load_sun_contaminated`
- LDM training and inference now apply manual metadata enrichment (`attach_solar_flux`, `add_galactic_coordinates`, `add_ecliptic_coordinates`) since `enrich=True` only adds `SKY_MAG_V_SPEC` and `ECLIPSE_FRAC`
- NaN/Inf rows in conditioning columns are now detected and removed during data loading
- `plot_conditional_validation_grid()` no longer crashes with small validation sets (e.g., 29 twilight samples): adaptive bin count and proper empty-bin handling
- `plot_cdf_comparison()` histograms now use shared bin edges between real and generated data for accurate visual comparison
- `plot_broadband_band_panel()` histogram bins reduced from 80 to 30; combined +/- sigma into single legend entry; relabeled zero-line as "0 (perfect)"
- Suppressed harmless speclite `log10` warnings from negative flux values in early-training generated spectra

## [0.5.0] - 2026-02-18

### Added

- **Weights & Biases experiment tracking** as optional dependency (`pip install desisky[wandb]`):
  - `WandbConfig` dataclass for configuring project, entity, tags, logging frequency
  - `init_wandb_run()` — sweep-aware run initialization (reuses active run inside `wandb.agent`)
  - `log_epoch_metrics()`, `log_figure()`, `finish_wandb_run()` — logging helpers
  - `eval_per_sigma_losses()` — compute EDM loss at fixed sigma levels for diagnostics
- **Spectral analysis module** `desisky.data._spectral`:
  - `measure_airglow_intensities()` — continuum-subtracted airglow emission line intensities (10 lines + 2 composites, following Noll et al. 2012)
  - `compute_broadband_mags()` — V, g, r, z broadband magnitudes via speclite
  - Constants: `LINE_BANDS`, `AIRGLOW_CDF_NAMES`, `BROADBAND_NAMES`, `FLUX_SCALE`
- **New visualization functions** in `desisky.visualization.wandb_plots` (all return matplotlib Figures, wandb-agnostic):
  - `plot_vae_reconstructions()` — original vs reconstructed spectra
  - `plot_latent_corner()` — corner/pair-plot of latent dimensions, colored by sky condition
  - `plot_latent_corner_comparison()` — corner plot comparing two latent distributions (e.g. real vs generated) with per-dimension EMD annotation
  - `plot_cdf_comparison()` — CDF + histogram with Wasserstein-1 (EMD) annotation
  - `plot_conditional_validation_grid()` — feature statistics vs conditioning variable with 16-84% CI bands
  - `plot_broadband_cdfs()` — broadband magnitude CDF comparison (convenience wrapper)
  - `plot_airglow_cdfs()` — airglow line intensity CDF comparison (convenience wrapper)
- `on_epoch_end` callback parameter on both `VAETrainer` and `LatentDiffusionTrainer` for custom per-epoch visualization logging
- `LDMTrainingConfig.early_stop_on_ema` parameter (default `True`) — early stopping and best-model saving gate on EMA validation loss instead of base model loss
- `LDMTrainingHistory.ema_val_losses` field — tracks EMA model validation loss per epoch
- Both base and EMA validation losses are now always computed and logged when EMA is enabled
- `tqdm` progress bar in both trainers (auto-detected; falls back to `print_every` when tqdm is unavailable)
- Example notebooks:
  - `07_vae_wandb_training.ipynb` — VAE training with wandb tracking, reconstruction plots, latent corner plots, and sweep example
  - `08_ldm_wandb_training.ipynb` — LDM training with wandb tracking, CDF comparisons, conditional validation grids, and sweep example
- Tests: `test_wandb_utils.py`, `test_wandb_plots.py`, `test_spectral.py`

### Changed

- `VAETrainer.train()` and `LatentDiffusionTrainer.train()` now accept optional `val_loader=None` / `test_loader=None` to train without validation (useful for final training on the full dataset after hyperparameters are validated)
- Trainers internally split into `train()` (wandb lifecycle with try/finally) and `_train_loop()` (core loop) for clean resource cleanup
- Checkpoint naming priority: user-specified `run_name` > wandb auto-generated name > default
- `LatentDiffusionTrainer._evaluate()` now takes an `eval_model` parameter to evaluate either the base or EMA model

## [0.4.0] - 2026-02-12

### Added

- EDM preconditioning framework (Karras et al. 2022) in `desisky.models.ldm`:
  - Preconditioning functions (`c_skip`, `c_out`, `c_in`, `c_noise`) and `edm_denoiser` wrapper
  - EDM constants (`EDM_SIGMA_MIN`, `EDM_SIGMA_MAX`, `EDM_P_MEAN`, `EDM_P_STD`)
  - `compute_sigma_data` utility for computing training data standard deviation
- EDM sampling via 2nd-order Heun ODE solver with Karras sigma schedule (`get_sigmas_karras`, `sample_edm`)
- Automatic conditioning normalization in `LatentDiffusionSampler` via `conditioning_scaler` parameter
- Conditioning normalization utilities (`fit_conditioning_scaler`, `normalize_conditioning`) in `desisky.training` — plain numpy, no scikit-learn dependency
- EMA (Exponential Moving Average) model tracking during LDM training (decay=0.9999)
- `val_expids` and `conditioning_scaler` stored in checkpoint metadata for reproducible inference
- `get_validation_mask` utility in `desisky.data` for identifying held-out samples from checkpoint metadata
- Tests for conditioning scaler utilities and validation mask
- Updated example notebooks (`05_ldm_inference.ipynb`, `06_ldm_training.ipynb`) for EDM API

### Changed

- `LatentDiffusionTrainer.train()` now returns 3 values `(model, ema_model, history)` instead of 2
- `LatentDiffusionSampler` now requires `sigma_data` parameter; `conditioning_scaler` enables auto-normalization of raw conditioning inputs
- LDM training uses continuous log-normal sigma sampling and EDM-weighted loss instead of discrete cosine beta schedule
- Retrained LDM checkpoints (`ldm_dark`, `ldm_moon`, `ldm_twilight`) with EDM framework — model sizes reduced from ~4.4 MB to ~1.3 MB

### Removed

- DDPM and DDIM sampling methods (`cosine_beta_schedule`, `guided_denoising_step`)
- `method` parameter from `LatentDiffusionSampler` (EDM Heun solver is now the only sampler)
- `n_T` parameter from `LDMTrainingConfig` (replaced by continuous `sigma_data`)

### Fixed

- `SkySpecVAC.get_sun_contaminated()` incorrectly used `MOONSEP` instead of `SUNSEP` for twilight filtering

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
- Variational Autoencoder (VAE) for sky spectra compression (7,781 wavelength bins → 8-dimensional latent space)
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

[unreleased]: https://github.com/MatthewDowicz/desisky/compare/v0.7.0...HEAD
[0.7.0]: https://github.com/MatthewDowicz/desisky/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/MatthewDowicz/desisky/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/MatthewDowicz/desisky/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/MatthewDowicz/desisky/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/MatthewDowicz/desisky/compare/v0.1.0...v0.3.0
[0.1.0]: https://github.com/MatthewDowicz/desisky/releases/tag/v0.1.0
