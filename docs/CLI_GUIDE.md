# CLI Training & Inference Guide

This guide documents the CLI commands for training and inference across all three model types: broadband MLP, VAE, and LDM.

## Quick Reference

```bash
# Download data
desisky-data fetch

# Train broadband (default: moon data, 500 epochs)
desisky-train-broadband --epochs 500

# Train broadband with wandb tracking
desisky-train-broadband --epochs 500 --wandb --wandb-project my-project

# Train VAE on full dataset
desisky-train-vae --epochs 100

# Train LDM dark-time variant
desisky-train-ldm --variant dark --epochs 200

# Train LDM with wandb and custom VAE
desisky-train-ldm --variant moon --epochs 300 --wandb --vae-path my_vae.eqx

# Run broadband inference
desisky-infer-broadband --output predictions.csv

# Run VAE inference on dark-time subset
desisky-infer-vae --subset dark --output dark_latents.npz

# Generate 500 dark-time spectra
desisky-infer-ldm --variant dark --n-samples 500

# Generate with custom conditioning
desisky-infer-ldm --variant dark --cond-path my_conditions.npz --n-samples 100
```

---

## User-Provided Data Formats

### Broadband Training/Inference (`--data-path`)

Accepts FITS (`.fits`) or CSV (`.csv`), auto-detected by extension.

**Required columns:**
- Input features: `MOONSEP`, `MOONFRAC`, `MOONALT`, `OBSALT`, `TRANSPARENCY_GFA`, `ECLIPSE_FRAC`
- Target magnitudes: `V`, `g`, `r`, `z`
- Optional: `EXPID` (exposure IDs, preserved in output)

### VAE Training/Inference (`--data-path`)

Expected `.npz` keys:
- `flux`: float32 array `(N, 7781)` — sky spectra

Wavelength grid is hardcoded to the DESI grid (via `SkySpecVAC.load_wavelength()`). Users do not need to provide a wavelength array.

### LDM Training (`--data-path`)

Supports three formats, auto-detected by extension:

**`.npz` (pre-processed):**
- `flux`: float32 array `(N, 7781)` — sky spectra
- `conditioning`: float32 array `(N, meta_dim)` — conditioning features
- `wavelength`: float32 array `(7781,)` — wavelength grid

**`.fits` (SkySpecVAC format):**
- Auto-filtered by `--variant` using the same quality cuts as the built-in subset methods
- Metadata enrichment (solar flux, galactic/ecliptic coordinates) applied automatically

**`.csv` + `--flux-path` (metadata + spectra):**
- CSV contains metadata columns; `--flux-path` points to a `.npy` flux array `(N, 7781)`
- Auto-filtered by `--variant`; enrichment applied automatically

### LDM Inference Conditioning (`--cond-path`)

Accepts `.npz` or `.csv`:
- `.npz`: key `conditioning` — float32 array `(N, meta_dim)`, raw conditioning vectors
- `.csv`: columns matching the variant's conditioning columns (e.g., `OBSALT`, `SUNALT`, ...)

Or use `--conditioning` for inline JSON: `'[[60.0, 0.9, -30.0, ...]]'`

---

## Output Data Formats

### Broadband (`desisky-infer-broadband`)

**CSV (default):** columns `EXPID, V_pred, g_pred, r_pred, z_pred, V_obs, g_obs, r_obs, z_obs`

**npz (optional, `--output-format npz`):** keys `predictions` (N,4), `observed` (N,4), `expids` (N,)

CSV is the default because broadband predictions are inherently tabular (4 magnitudes + EXPIDs) — users can open them in pandas, TOPCAT, or Excel.

### VAE (`desisky-infer-vae`)

**npz:** keys `wavelength` (7781,), `latents` (N,8), `means` (N,8), `logvars` (N,8), `reconstructed` (N,7781), `recon_error` (N,)

### LDM (`desisky-infer-ldm`)

**npz:** keys `spectra` (N,7781), `latents` (N,1,8), `conditioning` (N,meta_dim), `wavelength` (7781,)

---

## LDM Conditioning Columns by Variant

| Variant   | Columns | meta_dim |
|-----------|---------|----------|
| dark      | OBSALT, TRANSPARENCY_GFA, SUNALT, SOLFLUX, ECLLON, ECLLAT, GALLON, GALLAT | 8 |
| moon      | OBSALT, TRANSPARENCY_GFA, SUNALT, MOONALT, MOONSEP, MOONFRAC | 6 |
| twilight  | OBSALT, TRANSPARENCY_GFA, SUNALT, SUNSEP | 4 |

---

## wandb Integration

All training scripts support optional wandb tracking via the `--wandb` flag. When enabled:

- Scalar metrics (train/loss, val/loss, etc.) are logged every `--log-every` epochs
- Visualization figures are logged every `--viz-every` epochs
- When `--wandb` is not passed, no wandb imports or calls happen

Common wandb flags:
```
--wandb            Enable wandb tracking
--wandb-project    Project name (default varies by model)
--wandb-entity     wandb entity/team
--wandb-tags       Comma-separated tags
--log-every        Log scalar metrics every N epochs (default: 1)
--viz-every        Log visualization figures every N epochs
```
