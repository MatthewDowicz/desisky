# Setting Up desisky on NERSC Perlmutter

desisky does **not** require the DESI conda environment. The only connection to DESI is pointing to the data directory.

## Environment Setup

**Important**: Install to `$PSCRATCH`, not your home directory. NERSC home directories have a 40 GB quota, and conda environments with JAX + CUDA easily exceed that.

### Step 1: Create the environment on scratch

```bash
module load conda
conda create --prefix $PSCRATCH/envs/desisky python=3.11 -y
conda activate $PSCRATCH/envs/desisky
```

### Step 2: Install JAX with CUDA and desisky

```bash
pip install jax[cuda12]
cd $PSCRATCH/desisky        # or wherever your clone lives
pip install -e ".[all]"
```

### Step 3: Redirect model/data cache to scratch

By default, desisky caches downloaded model weights to `~/.cache/desisky/`, which will also exceed your home quota. Set the `DESISKY_CACHE_DIR` environment variable to redirect:

**In your shell (recommended for CLI / batch jobs):**
```bash
export DESISKY_CACHE_DIR=$PSCRATCH/desisky_cache
```

**In a notebook (add near the top, before any `load_model` calls):**
```python
import os
os.environ["DESISKY_CACHE_DIR"] = "/pscratch/sd/<first_letter>/<username>/desisky_cache"
```

## Interactive GPU Session

```bash
# Request a single GPU node for interactive work (1 hour)
salloc -N 1 -G 1 -C gpu -t 60 -q interactive -A <your_account>
```

## Example: Generate spectra on GPU

```bash
# CLI
export DESISKY_CACHE_DIR=$PSCRATCH/desisky_cache
desisky-infer-ldm --variant dark --n-samples 1000 --output $PSCRATCH/dark_spectra.npz
```

```python
# Python
import os
os.environ["DESISKY_CACHE_DIR"] = "/pscratch/sd/m/mdowicz/desisky_cache"

from desisky.io import load_model
from desisky.inference import LatentDiffusionSampler
import jax.random as jr
import jax.numpy as jnp

vae, _ = load_model("vae")
ldm, meta = load_model("ldm_dark")

sampler = LatentDiffusionSampler(
    ldm_model=ldm, vae_model=vae,
    sigma_data=meta["training"]["sigma_data"],
    conditioning_scaler=meta["training"]["conditioning_scaler"],
    num_steps=250,
)

conditioning = jnp.array([[70.0, 0.95, -40.0, 120.0, 180.0, 45.0, 90.0, 60.0]])
spectra = sampler.sample(key=jr.PRNGKey(42), conditioning=conditioning, guidance_scale=1.0)
print(f"Generated: {spectra.shape}")
```

## Troubleshooting

### Home directory already over quota

If your home directory is already over quota, conda can't even remove environments. Force-remove by deleting the directory:

```bash
rm -rf ~/.conda/envs/desisky
```

Move conda's package cache to scratch:

```bash
mv ~/.conda/pkgs $PSCRATCH/conda_pkgs && ln -s $PSCRATCH/conda_pkgs ~/.conda/pkgs
```

Clear pip / HuggingFace caches:

```bash
conda clean --all -y
pip cache purge
rm -rf ~/.cache/huggingface
```

## CPU vs GPU Performance

See [BENCHMARKS.md](BENCHMARKS.md) for full results. Key numbers:

| Operation | CPU | GPU (RTX 3090) | Speedup |
|-----------|-----|----------------|---------|
| LDM inference (100 spectra) | ~4,500 ms | ~130 ms | ~35x |
| VAE encode+decode (1,000 spectra) | ~27 ms | ~1.8 ms | ~15x |
| Broadband MLP (1,000 rows) | ~0.06 ms | ~0.06 ms | ~1x |

For practical inference (thousands of spectra), GPU is clearly worth it.
