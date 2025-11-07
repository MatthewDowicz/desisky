# Training Module

This module provides production-quality utilities for training broadband sky brightness models.

## Features

- **Dataset & DataLoaders**: PyTorch-compatible datasets that return NumPy/JAX arrays
- **Loss Functions**: L2 and Huber loss implementations optimized for sky magnitude prediction
- **Training Framework**: High-level trainer with automatic checkpointing and metric tracking
- **Visualization**: Built-in support for loss curves and model diagnostics

## Quick Start

```python
from desisky.data import SkySpecVAC
from desisky.training import (
    SkyBrightnessDataset,
    NumpyLoader,
    BroadbandTrainer,
    TrainingConfig,
)
import equinox as eqx
import jax
from torch.utils.data import random_split
import torch

# 1. Load data
vac = SkySpecVAC(download=True)
wave, flux, meta = vac.load_moon_contaminated()

# 2. Create dataset
input_features = ['MOONSEP', 'MOONFRAC', 'MOONALT', 'OBSALT',
                  'TRANSPARENCY_GFA', 'ECLIPSE_FRAC']
dataset = SkyBrightnessDataset(meta, flux, input_features)

# 3. Split into train/test
train_set, test_set = random_split(dataset, [0.7, 0.3],
                                    generator=torch.Generator().manual_seed(42))

# 4. Create loaders
train_loader = NumpyLoader(train_set, batch_size=32, shuffle=True)
test_loader = NumpyLoader(test_set, batch_size=32, shuffle=False)

# 5. Create model
model = eqx.nn.MLP(in_size=6, out_size=4, width_size=128, depth=5,
                    key=jax.random.PRNGKey(42))

# 6. Configure training
config = TrainingConfig(
    epochs=500,
    learning_rate=1e-4,
    loss="huber",
    huber_delta=0.25,
    save_best=True,
    run_name="my_broadband_model"
)

# 7. Train!
trainer = BroadbandTrainer(model, config)
model, history = trainer.train(train_loader, test_loader)

print(f"Best test loss: {history.best_test_loss:.4f}")
```

## Dataset

The `SkyBrightnessDataset` wraps DESI sky metadata and flux spectra for training.

### Input Features

Typical input features for moon contamination modeling:
- `MOONSEP`: Moon separation angle (degrees)
- `MOONFRAC`: Moon illumination fraction (0-1)
- `MOONALT`: Moon altitude (degrees)
- `OBSALT`: Observation altitude (degrees)
- `TRANSPARENCY_GFA`: Atmospheric transparency from GFA
- `ECLIPSE_FRAC`: Lunar eclipse umbral coverage (0-1)

### Target Bands

The model predicts 4 sky magnitude bands:
- V-band (Bessell V)
- g-band (DESI/SDSS g)
- r-band (DESI/SDSS r)
- z-band (DESI z)

## Training Configuration

Key configuration options:

```python
config = TrainingConfig(
    epochs=500,              # Number of training epochs
    learning_rate=1e-4,      # Learning rate for Adam optimizer
    loss="huber",            # Loss function: "l2" or "huber"
    huber_delta=0.25,        # Delta for Huber loss
    save_best=True,          # Save best model checkpoint
    save_dir=None,           # Custom save directory (default: ~/.cache/desisky/saved_models/broadband)
    run_name="my_model",     # Name for checkpoint file
    print_every=50,          # Print progress every N epochs
    validate_every=1,        # Validate every N epochs
)
```

## Model Checkpoints

Checkpoints are saved in JSON header + Equinox binary format:

```python
{
    "schema": 1,
    "arch": {
        "in_size": 6,
        "out_size": 4,
        "width_size": 128,
        "depth": 5
    },
    "training": {
        "date": "2025-01-15T10:30:00",
        "epoch": 342,
        "test_loss": 0.0421,
        "train_loss": 0.0398,
        "per_band_rmse": {
            "V": 0.145,
            "g": 0.132,
            "r": 0.128,
            "z": 0.151
        },
        "config": {
            "epochs": 500,
            "learning_rate": 0.0001,
            "loss": "huber",
            "huber_delta": 0.25
        }
    }
}
```

## Loading Saved Models

```python
from desisky.io import load_model

# Load best checkpoint
model, meta = load_model("broadband", path="path/to/checkpoint.eqx")

# Access training metadata
print(f"Trained for {meta['training']['epoch']} epochs")
print(f"Test loss: {meta['training']['test_loss']:.4f}")
print(f"V-band RMSE: {meta['training']['per_band_rmse']['V']:.4f}")
```

## Advanced Usage

### Custom Optimizer

```python
import optax

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(1e-4)
)

trainer = BroadbandTrainer(model, config, optimizer=optimizer)
```

### Custom Loss Function

Extend the loss module for custom objectives:

```python
from desisky.training.losses import loss_func
import jax.numpy as jnp

def custom_loss(pred, targets):
    # Your custom loss here
    return jnp.mean((pred - targets) ** 2)

# Use in training by modifying loss_func in losses.py
```

## See Also

- [Visualization Module](../visualization/README.md) - Plotting utilities
- [Examples](../../examples/) - Full training notebooks
- [Model I/O](../io/) - Checkpoint format specification
