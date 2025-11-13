# VAE Training for DESI Sky Spectra

This guide explains how to train a Variational Autoencoder (VAE) on DESI sky spectra using the `desisky` package.

## Overview

The VAE learns to compress high-dimensional sky spectra (7781 wavelength bins) into a low-dimensional latent representation while maintaining reconstruction quality. The trained VAE can be used for:

- **Dimensionality reduction**: Compress spectra to ~8D latent vectors
- **Data compression**: Store and transmit spectra more efficiently
- **Anomaly detection**: Identify unusual sky conditions via reconstruction error
- **Latent space interpolation**: Generate intermediate sky conditions
- **Generative modeling**: As the encoder/decoder for Latent Diffusion Models

## Quick Start

### Using the Command-Line Script

The easiest way to train a VAE is using the provided script:

```bash
python -m desisky.scripts.train_vae --epochs 100 --beta 1e-3 --lam 4.0
```

This will:
1. Download the DESI sky spectra VAC (if not already cached)
2. Create a 90/10 train/test split
3. Train a VAE with InfoVAE-MMD objective
4. Save the best model checkpoint
5. Generate training curves and example reconstructions

### Using the Python API

For more control, use the Python API directly:

```python
import jax.random as jr
import numpy as np
from torch.utils.data import random_split
import torch

from desisky.data import SkySpecVAC
from desisky.models.vae import make_SkyVAE
from desisky.training import (
    VAETrainer,
    VAETrainingConfig,
    NumpyLoader,
)

# Load data
vac = SkySpecVAC(version='v1.0', download=True)
wavelength, flux, metadata = vac.load()
flux = flux.astype(np.float32)

# Create train/test split
dataset_size = len(flux)
train_size = int(0.9 * dataset_size)
test_size = dataset_size - train_size

flux_tensor = torch.from_numpy(flux)
train_set, test_set = random_split(
    flux_tensor,
    [train_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

# Create data loaders
train_loader = NumpyLoader(train_set, batch_size=64, shuffle=True)
test_loader = NumpyLoader(test_set, batch_size=64, shuffle=False)

# Initialize VAE
model = make_SkyVAE(
    in_channels=7781,
    latent_dim=8,
    key=jr.PRNGKey(42)
)

# Configure training
config = VAETrainingConfig(
    epochs=100,
    learning_rate=1e-4,
    beta=1e-3,
    lam=4.0,
    run_name="my_vae_model"
)

# Train
trainer = VAETrainer(model, config)
trained_model, history = trainer.train(train_loader, test_loader)

print(f"Best test loss: {history.best_test_loss:.6f}")
```

## InfoVAE-MMD Objective

The training uses the **InfoVAE** objective function, which provides better control over the tradeoff between reconstruction quality and latent space regularization compared to standard β-VAE.

### Loss Function

The total loss combines three terms:

```
L = Reconstruction + β·KL + (λ-β)·MMD
```

Where:
- **Reconstruction**: Mean squared error between input and output spectra
- **KL**: KL divergence between posterior q(z|x) and prior p(z) = N(0,I)
- **MMD**: Maximum Mean Discrepancy between aggregated posterior and prior

### Hyperparameter Guide

**β (beta)** - Weight for KL divergence
- Controls how much the encoder distribution matches the prior
- Lower β → better reconstruction, less regularization
- Typical values: 1e-4 to 1e-2
- **Recommended**: 1e-3

**λ (lambda)** - Total latent regularization weight
- Controls overall latent space structure
- Higher λ → more structured latent space
- Typical values: 1.0 to 10.0
- **Recommended**: 4.0

**MMD weight** - Automatically computed as (λ - β)
- Ensures aggregated posterior matches prior distribution
- More robust than KL alone

**Kernel sigma** - RBF kernel bandwidth for MMD
- Set to `"auto"` to use heuristic σ = √(2/latent_dim)
- Can specify explicit float value
- **Recommended**: "auto"

### Common Configurations

```python
# High reconstruction quality
VAETrainingConfig(beta=1e-4, lam=2.0, ...)

# Balanced (recommended)
VAETrainingConfig(beta=1e-3, lam=4.0, ...)

# Strong regularization
VAETrainingConfig(beta=1e-2, lam=10.0, ...)
```

## Training Configuration

The `VAETrainingConfig` dataclass controls all training parameters:

```python
from desisky.training import VAETrainingConfig

config = VAETrainingConfig(
    # Required
    epochs=100,               # Number of training epochs
    learning_rate=1e-4,       # Adam learning rate

    # InfoVAE hyperparameters
    beta=1e-3,                # KL divergence weight
    lam=4.0,                  # Total regularization weight
    kernel_sigma="auto",      # RBF kernel bandwidth

    # Optimization
    clip_gradients=False,     # Enable gradient clipping

    # Checkpointing
    save_best=True,           # Save best model
    save_dir=None,            # Save directory (default: ~/.cache/desisky/saved_models/vae)
    run_name="vae_training",  # Checkpoint filename

    # Logging
    print_every=10,           # Print progress every N epochs
    validate_every=1,         # Run validation every N epochs
    random_seed=42,           # Random seed for reproducibility
)
```

## Training History

The `VAETrainer.train()` method returns a `VAETrainingHistory` object containing:

```python
history = VAETrainingHistory(
    train_losses=[...],      # Total training loss per epoch
    test_losses=[...],       # Total test loss per epoch
    train_recon=[...],       # Training reconstruction loss
    test_recon=[...],        # Test reconstruction loss
    train_kl=[...],          # Training KL divergence (weighted)
    test_kl=[...],           # Test KL divergence (weighted)
    train_mmd=[...],         # Training MMD (weighted)
    test_mmd=[...],          # Test MMD (weighted)
    best_test_loss=0.123,    # Best test loss achieved
    best_epoch=45,           # Epoch of best test loss
)
```

### Visualizing Training

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Total loss
axes[0, 0].plot(history.train_losses, label='Train')
axes[0, 0].plot(history.test_losses, label='Test')
axes[0, 0].set_ylabel('Total Loss')
axes[0, 0].set_yscale('log')
axes[0, 0].legend()

# Reconstruction
axes[0, 1].plot(history.train_recon, label='Train')
axes[0, 1].plot(history.test_recon, label='Test')
axes[0, 1].set_ylabel('Reconstruction (MSE)')
axes[0, 1].set_yscale('log')
axes[0, 1].legend()

# Latent regularization
axes[1, 0].plot(history.train_kl, label='KL')
axes[1, 0].plot(history.train_mmd, label='MMD')
axes[1, 0].set_ylabel('Regularization')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_yscale('log')
axes[1, 0].legend()

plt.tight_layout()
plt.show()
```

## Model Checkpoints

Models are automatically saved in the Equinox format with metadata:

```python
from desisky.io import load

# Load trained model
model, meta = load('~/.cache/desisky/saved_models/vae/my_vae_model.eqx')

# Inspect metadata
print(meta['arch'])           # Architecture: in_channels, latent_dim
print(meta['training'])       # Training info: date, epoch, losses, hyperparams
```

Checkpoint metadata includes:
- Model architecture (in_channels, latent_dim)
- Training date and epoch
- Final train/test losses
- Loss components (recon, KL, MMD)
- Hyperparameters (beta, lambda, learning rate)

## Using Trained Models

### Encoding

Compress spectra to latent vectors:

```python
import jax.numpy as jnp

# Single spectrum
spectrum = jnp.array(flux[0])  # Shape: (7781,)
mean, logvar = model.encode(spectrum)
print(f"Latent mean shape: {mean.shape}")  # (8,)

# Batch of spectra
import jax
spectra_batch = jnp.array(flux[:100])  # Shape: (100, 7781)
means, logvars = jax.vmap(model.encode)(spectra_batch)
print(f"Latent means shape: {means.shape}")  # (100, 8)
```

### Decoding

Reconstruct spectra from latent vectors:

```python
# Reconstruct single spectrum
latent = mean  # Use mean (no sampling)
reconstructed = model.decode(latent)
print(f"Reconstructed shape: {reconstructed.shape}")  # (7781,)

# Batch reconstruction
latents_batch = means
reconstructed_batch = jax.vmap(model.decode)(latents_batch)
print(f"Batch reconstructed shape: {reconstructed_batch.shape}")  # (100, 7781)
```

### Full Forward Pass

Encode, sample, and decode in one step:

```python
import jax.random as jr

# Single spectrum with sampling
result = model(spectrum, jr.PRNGKey(0))
print(result.keys())  # dict_keys(['mean', 'logvar', 'latent', 'output'])

# Batch processing
results = model(spectra_batch, jr.PRNGKey(0))
print(results['latent'].shape)  # (100, 8)
print(results['output'].shape)  # (100, 7781)
```

### Latent Space Interpolation

Generate intermediate sky conditions:

```python
import jax.numpy as jnp

# Encode two different sky conditions
mean1, _ = model.encode(flux[0])
mean2, _ = model.encode(flux[100])

# Interpolate in latent space
alphas = jnp.linspace(0, 1, 10)
interpolated_latents = jnp.outer(1 - alphas, mean1) + jnp.outer(alphas, mean2)

# Decode interpolated latents
interpolated_spectra = jax.vmap(model.decode)(interpolated_latents)
print(f"Interpolated spectra shape: {interpolated_spectra.shape}")  # (10, 7781)
```

## Advanced Topics

### Custom Optimizers

```python
import optax

# SGD with momentum
optimizer = optax.sgd(learning_rate=1e-3, momentum=0.9)
trainer = VAETrainer(model, config, optimizer=optimizer)

# Adam with gradient clipping
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(1e-4)
)
trainer = VAETrainer(model, config, optimizer=optimizer)
```

### Different Latent Dimensions

```python
# Smaller latent space (more compression)
model_4d = make_SkyVAE(in_channels=7781, latent_dim=4, key=jr.PRNGKey(42))

# Larger latent space (higher fidelity)
model_16d = make_SkyVAE(in_channels=7781, latent_dim=16, key=jr.PRNGKey(42))
```

### Reconstruction Quality Metrics

```python
import jax.numpy as jnp

# Compute reconstruction error
def reconstruction_error(model, spectra):
    """Compute per-spectrum reconstruction MSE."""
    reconstructed = jax.vmap(lambda x: model(x, jr.PRNGKey(0))['output'])(spectra)
    mse = jnp.mean((spectra - reconstructed) ** 2, axis=1)
    return mse

test_spectra = jnp.array(flux[train_size:])
errors = reconstruction_error(trained_model, test_spectra)

print(f"Mean reconstruction error: {errors.mean():.6f}")
print(f"Std reconstruction error: {errors.std():.6f}")
print(f"Max reconstruction error: {errors.max():.6f}")
```

## Best Practices

1. **Data Preprocessing**: Ensure flux is `float32` for JAX compatibility
2. **Batch Size**: 64 works well for most cases; increase if you have memory
3. **Learning Rate**: 1e-4 is a good default; reduce if training is unstable
4. **Validation**: Monitor test reconstruction loss to avoid overfitting
5. **Hyperparameters**: Start with β=1e-3, λ=4.0 and adjust based on reconstruction quality
6. **Random Seeds**: Set for reproducibility across runs
7. **Checkpointing**: Always save best model for later use

## Troubleshooting

**High reconstruction error**
- Decrease β (e.g., from 1e-3 to 1e-4)
- Decrease λ (e.g., from 4.0 to 2.0)
- Increase latent_dim (e.g., from 8 to 16)
- Train for more epochs

**NaN or Inf in loss**
- The code includes clipping for stability, but if you still see NaNs:
- Reduce learning rate
- Enable gradient clipping: `clip_gradients=True`
- Check input data for extreme values

**Poor latent space structure**
- Increase β and λ for stronger regularization
- Train for more epochs to converge
- Ensure kernel_sigma="auto" is being used

**Training too slow**
- Reduce batch size if memory-constrained
- Use GPU if available
- Decrease model size (fewer layers/smaller hidden dims)

## References

- **InfoVAE Paper**: Zhao et al., "InfoVAE: Balancing Learning and Inference in Variational Autoencoders" (AAAI 2019)
- **VAE Tutorial**: Kingma & Welling, "Auto-Encoding Variational Bayes" (ICLR 2014)
- **MMD**: Gretton et al., "A Kernel Two-Sample Test" (JMLR 2012)

## API Reference

See the full API documentation:
- `desisky.training.VAETrainer`: Main training class
- `desisky.training.VAETrainingConfig`: Training configuration
- `desisky.training.vae_loss_infovae`: InfoVAE-MMD loss function
- `desisky.models.vae.SkyVAE`: VAE model architecture
- `desisky.models.vae.make_SkyVAE`: Model constructor
