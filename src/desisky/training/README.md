# Training Module

Training infrastructure for all desisky models: broadband MLP, VAE, and latent diffusion models.

## Trainers

### BroadbandTrainer

Trains the broadband MLP to predict V, g, r, z surface brightness from observational metadata.

```python
from desisky.training import BroadbandTrainer, TrainingConfig

config = TrainingConfig(
    epochs=500,
    learning_rate=1e-4,
    loss="huber",          # "l2" or "huber"
    huber_delta=0.25,
    save_best=True,
    run_name="broadband",
)

trainer = BroadbandTrainer(model, config)
model, history = trainer.train(train_loader, test_loader)
```

### VAETrainer

Trains the VAE with InfoVAE-MMD objective for spectral compression.

```python
from desisky.training import VAETrainer, VAETrainingConfig

config = VAETrainingConfig(
    epochs=100,
    learning_rate=1e-4,
    beta=1e-3,           # KL weight
    lam=4.0,             # Total regularization (MMD weight = lam - beta)
    kernel_sigma="auto", # RBF bandwidth
)

trainer = VAETrainer(model, config)
model, history = trainer.train(train_loader, test_loader)
```

See [docs/VAE_TRAINING.md](../../../docs/VAE_TRAINING.md) for the full guide on hyperparameter tuning and the InfoVAE-MMD objective.

### LatentDiffusionTrainer

Trains the LDM with EDM framework (Karras et al. 2022), including EMA model tracking and conditioning normalization.

```python
from desisky.training import (
    LatentDiffusionTrainer, LDMTrainingConfig,
    fit_conditioning_scaler, normalize_conditioning,
)
from desisky.models.ldm import compute_sigma_data

sigma_data = compute_sigma_data(latent_train)
scaler = fit_conditioning_scaler(cond_train, conditioning_features)

config = LDMTrainingConfig(
    epochs=200,
    learning_rate=1e-4,
    meta_dim=8,
    sigma_data=sigma_data,
    ema_decay=0.9999,
    early_stop_on_ema=True,
    conditioning_scaler=scaler,
)

trainer = LatentDiffusionTrainer(model, config)
model, ema_model, history = trainer.train(train_loader, val_loader)
```

## W&B Integration

All trainers accept an optional `WandbConfig` for experiment tracking:

```python
from desisky.training import WandbConfig

wandb_config = WandbConfig(project="desisky", tags=["experiment"])
trainer = VAETrainer(model, config, wandb_config=wandb_config)
```

And an optional `on_epoch_end` callback for custom visualization:

```python
def on_epoch_end(model, history, epoch):
    # Log figures, compute metrics, etc.
    ...

trainer = VAETrainer(model, config, on_epoch_end=on_epoch_end)
```

## Dataset Utilities

```python
from desisky.training import SkyBrightnessDataset, NumpyLoader

# PyTorch Dataset that returns numpy arrays
dataset = SkyBrightnessDataset(metadata, flux, input_features)

# DataLoader that collates into numpy (not torch tensors)
loader = NumpyLoader(dataset, batch_size=64, shuffle=True)
```

## Loss Functions

- `loss_l2(pred, target)` — L2 regression loss
- `loss_huber(pred, target, delta)` — Huber loss
- `vae_loss_infovae(recon, mean, logvar, latents, beta, lam)` — InfoVAE-MMD
- `mmd_rbf_biased(z_q, z_p, sigma)` — MMD with RBF kernel

## See Also

- [Examples](../../../examples/) — Training notebooks (01, 04, 06, 07, 08)
- [docs/VAE_TRAINING.md](../../../docs/VAE_TRAINING.md) — Detailed VAE training guide
