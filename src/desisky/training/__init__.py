# SPDX-FileCopyrightText: 2025-present MatthewDowicz <mjdowicz@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Training utilities for DESI sky brightness models."""

from .dataset import SkyBrightnessDataset, NumpyLoader, numpy_collate, gather_full_data
from .losses import loss_l2, loss_huber, loss_func
from .trainer import BroadbandTrainer, TrainingConfig, TrainingHistory
from .vae_losses import (
    vae_loss_infovae,
    mmd_rbf_biased,
    default_kernel_sigma,
)
from .vae_trainer import VAETrainer, VAETrainingConfig, VAETrainingHistory
from .ldm_trainer import (
    LatentDiffusionTrainer,
    LDMTrainingConfig,
    LDMTrainingHistory,
    edm_loss,
    edm_loss_weight,
    sample_edm_sigma,
    ema_update,
    ema_update_jit,
    fit_conditioning_scaler,
    normalize_conditioning,
)

# Wandb integration (optional -- requires pip install desisky[wandb])
try:
    from .wandb_utils import (
        WandbConfig,
        init_wandb_run,
        log_epoch_metrics,
        log_figure,
        finish_wandb_run,
        eval_per_sigma_losses,
        DEFAULT_SIGMA_LEVELS,
    )
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

__all__ = [
    # Dataset utilities
    "SkyBrightnessDataset",
    "NumpyLoader",
    "numpy_collate",
    "gather_full_data",
    # Broadband training
    "loss_l2",
    "loss_huber",
    "loss_func",
    "BroadbandTrainer",
    "TrainingConfig",
    "TrainingHistory",
    # VAE training
    "vae_loss_infovae",
    "mmd_rbf_biased",
    "default_kernel_sigma",
    "VAETrainer",
    "VAETrainingConfig",
    "VAETrainingHistory",
    # LDM training (EDM)
    "LatentDiffusionTrainer",
    "LDMTrainingConfig",
    "LDMTrainingHistory",
    "edm_loss",
    "edm_loss_weight",
    "sample_edm_sigma",
    "ema_update",
    "ema_update_jit",
    "fit_conditioning_scaler",
    "normalize_conditioning",
]

# Add wandb exports if available
if _WANDB_AVAILABLE:
    __all__ += [
        "WandbConfig",
        "init_wandb_run",
        "log_epoch_metrics",
        "log_figure",
        "finish_wandb_run",
        "eval_per_sigma_losses",
        "DEFAULT_SIGMA_LEVELS",
    ]
