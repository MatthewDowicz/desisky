#!/usr/bin/env python
# SPDX-FileCopyrightText: 2025-present MatthewDowicz <mjdowicz@gmail.com>
#
# SPDX-License-Identifier: MIT

"""
Train a VAE on DESI Sky Spectra
================================

This script trains a Variational Autoencoder (VAE) on DESI sky spectra using
the InfoVAE-MMD objective. The VAE learns to compress 7781-dimensional sky
spectra into a low-dimensional latent representation while maintaining
reconstruction quality.

The trained VAE can be used for:
- Dimensionality reduction and data compression
- Anomaly detection via reconstruction error
- Interpolation in latent space between sky conditions
- As part of a Latent Diffusion Model pipeline

Usage:
------
    python -m desisky.scripts.train_vae [--epochs EPOCHS] [--latent-dim DIM] [--beta BETA] [--lam LAM]

    Optional arguments:
        --epochs EPOCHS         Number of training epochs (default: 100)
        --latent-dim DIM        Latent space dimensionality (default: 8)
        --beta BETA             KL divergence weight (default: 1e-3)
        --lam LAM               Total regularization weight (default: 4.0)
        --learning-rate LR      Learning rate (default: 1e-4)
        --batch-size BS         Batch size (default: 64)
        --run-name NAME         Name for this training run (default: sky_vae_v1)
        --save-dir DIR          Directory to save checkpoints (default: auto)
        --no-save               Don't save checkpoints
        --no-plot               Don't generate training plots
        --seed SEED             Random seed (default: 42)
"""

import argparse
import sys
from pathlib import Path

import jax.random as jr
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, random_split
import torch

from desisky.data import SkySpecVAC
from desisky.models.vae import make_SkyVAE
from desisky.training import (
    VAETrainer,
    VAETrainingConfig,
    NumpyLoader,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a VAE on DESI sky spectra",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model architecture
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=8,
        help="Dimensionality of latent space"
    )

    # Training schedule
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for Adam optimizer"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training"
    )

    # InfoVAE hyperparameters
    parser.add_argument(
        "--beta",
        type=float,
        default=1e-3,
        help="Weight for KL divergence term (lower = better reconstruction)"
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=4.0,
        help="Total weight for latent regularization"
    )
    parser.add_argument(
        "--kernel-sigma",
        type=str,
        default="auto",
        help="RBF kernel bandwidth (auto or float value)"
    )

    # Checkpointing
    parser.add_argument(
        "--run-name",
        type=str,
        default="sky_vae_v1",
        help="Name for this training run"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save checkpoints (default: ~/.cache/desisky/saved_models/vae)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save model checkpoints"
    )

    # Visualization
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Don't generate training plots"
    )
    parser.add_argument(
        "--plot-path",
        type=str,
        default="vae_training_results.png",
        help="Path to save training plots"
    )

    # Other
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=10,
        help="Print progress every N epochs"
    )

    return parser.parse_args()


def main():
    """Run VAE training on DESI sky spectra."""
    args = parse_args()

    print("=" * 80)
    print("DESI Sky VAE Training")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Load Data
    # -------------------------------------------------------------------------
    print("\n[1/6] Loading DESI sky spectra...")

    vac = SkySpecVAC(version='v1.0', download=True)
    wavelength, flux, metadata = vac.load()

    print(f"  ✓ Loaded {flux.shape[0]:,} sky spectra")
    print(f"  ✓ Spectrum shape: {flux.shape[1]:,} wavelength bins")
    print(f"  ✓ Wavelength range: {wavelength.min():.1f} - {wavelength.max():.1f} Å")
    print(f"  ✓ Flux range: {flux.min():.2f} - {flux.max():.2f}")
    print(f"  ✓ Data type: {flux.dtype}")

    # -------------------------------------------------------------------------
    # 2. Create Train/Test Split
    # -------------------------------------------------------------------------
    print("\n[2/6] Creating train/test split...")

    # Ensure flux is float32 for JAX
    flux = flux.astype(np.float32)

    # 90/10 train/test split
    dataset_size = len(flux)
    train_size = int(0.9 * dataset_size)
    test_size = dataset_size - train_size

    # Convert to PyTorch dataset for splitting
    flux_tensor = torch.from_numpy(flux)
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, test_dataset = random_split(
        flux_tensor,
        [train_size, test_size],
        generator=generator
    )

    print(f"  ✓ Training samples: {len(train_dataset):,}")
    print(f"  ✓ Test samples: {len(test_dataset):,}")

    # -------------------------------------------------------------------------
    # 3. Create Data Loaders
    # -------------------------------------------------------------------------
    print("\n[3/6] Creating data loaders...")

    train_loader = NumpyLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    test_loader = NumpyLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    print(f"  ✓ Batch size: {args.batch_size}")
    print(f"  ✓ Training batches: {len(train_loader)}")
    print(f"  ✓ Test batches: {len(test_loader)}")

    # -------------------------------------------------------------------------
    # 4. Initialize VAE Model
    # -------------------------------------------------------------------------
    print("\n[4/6] Initializing VAE model...")

    # Model architecture
    in_channels = flux.shape[1]  # 7781 for DESI

    model = make_SkyVAE(
        in_channels=in_channels,
        latent_dim=args.latent_dim,
        key=jr.PRNGKey(args.seed)
    )

    print(f"  ✓ Input channels: {in_channels}")
    print(f"  ✓ Latent dimension: {args.latent_dim}")
    print(f"  ✓ Compression ratio: {in_channels / args.latent_dim:.1f}x")

    # -------------------------------------------------------------------------
    # 5. Configure Training
    # -------------------------------------------------------------------------
    print("\n[5/6] Configuring training...")

    # Parse kernel_sigma (could be "auto" or a float)
    kernel_sigma = args.kernel_sigma
    if kernel_sigma != "auto":
        try:
            kernel_sigma = float(kernel_sigma)
        except ValueError:
            print(f"Warning: Invalid kernel_sigma '{kernel_sigma}', using 'auto'")
            kernel_sigma = "auto"

    config = VAETrainingConfig(
        # Training schedule
        epochs=args.epochs,
        learning_rate=args.learning_rate,

        # InfoVAE hyperparameters
        beta=args.beta,
        lam=args.lam,
        kernel_sigma=kernel_sigma,

        # Optimization
        clip_gradients=False,

        # Checkpointing
        save_best=not args.no_save,
        save_dir=args.save_dir,
        run_name=args.run_name,

        # Logging
        print_every=args.print_every,
        validate_every=1,
        random_seed=args.seed,
    )

    print(f"  ✓ Epochs: {config.epochs}")
    print(f"  ✓ Learning rate: {config.learning_rate}")
    print(f"  ✓ Beta (KL weight): {config.beta}")
    print(f"  ✓ Lambda (total reg): {config.lam}")
    print(f"  ✓ MMD weight: {config.lam - config.beta}")
    print(f"  ✓ Kernel sigma: {config.kernel_sigma}")

    # -------------------------------------------------------------------------
    # 6. Train VAE
    # -------------------------------------------------------------------------
    print("\n[6/6] Training VAE...")
    print("-" * 80)

    trainer = VAETrainer(model, config)
    trained_model, history = trainer.train(train_loader, test_loader)

    print("-" * 80)
    print("\n✓ Training complete!")
    print(f"  • Best test loss: {history.best_test_loss:.6f} (epoch {history.best_epoch})")
    print(f"  • Final train loss: {history.train_losses[-1]:.6f}")
    print(f"  • Final test loss: {history.test_losses[-1]:.6f}")
    print(f"  • Final reconstruction: {history.test_recon[-1]:.6f}")

    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------
    if not args.no_plot:
        print("\n[Visualization] Creating training plots...")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Total Loss
        ax = axes[0, 0]
        ax.plot(history.train_losses, label='Train', linewidth=2)
        ax.plot(history.test_losses, label='Test', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Total Loss', fontsize=12)
        ax.set_title('Training Progress', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        # Plot 2: Reconstruction Loss
        ax = axes[0, 1]
        ax.plot(history.train_recon, label='Train', linewidth=2)
        ax.plot(history.test_recon, label='Test', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Reconstruction Loss (MSE)', fontsize=12)
        ax.set_title('Reconstruction Quality', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        # Plot 3: Latent Space Regularization
        ax = axes[1, 0]
        ax.plot(history.train_kl, label='KL (weighted)', linewidth=2)
        ax.plot(history.train_mmd, label='MMD (weighted)', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss Component', fontsize=12)
        ax.set_title('Latent Space Regularization', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        # Plot 4: Example Reconstruction
        ax = axes[1, 1]

        # Get a test sample and reconstruct it
        test_sample = flux[train_size:train_size+1]  # First test sample
        import jax.numpy as jnp
        result = trained_model(jnp.array(test_sample), jr.PRNGKey(0))
        reconstructed = np.array(result['output'][0])

        ax.plot(wavelength, test_sample[0], label='Original', linewidth=2, alpha=0.7)
        ax.plot(wavelength, reconstructed, label='Reconstructed', linewidth=2, alpha=0.7, linestyle='--')
        ax.set_xlabel('Wavelength (Å)', fontsize=12)
        ax.set_ylabel('Flux', fontsize=12)
        ax.set_title('Example Reconstruction', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(args.plot_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved training plots to '{args.plot_path}'")

    # -------------------------------------------------------------------------
    # Additional Analysis
    # -------------------------------------------------------------------------
    print("\n[Analysis] Computing reconstruction metrics...")

    # Compute reconstruction error on test set
    test_recon_errors = []
    import jax.numpy as jnp
    for x in test_loader:
        if not isinstance(x, jnp.ndarray):
            x = jnp.asarray(x)
        result = trained_model(x, jr.PRNGKey(0))
        error = jnp.mean((result['output'] - x) ** 2, axis=1)
        test_recon_errors.extend(error)

    test_recon_errors = np.array(test_recon_errors)

    print(f"  • Mean reconstruction error: {test_recon_errors.mean():.6f}")
    print(f"  • Std reconstruction error: {test_recon_errors.std():.6f}")
    print(f"  • Min reconstruction error: {test_recon_errors.min():.6f}")
    print(f"  • Max reconstruction error: {test_recon_errors.max():.6f}")

    # Latent space statistics
    print("\n[Analysis] Analyzing latent space...")

    all_latents = []
    for x in test_loader:
        if not isinstance(x, jnp.ndarray):
            x = jnp.asarray(x)
        result = trained_model(x, jr.PRNGKey(0))
        all_latents.append(np.array(result['latent']))

    all_latents = np.vstack(all_latents)

    print(f"  • Latent mean (per dim): {all_latents.mean(axis=0)}")
    print(f"  • Latent std (per dim): {all_latents.std(axis=0)}")

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)

    if not args.no_save:
        if args.save_dir:
            save_path = Path(args.save_dir) / f"{args.run_name}.eqx"
        else:
            save_path = Path.home() / ".cache" / "desisky" / "saved_models" / "vae" / f"{args.run_name}.eqx"

        print(f"\nModel checkpoint saved to: {save_path}")
        print("\nNext steps:")
        print("  1. Load the trained model:")
        print("     from desisky.io import load")
        print(f"     model, meta = load('{save_path}')")
        print("  2. Use for compression: latent = model.encode(spectrum)")
        print("  3. Reconstruct spectra: reconstructed = model.decode(latent)")
        print("  4. Train a diffusion model on the latent space")
    else:
        print("\nModel was not saved (--no-save flag was used)")

    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during training: {e}")
        raise
