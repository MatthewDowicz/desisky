# SPDX-FileCopyrightText: 2025-present MatthewDowicz <mjdowicz@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Loss functions for VAE training with InfoVAE-MMD objective."""

from __future__ import annotations
from typing import Dict

import jax
import jax.numpy as jnp
import optax
import equinox as eqx


def default_kernel_sigma(latent_dim: int) -> float:
    """
    Compute heuristic bandwidth for the RBF kernel.

    Uses the heuristic σ = √(2 / d) recommended in the InfoVAE literature
    for Gaussian-to-Gaussian MMD estimation.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the latent space.

    Returns
    -------
    sigma : float
        Recommended kernel bandwidth.

    Examples
    --------
    >>> sigma = default_kernel_sigma(latent_dim=8)
    >>> print(f"Kernel bandwidth: {sigma:.4f}")
    Kernel bandwidth: 0.5000
    """
    return (2.0 / float(latent_dim)) ** 0.5


def _rbf_kernel(
    x: jnp.ndarray,
    y: jnp.ndarray,
    *,
    sigma: float = 1.0
) -> jnp.ndarray:
    """
    Compute isotropic RBF (Gaussian) kernel between batches.

    Computes k(x, y) = exp(-‖x-y‖² / (2σ²)) for all pairs in batches x and y.
    This is the standard Gaussian/RBF kernel used in MMD computation.

    Parameters
    ----------
    x : jnp.ndarray
        First batch of vectors. Shape: (n, d).
    y : jnp.ndarray
        Second batch of vectors. Shape: (m, d).
    sigma : float, default 1.0
        Kernel bandwidth parameter.

    Returns
    -------
    kernel_matrix : jnp.ndarray
        Kernel evaluations between all pairs. Shape: (n, m).

    Notes
    -----
    Uses the efficient expansion: ‖x-y‖² = ‖x‖² + ‖y‖² - 2⟨x,y⟩
    to avoid explicit pairwise distance computation.
    """
    # Compute squared norms: shape (n, 1) and (m, 1)
    x2 = jnp.sum(x * x, axis=1, keepdims=True)
    y2 = jnp.sum(y * y, axis=1, keepdims=True)

    # Compute cross term: shape (n, m)
    cross = jnp.dot(x, y.T)

    # Compute squared distances: ‖x‖² + ‖y‖² - 2⟨x,y⟩
    dist2 = x2 + y2.T - 2.0 * cross

    # Apply Gaussian kernel
    k = jnp.exp(-dist2 / (2.0 * sigma ** 2))
    return k


def mmd_rbf_biased(
    x: jnp.ndarray,
    y: jnp.ndarray,
    *,
    sigma: float = 1.0
) -> jnp.ndarray:
    """
    Compute biased MMD² estimate using RBF kernel.

    Maximum Mean Discrepancy (MMD) is a distance metric between distributions
    based on kernel embeddings. This computes the biased U-statistic estimator:

        MMD²(P, Q) ≈ E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]

    where x ~ P, y ~ Q, and k is the RBF kernel.

    Parameters
    ----------
    x : jnp.ndarray
        Samples from first distribution. Shape: (n, d).
    y : jnp.ndarray
        Samples from second distribution. Shape: (m, d).
    sigma : float, default 1.0
        RBF kernel bandwidth.

    Returns
    -------
    mmd_squared : jnp.ndarray
        Scalar MMD² estimate (always ≥ 0 in theory, may be slightly negative
        due to finite sampling).

    References
    ----------
    Gretton et al. "A Kernel Two-Sample Test" (JMLR 2012)
    Zhao et al. "InfoVAE: Balancing Learning and Inference" (AAAI 2019)

    Examples
    --------
    >>> import jax.random as jr
    >>> x = jr.normal(jr.PRNGKey(0), (100, 8))
    >>> y = jr.normal(jr.PRNGKey(1), (100, 8))
    >>> mmd = mmd_rbf_biased(x, y, sigma=0.5)
    """
    xx = _rbf_kernel(x, x, sigma=sigma).mean()
    yy = _rbf_kernel(y, y, sigma=sigma).mean()
    xy = _rbf_kernel(x, y, sigma=sigma).mean()
    return xx + yy - 2 * xy


def vae_loss_infovae(
    model: eqx.Module,
    x: jnp.ndarray,
    key: jax.random.PRNGKey,
    *,
    beta: float = 1.0,
    lam: float = 10.0,
    kernel_sigma: float | str = "auto"
) -> tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    InfoVAE-MMD objective for training a VAE.

    Computes the InfoVAE loss with MMD regularization:

        L = E[‖x - x̂‖²] + β·KL[q(z|x) ‖ p(z)] + (λ-β)·MMD[q(z) , p(z)]

    where:
    - Reconstruction term: MSE between input and output
    - KL term: KL divergence between posterior and standard normal prior
    - MMD term: Maximum Mean Discrepancy between aggregated posterior and prior

    Parameters
    ----------
    model : eqx.Module
        VAE model with __call__ method that returns dict with keys:
        'output', 'mean', 'logvar', 'latent'.
    x : jnp.ndarray
        Batch of input spectra. Shape: (batch_size, in_channels).
    key : jax.random.PRNGKey
        Random key for sampling latent vectors.
    beta : float, default 1.0
        Weight for KL divergence term (corresponds to 1-α in InfoVAE paper).
        Lower values relax the constraint on matching the prior.
    lam : float, default 10.0
        Total weight for latent regularization (λ in InfoVAE paper).
        The MMD term gets weight (λ-β).
    kernel_sigma : float or "auto", default "auto"
        Bandwidth for RBF kernel in MMD computation.
        If "auto", uses heuristic σ = √(2/latent_dim).

    Returns
    -------
    total_loss : jnp.ndarray
        Scalar total loss value.
    aux : dict
        Dictionary containing loss components:
        - 'recon': Reconstruction loss (MSE)
        - 'kl_weighted': β * KL divergence
        - 'mmd_weighted': (λ-β) * MMD
        - 'loss_z': Total latent regularization (KL + MMD terms)

    Notes
    -----
    The clipping of logvar and latent values is critical for numerical stability:
    - logvar clipping prevents exp(logvar) from exploding during sampling
    - latent clipping prevents MMD computation from becoming unstable

    These safeguards protect both the forward pass and gradient computation.

    References
    ----------
    Zhao et al. "InfoVAE: Balancing Learning and Inference in Variational
    Autoencoders" (AAAI 2019)

    Examples
    --------
    >>> from desisky.models.vae import make_SkyVAE
    >>> import jax.random as jr
    >>>
    >>> model = make_SkyVAE(in_channels=7781, latent_dim=8, key=jr.PRNGKey(0))
    >>> x = jr.normal(jr.PRNGKey(1), (64, 7781))
    >>>
    >>> loss, aux = vae_loss_infovae(model, x, jr.PRNGKey(2), beta=1e-3, lam=4)
    >>> print(f"Total: {loss:.4f}, Recon: {aux['recon']:.4f}")
    """
    # Determine kernel bandwidth
    if kernel_sigma is None or kernel_sigma == "auto":
        kernel_sigma = default_kernel_sigma(model.latent_dim)

    # Forward pass through VAE
    out = model(x, key)

    # 1. Reconstruction loss (MSE)
    recon_loss = jnp.mean(optax.l2_loss(out["output"], x))

    # 2. KL divergence: KL[q(z|x) ‖ N(0,I)]
    # Formula: -0.5 * Σ(1 + log(σ²) - μ² - σ²)
    mean, logvar = out["mean"], out["logvar"]

    # Clip logvar to prevent gradient explosion
    # Without this, exp(logvar) can become huge → NaN gradients
    logvar = jnp.clip(logvar, -10.0, 10.0)

    kl_loss = -0.5 * jnp.mean(1 + logvar - jnp.square(mean) - jnp.exp(logvar))

    # 3. MMD: Maximum Mean Discrepancy between q(z) and p(z) = N(0,I)
    z = out["latent"]  # (batch_size, latent_dim)

    # Clip latent values to prevent MMD numerical instability
    # Large z values (>20) cause RBF kernel computation to explode
    z = jnp.clip(z, -10.0, 10.0)

    # Sample from prior with same shape
    z_prior = jax.random.normal(key, shape=z.shape)

    mmd_loss = mmd_rbf_biased(z, z_prior, sigma=kernel_sigma)

    # Total InfoVAE loss
    total = recon_loss + beta * kl_loss + (lam - beta) * mmd_loss

    # Auxiliary outputs for monitoring
    aux = {
        'recon': recon_loss,
        'kl_weighted': beta * kl_loss,
        'mmd_weighted': (lam - beta) * mmd_loss,
        'loss_z': beta * kl_loss + (lam - beta) * mmd_loss
    }

    return total, aux
