"""
Sampling algorithms for Latent Diffusion Models using EDM (Karras et al. 2022).

This module provides production-ready sampling functions for generating sky spectra
using trained latent diffusion models. Two sampling modes are available:

- **Deterministic (default)**: Heun ODE solver — same seed + same conditioning
  always produces identical output.
- **Stochastic**: Langevin-corrected reverse SDE (Algorithm 2 from Karras et al.
  2022) — injects noise at each step for diverse ensemble generation.

Both use:

- **Karras sigma schedule**: Geometrically spaced noise levels
- **Heun solver**: Second-order method for better accuracy

Examples
--------
Basic usage:

>>> from desisky.inference import LatentDiffusionSampler
>>> from desisky.io import load_builtin
>>>
>>> # Load models
>>> ldm, meta = load_builtin("ldm_dark")
>>> vae, _ = load_builtin("vae")
>>>
>>> # Create sampler (sigma_data and scaler from training metadata)
>>> sampler = LatentDiffusionSampler(
...     ldm, vae,
...     sigma_data=meta["training"]["sigma_data"],
...     conditioning_scaler=meta["training"]["conditioning_scaler"],
... )
>>>
>>> # Generate samples (raw conditioning is auto-normalized)
>>> import jax.random as jr
>>> conditioning = jnp.array([[60.0, 0.9, -30.0, 150.0, 45.0, 10.0, 120.0, 5.0]])
>>> spectra = sampler.sample(
...     key=jr.PRNGKey(0),
...     conditioning=conditioning,
...     n_samples=10,
...     guidance_scale=2.0
... )
>>> spectra.shape
(10, 7781)
"""

from typing import Tuple, Optional
import jax
import jax.numpy as jnp
import equinox as eqx
from dataclasses import dataclass

from desisky.models.ldm import (
    edm_denoiser,
    EDM_SIGMA_MIN,
    EDM_SIGMA_MAX,
)


# =========================================================================
# EDM Sigma Schedule (Karras et al. 2022)
# =========================================================================

def get_sigmas_karras(
    n_steps: int,
    sigma_min: float = EDM_SIGMA_MIN,
    sigma_max: float = EDM_SIGMA_MAX,
    rho: float = 7.0,
) -> jnp.ndarray:
    """
    Karras et al. 2022 sigma schedule (Eq. 5).

    Generates geometrically spaced sigma values from sigma_max down to 0,
    with the spacing controlled by rho.

    Parameters
    ----------
    n_steps : int
        Number of sampling steps.
    sigma_min : float
        Minimum sigma value.
    sigma_max : float
        Maximum sigma value.
    rho : float
        Controls spacing (higher = more steps at low noise).

    Returns
    -------
    jnp.ndarray
        Sigma schedule of shape (n_steps + 1,), ending with 0.
    """
    ramp = jnp.linspace(0, 1, n_steps)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return jnp.concatenate([sigmas, jnp.array([0.0])])


# =========================================================================
# Classifier-Free Guidance for EDM
# =========================================================================

def guided_denoiser(
    model: eqx.Module,
    x_noisy: jnp.ndarray,
    sigma: jnp.ndarray,
    cond: jnp.ndarray,
    sigma_data: float,
    guidance_scale: float = 2.0,
) -> jnp.ndarray:
    """
    Classifier-free guidance for the EDM preconditioned denoiser.

    D_guided = D_uncond + w * (D_cond - D_uncond)

    Parameters
    ----------
    model : eqx.Module
        Raw UNet network F_theta.
    x_noisy : jnp.ndarray
        Noisy latent, shape (C, L) -- single sample.
    sigma : jnp.ndarray
        Noise level (scalar).
    cond : jnp.ndarray
        Conditioning metadata, shape (meta_dim,).
    sigma_data : float
        Standard deviation of training data.
    guidance_scale : float
        Guidance strength:
        - w=0: purely unconditional
        - w=1: standard conditional
        - w>1: amplified conditioning (typical: 2-4)

    Returns
    -------
    jnp.ndarray
        Guided denoised estimate, shape (C, L).
    """
    D_uncond = edm_denoiser(
        model, x_noisy, sigma, jnp.zeros_like(cond), sigma_data, None, 0.0
    )
    D_cond = edm_denoiser(
        model, x_noisy, sigma, cond, sigma_data, None, 0.0
    )
    return D_uncond + guidance_scale * (D_cond - D_uncond)


# =========================================================================
# EDM Sampling with Heun Solver
# =========================================================================

@eqx.filter_jit
def sample_edm(
    key: jax.random.PRNGKey,
    model: eqx.Module,
    cond_vec: jnp.ndarray,
    n_sample: int,
    size: Tuple[int, int],
    sigma_data: float,
    guidance_scale: float = 1.0,
    num_steps: int = 100,
    sigma_min: float = EDM_SIGMA_MIN,
    sigma_max: float = EDM_SIGMA_MAX,
    rho: float = 7.0,
) -> jnp.ndarray:
    """
    EDM sampling with 2nd-order Heun solver.

    Solves the probability flow ODE: dx/d_sigma = (x - D(x; sigma)) / sigma
    using the Karras sigma schedule and Heun's method for second-order accuracy.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for initial noise.
    model : eqx.Module
        Raw UNet network F_theta.
    cond_vec : jnp.ndarray
        Conditioning vectors, shape (n_sample, meta_dim).
    n_sample : int
        Number of samples.
    size : tuple of int
        Sample size (channels, latent_dim), e.g. (1, 8).
    sigma_data : float
        Standard deviation of training data.
    guidance_scale : float
        CFG guidance weight.
    num_steps : int
        Number of solver steps.
    sigma_min : float
        Minimum sigma.
    sigma_max : float
        Maximum sigma.
    rho : float
        Sigma schedule spacing parameter.

    Returns
    -------
    jnp.ndarray
        Generated latent samples, shape (n_sample, channels, latent_dim).
    """
    sigmas = get_sigmas_karras(num_steps, sigma_min, sigma_max, rho)

    # Initialize from noise at sigma_max
    x_i = jax.random.normal(key, (n_sample, *size)) * sigmas[0]

    # Sigma pairs for stepping: (sigma_cur, sigma_next)
    sigma_pairs = jnp.stack([sigmas[:-1], sigmas[1:]], axis=1)

    # Vectorized guided denoiser over batch
    guided_vmap = jax.vmap(
        guided_denoiser,
        in_axes=(None, 0, None, 0, None, None),
    )

    def _heun_step(x_i, sigma_pair):
        sigma_cur, sigma_next = sigma_pair[0], sigma_pair[1]

        def do_step(x_i):
            # Euler predictor
            D_cur = guided_vmap(
                model, x_i, sigma_cur, cond_vec, sigma_data, guidance_scale
            )
            d_cur = (x_i - D_cur) / sigma_cur
            x_euler = x_i + (sigma_next - sigma_cur) * d_cur

            # Heun corrector (if not at final step, i.e. sigma_next > 0)
            def heun_correct(x_euler):
                D_next = guided_vmap(
                    model, x_euler, sigma_next, cond_vec,
                    sigma_data, guidance_scale
                )
                d_next = (x_euler - D_next) / sigma_next
                d_avg = 0.5 * (d_cur + d_next)
                return x_i + (sigma_next - sigma_cur) * d_avg

            return jax.lax.cond(
                sigma_next > 0, heun_correct, lambda x: x_euler, x_euler
            )

        # Skip step if sigma_cur is effectively zero
        return jax.lax.cond(
            sigma_cur > 1e-8, do_step, lambda x: x, x_i
        ), None

    x_0, _ = jax.lax.scan(_heun_step, x_i, sigma_pairs)
    return x_0


# =========================================================================
# EDM Stochastic Sampling (Algorithm 2, Karras et al. 2022)
# =========================================================================

@eqx.filter_jit
def sample_edm_stochastic(
    key: jax.random.PRNGKey,
    model: eqx.Module,
    cond_vec: jnp.ndarray,
    n_sample: int,
    size: Tuple[int, int],
    sigma_data: float,
    guidance_scale: float = 1.0,
    num_steps: int = 100,
    sigma_min: float = EDM_SIGMA_MIN,
    sigma_max: float = EDM_SIGMA_MAX,
    rho: float = 7.0,
    S_churn: float = 40.0,
    S_tmin: float = 0.05,
    S_tmax: float = 50.0,
    S_noise: float = 1.003,
) -> jnp.ndarray:
    """
    EDM stochastic sampling with Langevin-like noise injection (Algorithm 2).

    Combines the 2nd-order Heun ODE integrator with explicit Langevin-like
    "churn" — adding and removing noise at each step. This produces different
    outputs even with the same seed, useful for generating diverse ensembles.

    At each step i:
      1. Increase noise level from t_i to t_hat_i by injecting fresh noise
      2. Denoise from t_hat_i to t_{i+1} using Heun's method

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for initial noise and stochastic injection.
    model : eqx.Module
        Raw UNet network F_theta.
    cond_vec : jnp.ndarray
        Conditioning vectors, shape (n_sample, meta_dim).
    n_sample : int
        Number of samples.
    size : tuple of int
        Sample size (channels, latent_dim).
    sigma_data : float
        Standard deviation of training data.
    guidance_scale : float
        CFG guidance weight.
    num_steps : int
        Number of solver steps.
    sigma_min : float
        Minimum sigma.
    sigma_max : float
        Maximum sigma.
    rho : float
        Sigma schedule spacing parameter.
    S_churn : float
        Overall amount of stochasticity. Higher values inject more noise.
    S_tmin : float
        Minimum sigma for enabling stochasticity.
    S_tmax : float
        Maximum sigma for enabling stochasticity.
    S_noise : float
        Noise inflation factor (slightly > 1 counteracts detail loss).

    Returns
    -------
    jnp.ndarray
        Generated latent samples, shape (n_sample, channels, latent_dim).
    """
    sigmas = get_sigmas_karras(num_steps, sigma_min, sigma_max, rho)

    key, init_key = jax.random.split(key)
    x_i = jax.random.normal(init_key, (n_sample, *size)) * sigmas[0]

    sigma_pairs = jnp.stack([sigmas[:-1], sigmas[1:]], axis=1)

    guided_vmap = jax.vmap(
        guided_denoiser,
        in_axes=(None, 0, None, 0, None, None),
    )

    # Upper bound on per-step noise injection (clamped by sqrt(2)-1)
    gamma_max = jnp.minimum(S_churn / num_steps, jnp.sqrt(2.0) - 1.0)

    def _stochastic_step(carry, sigma_pair):
        x_i, step_key = carry
        sigma_cur, sigma_next = sigma_pair[0], sigma_pair[1]

        step_key, noise_key = jax.random.split(step_key)

        def do_step(args):
            x_i, noise_key = args

            # Only inject noise when sigma_cur is in [S_tmin, S_tmax]
            in_range = (sigma_cur >= S_tmin) & (sigma_cur <= S_tmax)
            gamma_i = jnp.where(in_range, gamma_max, 0.0)

            # Temporarily increase noise level
            sigma_hat = sigma_cur + gamma_i * sigma_cur
            eps = jax.random.normal(noise_key, x_i.shape) * S_noise
            x_hat = x_i + jnp.sqrt(jnp.maximum(sigma_hat**2 - sigma_cur**2, 0.0)) * eps

            # Evaluate dx/dt at sigma_hat
            D_hat = guided_vmap(
                model, x_hat, sigma_hat, cond_vec, sigma_data, guidance_scale
            )
            d_cur = (x_hat - D_hat) / jnp.maximum(sigma_hat, 1e-12)

            # Euler step from sigma_hat to sigma_next
            x_euler = x_hat + (sigma_next - sigma_hat) * d_cur

            # Heun corrector (if sigma_next > 0)
            def heun_correct(x_euler):
                D_next = guided_vmap(
                    model, x_euler, sigma_next, cond_vec,
                    sigma_data, guidance_scale
                )
                d_next = (x_euler - D_next) / jnp.maximum(sigma_next, 1e-12)
                d_avg = 0.5 * (d_cur + d_next)
                return x_hat + (sigma_next - sigma_hat) * d_avg

            return jax.lax.cond(
                sigma_next > 0, heun_correct, lambda x: x_euler, x_euler
            )

        x_next = jax.lax.cond(
            sigma_cur > 1e-8,
            do_step,
            lambda args: args[0],
            (x_i, noise_key),
        )

        return (x_next, step_key), None

    (x_0, _), _ = jax.lax.scan(_stochastic_step, (x_i, key), sigma_pairs)
    return x_0


# =========================================================================
# High-Level Sampler Class
# =========================================================================

@dataclass
class SamplerConfig:
    """
    Configuration for EDM latent diffusion sampling.

    Attributes
    ----------
    num_steps : int
        Number of Heun solver steps.
    sigma_min : float
        Minimum sigma for Karras schedule.
    sigma_max : float
        Maximum sigma for Karras schedule.
    rho : float
        Sigma schedule spacing parameter.
    latent_channels : int
        Number of channels in latent space (typically 1).
    latent_dim : int
        Latent dimension (typically 8 for VAE).
    """
    num_steps: int = 100
    sigma_min: float = EDM_SIGMA_MIN
    sigma_max: float = EDM_SIGMA_MAX
    rho: float = 7.0
    latent_channels: int = 1
    latent_dim: int = 8


class LatentDiffusionSampler:
    """
    High-level interface for sampling from latent diffusion models.

    Uses the EDM framework (Karras et al. 2022) with the Heun solver
    and Karras sigma schedule. Handles classifier-free guidance and
    VAE decoding.

    Parameters
    ----------
    ldm_model : eqx.Module
        Trained latent diffusion model (raw UNet F_theta).
    vae_model : eqx.Module
        Trained VAE for decoding latents to spectra.
    sigma_data : float
        Standard deviation of the training latents. This should match
        the value used during training (stored in checkpoint metadata
        under ``meta["training"]["sigma_data"]``).
    conditioning_scaler : dict, optional
        Conditioning feature normalization parameters from training.
        If provided, raw conditioning inputs are automatically
        standardized before sampling. Expected keys: ``"mean"`` and
        ``"scale"`` (lists of per-feature values). Stored in checkpoint
        metadata under ``meta["training"]["conditioning_scaler"]``.
    num_steps : int
        Number of Heun solver steps (more steps = higher quality).
    sigma_min : float
        Minimum sigma for Karras schedule.
    sigma_max : float
        Maximum sigma for Karras schedule.
    rho : float
        Sigma schedule spacing parameter.
    latent_channels : int
        Number of latent channels (typically 1).
    latent_dim : int
        Latent dimension (typically 8).

    Examples
    --------
    >>> from desisky.inference import LatentDiffusionSampler
    >>> from desisky.io import load_builtin
    >>> import jax.random as jr
    >>>
    >>> ldm, meta = load_builtin("ldm_dark")
    >>> vae, _ = load_builtin("vae")
    >>>
    >>> sampler = LatentDiffusionSampler(
    ...     ldm, vae,
    ...     sigma_data=meta["training"]["sigma_data"],
    ...     conditioning_scaler=meta["training"]["conditioning_scaler"],
    ... )
    >>>
    >>> cond = jnp.array([
    ...     [60.0, 0.9, -30.0, 150.0, 45.0, 10.0, 120.0, 5.0],
    ... ])
    >>> spectra = sampler.sample(jr.PRNGKey(0), cond, guidance_scale=2.0)
    >>> spectra.shape
    (1, 7781)
    """

    def __init__(
        self,
        ldm_model: eqx.Module,
        vae_model: eqx.Module,
        sigma_data: float,
        conditioning_scaler: Optional[dict] = None,
        num_steps: int = 100,
        sigma_min: float = EDM_SIGMA_MIN,
        sigma_max: float = EDM_SIGMA_MAX,
        rho: float = 7.0,
        latent_channels: int = 1,
        latent_dim: int = 8,
    ):
        self.ldm = ldm_model
        self.vae = vae_model
        self.sigma_data = sigma_data

        # Conditioning normalization (StandardScaler params from training)
        self._scaler_mean = None
        self._scaler_scale = None
        if conditioning_scaler is not None:
            self._scaler_mean = jnp.array(conditioning_scaler["mean"])
            self._scaler_scale = jnp.array(conditioning_scaler["scale"])

        self.config = SamplerConfig(
            num_steps=num_steps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rho=rho,
            latent_channels=latent_channels,
            latent_dim=latent_dim,
        )

    def sample_latents(
        self,
        key: jax.random.PRNGKey,
        conditioning: jnp.ndarray,
        n_samples: Optional[int] = None,
        guidance_scale: float = 2.0,
        stochastic: bool = False,
        S_churn: float = 40.0,
        S_tmin: float = 0.05,
        S_tmax: float = 50.0,
        S_noise: float = 1.003,
    ) -> jnp.ndarray:
        """
        Sample latent representations from the diffusion model.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random key for sampling.
        conditioning : jnp.ndarray
            Conditioning metadata, shape (n_samples, meta_dim) or (meta_dim,).
        n_samples : int, optional
            Number of samples. If None, inferred from conditioning shape.
        guidance_scale : float
            Classifier-free guidance strength (typical: 1-4).
        stochastic : bool
            If True, use the Langevin-corrected stochastic sampler
            (Algorithm 2, Karras et al. 2022) instead of the deterministic
            Heun ODE solver. Produces different outputs even with the same
            seed, useful for generating diverse ensembles.
        S_churn : float
            Stochastic sampler only. Overall noise injection amount.
        S_tmin : float
            Stochastic sampler only. Minimum sigma for noise injection.
        S_tmax : float
            Stochastic sampler only. Maximum sigma for noise injection.
        S_noise : float
            Stochastic sampler only. Noise inflation factor (slightly > 1
            counteracts detail loss from non-conservative denoisers).

        Returns
        -------
        jnp.ndarray
            Generated latents, shape (n_samples, latent_channels, latent_dim).
        """
        if conditioning.ndim == 1:
            conditioning = conditioning[None, :]

        # Auto-normalize conditioning if scaler was provided
        if self._scaler_mean is not None:
            conditioning = (conditioning - self._scaler_mean) / self._scaler_scale

        if n_samples is None:
            n_samples = conditioning.shape[0]
        elif n_samples != conditioning.shape[0]:
            raise ValueError(
                f"n_samples ({n_samples}) must match conditioning batch size "
                f"({conditioning.shape[0]})"
            )

        latent_size = (self.config.latent_channels, self.config.latent_dim)

        if stochastic:
            latents = sample_edm_stochastic(
                key,
                self.ldm,
                conditioning,
                n_samples,
                latent_size,
                self.sigma_data,
                guidance_scale,
                self.config.num_steps,
                self.config.sigma_min,
                self.config.sigma_max,
                self.config.rho,
                S_churn,
                S_tmin,
                S_tmax,
                S_noise,
            )
        else:
            latents = sample_edm(
                key,
                self.ldm,
                conditioning,
                n_samples,
                latent_size,
                self.sigma_data,
                guidance_scale,
                self.config.num_steps,
                self.config.sigma_min,
                self.config.sigma_max,
                self.config.rho,
            )

        return latents

    def sample(
        self,
        key: jax.random.PRNGKey,
        conditioning: jnp.ndarray,
        n_samples: Optional[int] = None,
        guidance_scale: float = 2.0,
        return_latents: bool = False,
        stochastic: bool = False,
        S_churn: float = 40.0,
        S_tmin: float = 0.05,
        S_tmax: float = 50.0,
        S_noise: float = 1.003,
    ) -> jnp.ndarray | Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Sample sky spectra from the latent diffusion model.

        Generates latent samples via the EDM framework and decodes them
        to full-resolution spectra via the VAE.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random key for sampling.
        conditioning : jnp.ndarray
            Conditioning metadata, shape (n_samples, meta_dim) or (meta_dim,).
        n_samples : int, optional
            Number of samples. If None, inferred from conditioning.
        guidance_scale : float
            Guidance strength (higher = stronger conditioning).
        return_latents : bool
            If True, return (spectra, latents).
        stochastic : bool
            If True, use the Langevin-corrected stochastic sampler
            (Algorithm 2, Karras et al. 2022) instead of the deterministic
            Heun ODE solver.
        S_churn : float
            Stochastic sampler only. Overall noise injection amount.
        S_tmin : float
            Stochastic sampler only. Minimum sigma for noise injection.
        S_tmax : float
            Stochastic sampler only. Maximum sigma for noise injection.
        S_noise : float
            Stochastic sampler only. Noise inflation factor.

        Returns
        -------
        spectra : jnp.ndarray
            Generated sky spectra, shape (n_samples, 7781).
        latents : jnp.ndarray, optional
            Generated latents (if return_latents=True).
        """
        latents = self.sample_latents(
            key, conditioning, n_samples, guidance_scale,
            stochastic, S_churn, S_tmin, S_tmax, S_noise,
        )

        # VAE decoder expects shape (latent_dim,), so squeeze channel dimension
        latents_squeezed = latents.squeeze(1)
        spectra = jax.vmap(self.vae.decode)(latents_squeezed)

        if return_latents:
            return spectra, latents
        return spectra

    def __repr__(self) -> str:
        normalize = self._scaler_mean is not None
        return (
            f"LatentDiffusionSampler(\n"
            f"  num_steps={self.config.num_steps},\n"
            f"  sigma_data={self.sigma_data},\n"
            f"  sigma_range=[{self.config.sigma_min}, {self.config.sigma_max}],\n"
            f"  rho={self.config.rho},\n"
            f"  latent_shape=({self.config.latent_channels}, {self.config.latent_dim}),\n"
            f"  auto_normalize={normalize}\n"
            f")"
        )
