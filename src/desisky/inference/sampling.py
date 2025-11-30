"""
Sampling algorithms for Latent Diffusion Models.

This module provides production-ready sampling functions for generating sky spectra
using trained latent diffusion models. Three sampling algorithms are available:

1. **DDPM** (Ho et al., 2020): Stochastic ancestral sampling
2. **DDIM** (Song et al., 2020): Deterministic sampling with η=0
3. **Heun** (Karras et al., 2022): Second-order probability-flow ODE solver (recommended)

Examples
--------
Basic usage:

>>> from desisky.inference import LatentDiffusionSampler
>>> from desisky.io import load_builtin
>>>
>>> # Load models
>>> ldm, _ = load_builtin("ldm_dark")
>>> vae, _ = load_builtin("vae")
>>>
>>> # Create sampler
>>> sampler = LatentDiffusionSampler(ldm, vae, method="heun")
>>>
>>> # Generate samples
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

Advanced usage with custom parameters:

>>> sampler = LatentDiffusionSampler(
...     ldm,
...     vae,
...     method="ddim",
...     num_steps=50,
...     latent_channels=1,
...     latent_dim=8
... )
>>> spectra = sampler.sample(
...     key=jr.PRNGKey(42),
...     conditioning=my_conditions,
...     n_samples=100,
...     guidance_scale=3.0,
...     return_latents=False
... )
"""

from typing import Literal, Tuple, Optional, Dict, Any
import jax
import jax.numpy as jnp
import equinox as eqx
from dataclasses import dataclass


# =========================================================================
# Noise Schedules
# =========================================================================

def cosine_beta_schedule(T: int, s: float = 0.008) -> Dict[str, jnp.ndarray]:
    """
    Cosine noise schedule from Nichol & Dhariwal (2021).

    Provides smoother noise progression compared to linear schedules.

    Parameters
    ----------
    T : int
        Total number of timesteps
    s : float
        Small offset to prevent singularity at t=0 (default: 0.008 from paper)

    Returns
    -------
    dict
        Dictionary containing all schedule components:
        - beta_t: noise variance at each step
        - alpha_t: 1 - beta_t
        - alphabar_t: cumulative product of alphas
        - sqrtab: sqrt(alphabar_t)
        - sqrtmab: sqrt(1 - alphabar_t)
        - oneover_sqrta: 1 / sqrt(alpha_t)
        - sqrt_beta_t: sqrt(beta_t)
        - mab_over_sqrtmab: (1 - alpha_t) / sqrt(1 - alphabar_t)
        - sqrt_posterior_var: sqrt of posterior variance for DDPM
    """
    t = jnp.linspace(0, T, T + 1, dtype=jnp.float32)
    f = lambda t_: jnp.cos((t_ / T + s) / (1 + s) * jnp.pi / 2) ** 2
    alphabar_t = f(t) / f(0)

    beta_t = 1 - alphabar_t[1:] / alphabar_t[:-1]
    beta_t = jnp.clip(beta_t, a_min=1e-5, a_max=0.999)
    beta_t = jnp.concatenate([jnp.array([1e-5]), beta_t])

    alpha_t = 1 - beta_t
    log_alpha_t = jnp.log(alpha_t)
    alphabar_t = jnp.exp(jnp.cumsum(log_alpha_t))

    sqrtab = jnp.sqrt(alphabar_t)
    sqrtmab = jnp.sqrt(1 - alphabar_t)
    oneover_sqrta = 1 / jnp.sqrt(alpha_t)
    sqrt_beta_t = jnp.sqrt(beta_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    # Posterior variance for DDPM
    alphabar_tm1 = jnp.concatenate([jnp.array([1.0], dtype=jnp.float32),
                                    alphabar_t[:-1]])
    posterior_var = beta_t * (1 - alphabar_tm1) / (1 - alphabar_t)
    sqrt_posterior_var = jnp.sqrt(jnp.clip(posterior_var, 1e-20, 1.0))

    return {
        "beta_t": beta_t,
        "alpha_t": alpha_t,
        "oneover_sqrta": oneover_sqrta,
        "sqrt_beta_t": sqrt_beta_t,
        "alphabar_t": alphabar_t,
        "sqrtab": sqrtab,
        "sqrtmab": sqrtmab,
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,
        "sqrt_posterior_var": sqrt_posterior_var
    }


# =========================================================================
# Classifier-Free Guidance
# =========================================================================

def guided_denoising_step(
    model: eqx.Module,
    x_t: jnp.ndarray,
    t: jnp.ndarray,
    cond: jnp.ndarray,
    guidance_scale: float = 2.0
) -> jnp.ndarray:
    """
    Compute guided noise prediction using classifier-free guidance.

    Combines unconditional and conditional predictions:
        ε̂ = ε_uncond + w * (ε_cond − ε_uncond)

    Parameters
    ----------
    model : eqx.Module
        Diffusion model (UNet)
    x_t : jnp.ndarray
        Noisy latent at timestep t, shape (C, L)
    t : jnp.ndarray
        Normalized timestep (0 to 1), shape (1,)
    cond : jnp.ndarray
        Conditioning metadata, shape (meta_dim,)
    guidance_scale : float
        Guidance strength:
        - w=0: purely unconditional
        - w=1: standard conditional
        - w>1: amplified conditioning (typical: 2-4)

    Returns
    -------
    jnp.ndarray
        Guided noise prediction, shape (C, L)
    """
    eps_uncond = model(x_t, t, jnp.zeros_like(cond), key=None, dropout_p=0.0)
    eps_cond = model(x_t, t, cond, key=None, dropout_p=0.0)
    return eps_uncond + guidance_scale * (eps_cond - eps_uncond)


# =========================================================================
# Low-Level Sampling Functions
# =========================================================================

def _make_schedule_indices(T: int, K: int) -> jnp.ndarray:
    """Subsample T timesteps down to K evenly-spaced steps."""
    return jnp.round(jnp.linspace(T, 1, K)).astype(jnp.int32)


def _sample_ddpm(
    key: jax.random.PRNGKey,
    model: eqx.Module,
    cond_vec: jnp.ndarray,
    n_sample: int,
    size: Tuple[int, int],
    n_T: int,
    guidance_scale: float
) -> jnp.ndarray:
    """DDPM ancestral sampling (stochastic)."""
    schedules = cosine_beta_schedule(T=n_T)
    x_i = jax.random.normal(key, (n_sample, *size))

    guided_vmap = jax.vmap(guided_denoising_step, in_axes=(None, 0, 0, 0, None))

    for i in range(n_T, 0, -1):
        t = jnp.full((n_sample, 1), i / n_T, dtype=jnp.float32)
        eps = guided_vmap(model, x_i, t, cond_vec, guidance_scale)
        x_i = (schedules["oneover_sqrta"][i] *
               (x_i - eps * schedules["mab_over_sqrtmab"][i]))

        if i > 1:
            key, subkey = jax.random.split(key)
            z = jax.random.normal(subkey, x_i.shape)
            x_i += schedules["sqrt_posterior_var"][i] * z

    return x_i


@eqx.filter_jit
def _sample_ddim(
    key: jax.random.PRNGKey,
    model: eqx.Module,
    cond_vec: jnp.ndarray,
    n_sample: int,
    size: Tuple[int, int],
    n_T: int,
    guidance_scale: float,
    num_steps: int
) -> jnp.ndarray:
    """DDIM sampling with η=0 (deterministic)."""
    sched = cosine_beta_schedule(T=n_T)
    sqrtab = sched["sqrtab"]
    sqrtmab = sched["sqrtmab"]

    idx_pairs = jnp.stack(
        [_make_schedule_indices(n_T, num_steps)[:-1],
         _make_schedule_indices(n_T, num_steps)[1:]],
        axis=1,
    )

    x_T = jax.random.normal(key, (n_sample, *size))
    guided_vmap = jax.vmap(guided_denoising_step, in_axes=(None, 0, 0, 0, None))

    def _ddim_step(x_i, idx_pair):
        i, i_prev = idx_pair

        t_i = jnp.full((n_sample, 1), i / n_T, dtype=jnp.float32)
        eps_i = guided_vmap(model, x_i, t_i, cond_vec, guidance_scale)

        # Predict x0
        pred_x0 = (x_i - sqrtmab[i] * eps_i) / sqrtab[i]

        # DDIM update (η=0)
        x_next = sqrtab[i_prev] * pred_x0 + sqrtmab[i_prev] * eps_i
        return x_next, None

    x_0, _ = jax.lax.scan(_ddim_step, x_T, idx_pairs)
    return x_0


@eqx.filter_jit
def _sample_heun(
    key: jax.random.PRNGKey,
    model: eqx.Module,
    cond_vec: jnp.ndarray,
    n_sample: int,
    size: Tuple[int, int],
    n_T: int,
    guidance_scale: float,
    num_steps: int
) -> jnp.ndarray:
    """Heun's method for probability-flow ODE (deterministic, second-order)."""
    sched = cosine_beta_schedule(T=n_T)
    oa = sched["oneover_sqrta"]
    mab_osm = sched["mab_over_sqrtmab"]

    idx_pairs = jnp.stack(
        [_make_schedule_indices(n_T, num_steps)[:-1],
         _make_schedule_indices(n_T, num_steps)[1:]],
        axis=1,
    )

    x_T = jax.random.normal(key, (n_sample, *size))
    guided_vmap = jax.vmap(guided_denoising_step, in_axes=(None, 0, 0, 0, None))

    def _heun_step(x_i, idx_pair):
        i, i_prev = idx_pair

        # Predictor (Euler step)
        t_i = jnp.full((n_sample, 1), i / n_T, dtype=jnp.float32)
        eps_i = guided_vmap(model, x_i, t_i, cond_vec, guidance_scale)
        x_euler = oa[i] * (x_i - mab_osm[i] * eps_i)

        # Corrector (average slopes)
        t_prev = jnp.full((n_sample, 1), i_prev / n_T, dtype=jnp.float32)
        eps_prev = guided_vmap(model, x_euler, t_prev, cond_vec, guidance_scale)
        eps_avg = 0.5 * (eps_i + eps_prev)

        x_next = oa[i_prev] * (x_i - mab_osm[i_prev] * eps_avg)
        return x_next, None

    x_0, _ = jax.lax.scan(_heun_step, x_T, idx_pairs)
    return x_0


# =========================================================================
# High-Level Sampler Class
# =========================================================================

@dataclass
class SamplerConfig:
    """
    Configuration for latent diffusion sampling.

    Attributes
    ----------
    method : str
        Sampling algorithm: "heun", "ddim", or "ddpm"
    num_steps : int
        Number of sampling steps (for heun/ddim; ddpm always uses n_T steps)
    n_T : int
        Number of training timesteps (typically 1000)
    latent_channels : int
        Number of channels in latent space (typically 1)
    latent_dim : int
        Latent dimension (typically 8 for VAE)
    """
    method: Literal["heun", "ddim", "ddpm"] = "heun"
    num_steps: int = 40
    n_T: int = 1000
    latent_channels: int = 1
    latent_dim: int = 8


class LatentDiffusionSampler:
    """
    High-level interface for sampling from latent diffusion models.

    This class provides a clean, user-friendly API for generating sky spectra
    using trained latent diffusion models. It handles all the complexity of
    noise scheduling, classifier-free guidance, and VAE decoding.

    Parameters
    ----------
    ldm_model : eqx.Module
        Trained latent diffusion model (UNet)
    vae_model : eqx.Module
        Trained VAE for encoding/decoding
    method : str
        Sampling algorithm: "heun" (recommended), "ddim", or "ddpm"
    num_steps : int
        Number of sampling steps (for heun/ddim)
    n_T : int
        Number of training timesteps
    latent_channels : int
        Number of latent channels (typically 1)
    latent_dim : int
        Latent dimension (typically 8)

    Examples
    --------
    >>> from desisky.inference import LatentDiffusionSampler
    >>> from desisky.io import load_builtin
    >>> import jax.random as jr
    >>>
    >>> # Load models
    >>> ldm, _ = load_builtin("ldm_dark")
    >>> vae, _ = load_builtin("vae")
    >>>
    >>> # Create sampler
    >>> sampler = LatentDiffusionSampler(ldm, vae)
    >>>
    >>> # Prepare conditioning (OBSALT, TRANSP, SUNALT, SOLFLUX, ECLLON, ECLLAT, GALLON, GALLAT)
    >>> conditioning = jnp.array([
    ...     [60.0, 0.9, -30.0, 150.0, 45.0, 10.0, 120.0, 5.0],
    ...     [70.0, 0.85, -25.0, 155.0, 50.0, 12.0, 125.0, 6.0],
    ... ])
    >>>
    >>> # Generate samples
    >>> spectra = sampler.sample(
    ...     key=jr.PRNGKey(0),
    ...     conditioning=conditioning,
    ...     guidance_scale=2.0
    ... )
    >>> spectra.shape
    (2, 7781)
    """

    def __init__(
        self,
        ldm_model: eqx.Module,
        vae_model: eqx.Module,
        method: Literal["heun", "ddim", "ddpm"] = "heun",
        num_steps: int = 40,
        n_T: int = 1000,
        latent_channels: int = 1,
        latent_dim: int = 8
    ):
        self.ldm = ldm_model
        self.vae = vae_model
        self.config = SamplerConfig(
            method=method,
            num_steps=num_steps,
            n_T=n_T,
            latent_channels=latent_channels,
            latent_dim=latent_dim
        )

        # Select sampling function
        self._samplers = {
            "heun": _sample_heun,
            "ddim": _sample_ddim,
            "ddpm": _sample_ddpm
        }

        if method not in self._samplers:
            raise ValueError(
                f"Unknown sampling method '{method}'. "
                f"Choose from: {list(self._samplers.keys())}"
            )

    def sample_latents(
        self,
        key: jax.random.PRNGKey,
        conditioning: jnp.ndarray,
        n_samples: Optional[int] = None,
        guidance_scale: float = 2.0
    ) -> jnp.ndarray:
        """
        Sample latent representations from the diffusion model.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random key for sampling
        conditioning : jnp.ndarray
            Conditioning metadata, shape (n_samples, meta_dim) or (meta_dim,)
        n_samples : int, optional
            Number of samples to generate. If None, inferred from conditioning shape.
        guidance_scale : float
            Classifier-free guidance strength (typical: 1-4)

        Returns
        -------
        jnp.ndarray
            Generated latents, shape (n_samples, latent_channels, latent_dim)
        """
        # Handle conditioning shape
        if conditioning.ndim == 1:
            conditioning = conditioning[None, :]

        if n_samples is None:
            n_samples = conditioning.shape[0]
        elif n_samples != conditioning.shape[0]:
            raise ValueError(
                f"n_samples ({n_samples}) must match conditioning batch size "
                f"({conditioning.shape[0]})"
            )

        # Get sampling function
        sample_fn = self._samplers[self.config.method]

        # Sample
        latent_size = (self.config.latent_channels, self.config.latent_dim)

        if self.config.method == "ddpm":
            # DDPM doesn't use num_steps
            latents = sample_fn(
                key, self.ldm, conditioning,
                n_samples, latent_size,
                self.config.n_T, guidance_scale
            )
        else:
            # DDIM and Heun use num_steps
            latents = sample_fn(
                key, self.ldm, conditioning,
                n_samples, latent_size,
                self.config.n_T, guidance_scale,
                self.config.num_steps
            )

        return latents

    def sample(
        self,
        key: jax.random.PRNGKey,
        conditioning: jnp.ndarray,
        n_samples: Optional[int] = None,
        guidance_scale: float = 2.0,
        return_latents: bool = False
    ) -> jnp.ndarray | Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Sample sky spectra from the latent diffusion model.

        This is the main user-facing sampling method. It generates latent samples
        and decodes them to full-resolution spectra.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random key for sampling
        conditioning : jnp.ndarray
            Conditioning metadata, shape (n_samples, 8) or (8,)
            Features: [OBSALT, TRANSP, SUNALT, SOLFLUX, ECLLON, ECLLAT, GALLON, GALLAT]
        n_samples : int, optional
            Number of samples. If None, inferred from conditioning.
        guidance_scale : float
            Guidance strength. Higher = stronger conditioning.
            - 0: unconditional
            - 1: standard conditional
            - 2-4: typical range for good results
        return_latents : bool
            If True, return (spectra, latents). If False, return only spectra.

        Returns
        -------
        spectra : jnp.ndarray
            Generated sky spectra, shape (n_samples, 7781)
        latents : jnp.ndarray, optional
            Generated latents (if return_latents=True), shape (n_samples, latent_channels, latent_dim)

        Examples
        --------
        >>> # Single sample
        >>> cond = jnp.array([60.0, 0.9, -30.0, 150.0, 45.0, 10.0, 120.0, 5.0])
        >>> spec = sampler.sample(jr.PRNGKey(0), cond, guidance_scale=2.0)
        >>>
        >>> # Multiple samples
        >>> conds = jnp.array([[60.0, 0.9, ...], [70.0, 0.85, ...]])
        >>> specs = sampler.sample(jr.PRNGKey(1), conds, guidance_scale=3.0)
        >>>
        >>> # Return latents too
        >>> specs, lats = sampler.sample(jr.PRNGKey(2), conds, return_latents=True)
        """
        # Sample latents
        latents = self.sample_latents(key, conditioning, n_samples, guidance_scale)

        # Decode to spectra
        # VAE decoder expects shape (latent_dim,), so squeeze channel dimension
        latents_squeezed = latents.squeeze(1)  # (n_samples, latent_dim)
        spectra = jax.vmap(self.vae.decode)(latents_squeezed)

        if return_latents:
            return spectra, latents
        return spectra

    def __repr__(self) -> str:
        return (
            f"LatentDiffusionSampler(\n"
            f"  method={self.config.method},\n"
            f"  num_steps={self.config.num_steps},\n"
            f"  n_T={self.config.n_T},\n"
            f"  latent_shape=({self.config.latent_channels}, {self.config.latent_dim})\n"
            f")"
        )
