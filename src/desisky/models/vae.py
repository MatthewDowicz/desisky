"""
Variational Autoencoder (VAE) for DESI sky spectra.

This module provides a VAE architecture for encoding sky spectra into a low-dimensional
latent space and reconstructing them. The VAE can be used for:
- Dimensionality reduction and compression of sky spectra
- Anomaly detection via reconstruction error
- Interpolation in latent space between different sky conditions
- Analysis of latent-space structure vs. physical parameters

The VAE is typically used as part of the Latent Diffusion Model (LDM) pipeline,
but can also be used standalone for the above tasks.
"""

from typing import Any, Optional, Dict
import jax
import jax.numpy as jnp
import equinox as eqx

from desisky.io.model_io import register_model, ModelSpec


class SkyVAE(eqx.Module):
    """
    Variational Autoencoder for DESI sky spectra.

    This VAE compresses sky spectra into a low-dimensional latent representation
    and can reconstruct them. The architecture consists of:
    - Encoder: Maps input spectrum → latent distribution (mean, log_var)
    - Decoder: Maps latent vector → reconstructed spectrum

    Attributes
    ----------
    in_channels : int
        Number of wavelength bins in input spectrum
    latent_dim : int
        Dimensionality of the latent space
    common_fc : eqx.nn.Sequential
        Shared encoder layers before mean/logvar split
    mean_fc : eqx.nn.Sequential
        Layers that produce latent mean
    log_var_fc : eqx.nn.Sequential
        Layers that produce latent log-variance
    decoder_fcs : eqx.nn.Sequential
        Decoder layers that reconstruct spectrum from latent

    Examples
    --------
    Basic usage with pretrained weights:

    >>> from desisky.io import load_builtin
    >>> vae, meta = load_builtin("vae")
    >>>
    >>> # Encode a sky spectrum
    >>> mean, logvar = vae.encode(sky_spectrum)
    >>>
    >>> # Sample from latent space and decode
    >>> import jax.random as jr
    >>> latent = vae.sample(mean, logvar, jr.PRNGKey(0))
    >>> reconstructed = vae.decode(latent)
    >>>
    >>> # Or do everything at once
    >>> result = vae(sky_spectrum, jr.PRNGKey(0))
    >>> # result contains: 'mean', 'logvar', 'output', 'latent'

    For batch processing:

    >>> # Encode batch of spectra
    >>> means, logvars = jax.vmap(vae.encode)(batch_of_spectra)
    >>>
    >>> # Decode batch of latents
    >>> reconstructed_batch = jax.vmap(vae.decode)(batch_of_latents)
    """

    in_channels: int
    latent_dim: int
    common_fc: eqx.nn.Sequential
    mean_fc: eqx.nn.Sequential
    log_var_fc: eqx.nn.Sequential
    decoder_fcs: eqx.nn.Sequential

    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        key: Optional[Any] = None
    ):
        """
        Initialize the VAE.

        Parameters
        ----------
        in_channels : int
            Number of wavelength bins in input spectrum (typically 7781 for DESI)
        latent_dim : int
            Dimensionality of latent space (e.g., 8)
        key : jax.random.PRNGKey, optional
            Random key for initialization. If None, uses PRNGKey(0)
        """
        if key is None:
            key = jax.random.PRNGKey(0)

        keys = jax.random.split(key, 16)

        self.in_channels = in_channels
        self.latent_dim = latent_dim

        # Encoder: spectrum → latent representation
        self.common_fc = eqx.nn.Sequential([
            eqx.nn.Linear(in_channels, 1000, key=keys[0]),
            eqx.nn.Lambda(jax.nn.relu),
            eqx.nn.Linear(1000, 800, key=keys[1]),
            eqx.nn.Lambda(jax.nn.relu),
            eqx.nn.Linear(800, 600, key=keys[2]),
            eqx.nn.Lambda(jax.nn.relu),
            eqx.nn.Linear(600, 500, key=keys[3]),
            eqx.nn.Lambda(jax.nn.relu),
            eqx.nn.Linear(500, 300, key=keys[4]),
            eqx.nn.Lambda(jax.nn.relu),
            eqx.nn.Linear(300, self.latent_dim, key=keys[5])
        ])

        # Mean and log-variance heads
        self.mean_fc = eqx.nn.Sequential([
            eqx.nn.Linear(self.latent_dim, self.latent_dim, key=keys[6]),
            eqx.nn.Lambda(jax.nn.relu),
            eqx.nn.Linear(self.latent_dim, self.latent_dim, key=keys[7])
        ])

        self.log_var_fc = eqx.nn.Sequential([
            eqx.nn.Linear(self.latent_dim, self.latent_dim, key=keys[8]),
            eqx.nn.Lambda(jax.nn.relu),
            eqx.nn.Linear(self.latent_dim, self.latent_dim, key=keys[9])
        ])

        # Decoder: latent → reconstructed spectrum
        self.decoder_fcs = eqx.nn.Sequential([
            eqx.nn.Linear(self.latent_dim, 300, key=keys[10]),
            eqx.nn.Lambda(jax.nn.relu),
            eqx.nn.Linear(300, 500, key=keys[11]),
            eqx.nn.Lambda(jax.nn.relu),
            eqx.nn.Linear(500, 600, key=keys[12]),
            eqx.nn.Lambda(jax.nn.relu),
            eqx.nn.Linear(600, 800, key=keys[13]),
            eqx.nn.Lambda(jax.nn.relu),
            eqx.nn.Linear(800, 1000, key=keys[14]),
            eqx.nn.Lambda(jax.nn.relu),
            eqx.nn.Linear(1000, in_channels, key=keys[15])
        ])

    def __call__(self, x: jnp.ndarray, key: jax.random.PRNGKey) -> Dict[str, jnp.ndarray]:
        """
        Full forward pass: encode, sample, decode.

        This method handles both single samples and batches:
        - Single sample: x.shape = (in_channels,)
        - Batch: x.shape = (batch_size, in_channels)

        For batched inputs, encode and decode are vmapped, but sample handles
        the batch internally via broadcasting.

        Parameters
        ----------
        x : jnp.ndarray
            Input spectrum. Shape (in_channels,) for single sample
            or (batch_size, in_channels) for batch.
        key : jax.random.PRNGKey
            Random key for sampling latent space

        Returns
        -------
        dict
            Dictionary containing:
            - 'mean': Latent mean(s)
            - 'logvar': Latent log-variance(s)
            - 'latent': Sampled latent vector(s)
            - 'output': Reconstructed spectrum/spectra

        Examples
        --------
        Single sample:
        >>> result = vae(single_spectrum, key)
        >>> result['mean'].shape  # (latent_dim,)

        Batch processing:
        >>> result = vae(batch_of_spectra, key)
        >>> result['mean'].shape  # (batch_size, latent_dim)
        """
        # Check if input is batched
        is_batched = x.ndim == 2

        if is_batched:
            # Batch processing: vmap encode and decode, but not sample
            mean, logvar = jax.vmap(self.encode)(x)
            z = self.sample(mean, logvar, key)  # sample handles batches internally
            out = jax.vmap(self.decode)(z)
        else:
            # Single sample processing
            mean, logvar = self.encode(x)
            z = self.sample(mean, logvar, key)
            out = self.decode(z)

        return {
            'mean': mean,
            'logvar': logvar,
            'latent': z,
            'output': out
        }

    def encode(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Encode input spectrum to latent distribution parameters.

        Parameters
        ----------
        x : jnp.ndarray
            Input spectrum of shape (in_channels,)

        Returns
        -------
        mean : jnp.ndarray
            Latent mean, shape (latent_dim,)
        logvar : jnp.ndarray
            Latent log-variance, shape (latent_dim,)
        """
        out = self.common_fc(x)
        mean = self.mean_fc(out)
        log_var = self.log_var_fc(out)
        return mean, log_var

    def sample(
        self,
        mean: jnp.ndarray,
        log_var: jnp.ndarray,
        key: jax.random.PRNGKey
    ) -> jnp.ndarray:
        """
        Sample from latent distribution using reparameterization trick.

        This method handles both single samples and batches through JAX broadcasting.
        When called with batched inputs, all samples use the same random key.

        Parameters
        ----------
        mean : jnp.ndarray
            Latent mean. Shape (latent_dim,) for single sample
            or (batch_size, latent_dim) for batch.
        log_var : jnp.ndarray
            Latent log-variance. Shape (latent_dim,) for single sample
            or (batch_size, latent_dim) for batch.
        key : jax.random.PRNGKey
            Random key for sampling (shared across batch if batched)

        Returns
        -------
        z : jnp.ndarray
            Sampled latent vector(s). Shape matches input mean/log_var.
        """
        std = jnp.exp(0.5 * log_var)
        epsilon = jax.random.normal(key, std.shape)
        z = epsilon * std + mean
        return z

    def decode(self, z: jnp.ndarray) -> jnp.ndarray:
        """
        Decode latent vector to reconstructed spectrum.

        Parameters
        ----------
        z : jnp.ndarray
            Latent vector, shape (latent_dim,)

        Returns
        -------
        spectrum : jnp.ndarray
            Reconstructed spectrum, shape (in_channels,)
        """
        out = self.decoder_fcs(z)
        return out


def make_SkyVAE(
    *,
    in_channels: int,
    latent_dim: int,
    key: Optional[Any] = None
) -> SkyVAE:
    """
    Constructor for the SkyVAE architecture.

    Accepts exactly the kwargs stored under 'meta["arch"]' in the checkpoint header.

    Parameters
    ----------
    in_channels : int
        Number of wavelength bins in input spectrum
    latent_dim : int
        Dimensionality of latent space
    key : jax.random.PRNGKey, optional
        Random key for initialization. If None, uses PRNGKey(0)

    Returns
    -------
    SkyVAE
        Initialized VAE model
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    return SkyVAE(
        in_channels=in_channels,
        latent_dim=latent_dim,
        key=key
    )


# Register this model kind with the IO registry
register_model(
    "vae",
    ModelSpec(constructor=make_SkyVAE, resource="vae_weights.eqx"),
)
