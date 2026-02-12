"""
Latent Diffusion Model (LDM) architecture for DESI sky spectra.

This module provides a 1D conditional U-Net for latent diffusion modeling of sky spectra.
The UNet operates on latent representations produced by the SkyVAE and is conditioned on
observational metadata (moon position, observing altitude, transparency, etc.).

The architecture supports classifier-free guidance for controllable generation.

EDM preconditioning (Karras et al. 2022) wraps the raw UNet to improve training
stability and sample quality. The preconditioned denoiser is:

    D(x; σ) = c_skip(σ) · x + c_out(σ) · F_θ(c_in(σ) · x; c_noise(σ))

where F_θ is the raw UNet network.
"""

from typing import Any, Optional, Callable, Sequence
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from desisky.io.model_io import register_model, ModelSpec


# -------------------------------------------------------------------------
# EDM Constants (Karras et al. 2022, Table 1)
# -------------------------------------------------------------------------

EDM_P_MEAN = -1.2       # Mean of log-normal noise distribution
EDM_P_STD = 1.2         # Std of log-normal noise distribution
EDM_SIGMA_MIN = 0.002   # Minimum sigma for sampling
EDM_SIGMA_MAX = 80.0    # Maximum sigma for sampling


# -------------------------------------------------------------------------
# EDM Preconditioning Functions (Karras et al. 2022, Table 1)
# -------------------------------------------------------------------------

def c_skip(sigma: jnp.ndarray, sigma_data: float) -> jnp.ndarray:
    """
    Skip connection coefficient: c_skip(σ) = σ_data² / (σ² + σ_data²).

    At low noise (σ→0): c_skip→1 (pass input through).
    At high noise (σ→∞): c_skip→0 (rely on network).
    """
    return sigma_data**2 / (sigma**2 + sigma_data**2)


def c_out(sigma: jnp.ndarray, sigma_data: float) -> jnp.ndarray:
    """
    Output scaling: c_out(σ) = σ · σ_data / √(σ² + σ_data²).

    Ensures denoiser output has appropriate variance.
    """
    return sigma * sigma_data / jnp.sqrt(sigma**2 + sigma_data**2)


def c_in(sigma: jnp.ndarray, sigma_data: float) -> jnp.ndarray:
    """
    Input scaling: c_in(σ) = 1 / √(σ² + σ_data²).

    Normalizes input to approximately unit variance.
    """
    return 1.0 / jnp.sqrt(sigma**2 + sigma_data**2)


def c_noise(sigma: jnp.ndarray) -> jnp.ndarray:
    """
    Noise conditioning: c_noise(σ) = (1/4) · ln(σ).

    Maps continuous σ to suitable range for sinusoidal embeddings.
    """
    return 0.25 * jnp.log(sigma)


def edm_denoiser(
    model: eqx.Module,
    x_noisy: jnp.ndarray,
    sigma: jnp.ndarray,
    cond: jnp.ndarray,
    sigma_data: float,
    key=None,
    dropout_p: float = 0.0,
) -> jnp.ndarray:
    """
    EDM preconditioned denoiser (single sample).

    D(x; σ) = c_skip(σ) · x + c_out(σ) · F_θ(c_in(σ) · x; c_noise(σ), cond)

    Parameters
    ----------
    model : eqx.Module
        Raw UNet network F_θ.
    x_noisy : jnp.ndarray
        Noisy input, shape (C, L) — single sample.
    sigma : jnp.ndarray
        Noise level (scalar).
    cond : jnp.ndarray
        Conditioning vector, shape (meta_dim,).
    sigma_data : float
        Standard deviation of training data.
    key : jax.random.PRNGKey, optional
        RNG key for CFG dropout (None at inference).
    dropout_p : float
        CFG dropout probability.

    Returns
    -------
    jnp.ndarray
        Denoised estimate, shape (C, L).
    """
    c_s = c_skip(sigma, sigma_data)
    c_o = c_out(sigma, sigma_data)
    c_i = c_in(sigma, sigma_data)
    c_n = c_noise(sigma)

    x_scaled = c_i * x_noisy
    F_out = model(x_scaled, c_n, cond, key=key, dropout_p=dropout_p)

    return c_s * x_noisy + c_o * F_out


def compute_sigma_data(latent_data: np.ndarray) -> float:
    """
    Compute σ_data (standard deviation of training data) for EDM.

    Parameters
    ----------
    latent_data : np.ndarray
        Training latents, shape (N, latent_dim) or (N, C, L).

    Returns
    -------
    float
        Standard deviation of the training data.
    """
    return float(np.std(latent_data))


# -------------------------------------------------------------------------
# Conditioning and Time Embedding Components
# -------------------------------------------------------------------------

def sinusoidal_embedding(t: jnp.ndarray, dim: int = 32) -> jnp.ndarray:
    """
    Sinusoidal timestep embedding for diffusion models.

    Parameters
    ----------
    t : jnp.ndarray
        Timestep values, shape () or (B,) or (B, 1)
    dim : int
        Embedding dimension (should be even)

    Returns
    -------
    jnp.ndarray
        Sinusoidal embeddings of shape (..., dim)

    Examples
    --------
    >>> t = jnp.array([0.5, 0.8])
    >>> emb = sinusoidal_embedding(t, dim=32)
    >>> emb.shape
    (2, 32)
    """
    if t.ndim > 0 and t.shape[-1] == 1:
        t = jnp.squeeze(t, -1)
    half = dim // 2
    freq = jnp.exp(-jnp.log(10000.0) * jnp.arange(half) / half)
    angles = t[..., None] * freq
    emb = jnp.stack([jnp.sin(angles), jnp.cos(angles)], -1)
    return emb.reshape(*emb.shape[:-2], -1)


class MetaEmbedding(eqx.Module):
    """
    MLP for embedding observational metadata (MOONALT, OBSALT, etc.).

    Attributes
    ----------
    mlp : eqx.nn.Sequential
        Two-layer MLP with ReLU activation
    emb_dim : int
        Output embedding dimension
    """
    mlp: eqx.nn.Sequential
    emb_dim: int

    def __init__(self, in_dim: int, emb_dim: int, *, key):
        """
        Parameters
        ----------
        in_dim : int
            Input dimension (number of metadata features)
        emb_dim : int
            Output embedding dimension
        key : jax.random.PRNGKey
            Random key for initialization
        """
        k1, k2 = jax.random.split(key)
        self.mlp = eqx.nn.Sequential([
            eqx.nn.Linear(in_dim, 4 * emb_dim, key=k1),
            eqx.nn.Lambda(jax.nn.relu),
            eqx.nn.Linear(4 * emb_dim, emb_dim, key=k2)
        ])
        self.emb_dim = emb_dim

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Embed metadata features."""
        return self.mlp(x)


# -------------------------------------------------------------------------
# Convolutional Building Blocks
# -------------------------------------------------------------------------

class DoubleConvCond(eqx.Module):
    """
    Two consecutive convolutions with group normalization and time/metadata conditioning.

    The conditioning embedding is projected and added as a bias to both convolutions.

    Attributes
    ----------
    conv1, conv2 : eqx.nn.Conv
        3x1 convolutions
    norm1, norm2 : eqx.nn.GroupNorm
        Group normalization layers
    t_proj : eqx.nn.Linear
        Projects conditioning embedding to channel bias
    act : Callable
        Activation function (default: ReLU)
    """
    conv1: eqx.nn.Conv
    norm1: eqx.nn.GroupNorm
    conv2: eqx.nn.Conv
    norm2: eqx.nn.GroupNorm
    t_proj: eqx.nn.Linear
    act: Callable = jax.nn.relu

    def __init__(
        self,
        num_spatial_dims: int,
        in_ch: int,
        out_ch: int,
        emb_dim: int,
        act: Callable,
        *,
        key,
        groups: int = 8,
    ):
        """
        Parameters
        ----------
        num_spatial_dims : int
            Number of spatial dimensions (1 for 1D convolution)
        in_ch : int
            Input channels
        out_ch : int
            Output channels
        emb_dim : int
            Conditioning embedding dimension
        act : Callable
            Activation function
        key : jax.random.PRNGKey
            Random key for initialization
        groups : int
            Number of groups for group normalization
        """
        k1, k2, k3 = jax.random.split(key, 3)
        self.conv1 = eqx.nn.Conv(num_spatial_dims, in_ch, out_ch, 3, padding=1, key=k1)
        self.norm1 = eqx.nn.GroupNorm(groups=groups, channels=out_ch)
        self.conv2 = eqx.nn.Conv(num_spatial_dims, out_ch, out_ch, 3, padding=1, key=k2)
        self.norm2 = eqx.nn.GroupNorm(groups=groups, channels=out_ch)
        self.t_proj = eqx.nn.Linear(emb_dim, out_ch, key=k3)
        self.act = act

    def __call__(self, x: jnp.ndarray, t_emb: jnp.ndarray):
        """
        Forward pass with conditioning.

        Parameters
        ----------
        x : jnp.ndarray
            Input tensor, shape (C, L)
        t_emb : jnp.ndarray
            Conditioning embedding, shape (emb_dim,)

        Returns
        -------
        jnp.ndarray
            Output tensor, shape (C, L)
        """
        bias = self.t_proj(t_emb)[:, None]  # (C, 1)

        x = self.conv1(x) + bias
        x = self.norm1(x)
        x = self.act(x)

        x = self.conv2(x) + bias
        x = self.norm2(x)
        x = self.act(x)

        return x


class ResBlockCond(eqx.Module):
    """
    Residual wrapper around DoubleConvCond.

    Implements: y = DoubleConvCond(x, emb) + proj(x)

    If in_ch != out_ch, uses a 1x1 conv for the skip connection.
    Otherwise, uses identity skip connection.

    Attributes
    ----------
    block : DoubleConvCond
        The convolutional block
    skip_proj : Optional[eqx.nn.Conv]
        1x1 conv for skip connection (None if in_ch == out_ch)
    scale : float
        Scaling factor for residual sum (1/√2)
    """
    block: DoubleConvCond
    skip_proj: Optional[eqx.nn.Conv]
    scale: float

    def __init__(
        self,
        num_spatial_dims: int,
        in_ch: int,
        out_ch: int,
        emb_dim: int,
        act: Callable = jax.nn.relu,
        *,
        key,
        groups: int = 8
    ):
        """
        Parameters
        ----------
        num_spatial_dims : int
            Number of spatial dimensions (1 for 1D convolution)
        in_ch : int
            Input channels
        out_ch : int
            Output channels
        emb_dim : int
            Conditioning embedding dimension
        act : Callable
            Activation function
        key : jax.random.PRNGKey
            Random key for initialization
        groups : int
            Number of groups for group normalization
        """
        bk, pk = jax.random.split(key)
        self.block = DoubleConvCond(
            num_spatial_dims=num_spatial_dims,
            in_ch=in_ch,
            out_ch=out_ch,
            emb_dim=emb_dim,
            act=act,
            key=bk,
            groups=groups
        )
        self.skip_proj = (
            None if in_ch == out_ch
            else eqx.nn.Conv(num_spatial_dims, in_ch, out_ch, 1, key=pk)
        )
        self.scale = 1.0 / jnp.sqrt(2.0)

    def __call__(self, x: jnp.ndarray, t_emb: jnp.ndarray):
        """Forward pass with residual connection."""
        out = self.block(x, t_emb)
        skip = x if self.skip_proj is None else self.skip_proj(x)
        return (out + skip) * self.scale


class SelfAttention1D(eqx.Module):
    """
    Self-attention block for 1D data (channels-first format).

    Attributes
    ----------
    norm : eqx.nn.GroupNorm
        Group normalization (acts like layer norm when groups=1)
    mha : eqx.nn.MultiheadAttention
        Multi-head attention module
    """
    norm: eqx.nn.GroupNorm
    mha: eqx.nn.MultiheadAttention

    def __init__(
        self,
        channels: int,
        heads: int = 4,
        dim_head: int = 32,
        *,
        key
    ):
        """
        Parameters
        ----------
        channels : int
            Number of input channels
        heads : int
            Number of attention heads
        dim_head : int
            Dimension per head
        key : jax.random.PRNGKey
            Random key for initialization
        """
        nk, ak = jax.random.split(key)
        self.norm = eqx.nn.GroupNorm(groups=1, channels=channels)

        self.mha = eqx.nn.MultiheadAttention(
            num_heads=heads,
            query_size=channels,
            key_size=channels,
            value_size=channels,
            output_size=channels,
            qk_size=dim_head,
            vo_size=dim_head,
            use_query_bias=False,
            use_key_bias=False,
            use_value_bias=False,
            use_output_bias=False,
            key=ak,
        )

    def __call__(self, x: jnp.ndarray, *, key=None):
        """
        Forward pass with self-attention.

        Parameters
        ----------
        x : jnp.ndarray
            Input tensor, shape (C, L)
        key : jax.random.PRNGKey, optional
            Random key for dropout (if enabled)

        Returns
        -------
        jnp.ndarray
            Output tensor with residual connection, shape (C, L)
        """
        x_norm = self.norm(x)
        # Transpose to (L, C) for Equinox MultiheadAttention API
        y = self.mha(x_norm.T, x_norm.T, x_norm.T, key=key)
        y = y.T  # Back to (C, L)
        return x + y  # Residual connection


# -------------------------------------------------------------------------
# Main UNet Architecture
# -------------------------------------------------------------------------

class UNet1D_cond(eqx.Module):
    """
    1D U-Net with timestep and metadata conditioning for latent diffusion.

    This architecture is designed for denoising latent representations of sky spectra.
    It includes:
    - Encoder path with downsampling
    - Bottleneck with self-attention
    - Decoder path with upsampling and skip connections
    - Self-attention after first upsampling layer
    - Conditioning via timestep + metadata embeddings

    Attributes
    ----------
    lifting : ResBlockCond
        Initial residual block
    down_convs : Sequence[eqx.nn.Conv]
        Downsampling convolutions
    left_blocks : Sequence[ResBlockCond]
        Encoder residual blocks
    right_blocks : Sequence[ResBlockCond]
        Decoder residual blocks
    up_convs : Sequence[eqx.nn.ConvTranspose]
        Upsampling transpose convolutions
    projection : eqx.nn.Conv
        Final 1x1 projection to output channels
    mid_attn : SelfAttention1D
        Bottleneck attention layer
    up_attn : SelfAttention1D
        Decoder attention layer (after first upsampling)
    emb_dim : int
        Conditioning embedding dimension
    meta_head : MetaEmbedding
        Metadata embedding network
    in_ch, out_ch, hidden, levels, meta_dim : int
        Architecture hyperparameters (stored for saving/loading)

    Examples
    --------
    Create and use a UNet for latent diffusion:

    >>> from desisky.io import load_builtin
    >>> unet, meta = load_builtin("ldm_dark")
    >>>
    >>> # Denoise a latent at timestep t=0.5
    >>> x_noisy = jax.random.normal(key, (1, 8))  # (C, L)
    >>> t = jnp.array(0.5)
    >>> metadata = jnp.array([moonalt, moonsep, moonfrac, obsalt, transp])
    >>> eps_pred = unet(x_noisy, t, metadata)

    For batch processing with vmap:

    >>> batch_unet = jax.vmap(unet, in_axes=(0, 0, 0, None))
    >>> eps_batch = batch_unet(x_batch, t_batch, meta_batch, None)
    """

    # Architecture components
    lifting: ResBlockCond
    down_convs: Sequence[eqx.nn.Conv]
    left_blocks: Sequence[ResBlockCond]
    right_blocks: Sequence[ResBlockCond]
    up_convs: Sequence[eqx.nn.ConvTranspose]
    projection: eqx.nn.Conv
    mid_attn: SelfAttention1D
    up_attn: SelfAttention1D

    # Conditioning
    emb_dim: int
    meta_head: MetaEmbedding

    # Hyperparameters (for serialization)
    in_ch: int
    out_ch: int
    hidden: int
    levels: int
    meta_dim: int

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        meta_dim: int = 5,
        hidden: int = 32,
        levels: int = 2,
        emb_dim: int = 32,
        act: Callable = jax.nn.relu,
        *,
        key,
    ):
        """
        Initialize the 1D conditional U-Net.

        Parameters
        ----------
        in_ch : int
            Input channels (typically 1 for latent diffusion)
        out_ch : int
            Output channels (typically 1, predicting noise)
        meta_dim : int
            Number of metadata conditioning features
            (e.g., 5 for MOONALT, MOONSEP, MOONFRAC, OBSALT, TRANSPARENCY)
        hidden : int
            Base number of channels (doubled at each level)
        levels : int
            Number of encoder/decoder levels (depth of U-Net)
        emb_dim : int
            Dimension for timestep and metadata embeddings
        act : Callable
            Activation function (default: ReLU)
        key : jax.random.PRNGKey
            Random key for initialization
        """
        self.emb_dim = emb_dim
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.hidden = hidden
        self.levels = levels
        self.meta_dim = meta_dim

        # Split random keys
        keys = jax.random.split(key, 6 + 5 * levels)
        meta_k, lift_k, proj_k, attn_mid_k, attn_up_k, *rest = keys

        # Conditioning embedding
        self.meta_head = MetaEmbedding(meta_dim, emb_dim, key=meta_k)

        # Stem and tail
        self.lifting = ResBlockCond(1, in_ch, hidden, emb_dim, act, key=lift_k)
        self.projection = eqx.nn.Conv(1, hidden, out_ch, 1, key=proj_k)

        # Build encoder/decoder channel progression
        chs = [hidden * 2**i for i in range(levels + 1)]

        # Attention layers
        self.mid_attn = SelfAttention1D(chs[-1], heads=4, dim_head=32, key=attn_mid_k)
        self.up_attn = SelfAttention1D(chs[-2], heads=4, dim_head=32, key=attn_up_k)

        # Build encoder and decoder paths
        self.down_convs, self.left_blocks = [], []
        self.right_blocks, self.up_convs = [], []

        for i, (up_c, down_c) in enumerate(zip(chs[:-1], chs[1:])):
            dk, lk, lp_k, rk, rp_k = rest[5 * i : 5 * (i + 1)]

            # Encoder: downsample, then residual block
            self.down_convs.append(
                eqx.nn.Conv(1, up_c, up_c, 3, stride=2, padding=1, key=dk)
            )
            self.left_blocks.append(
                ResBlockCond(1, up_c, down_c, emb_dim, act, key=lk)
            )

            # Decoder: upsample, then residual block
            self.up_convs.append(
                eqx.nn.ConvTranspose(1, down_c, up_c, 3, stride=2, padding=1,
                                    output_padding=1, key=rk)
            )
            self.right_blocks.append(
                ResBlockCond(1, down_c, up_c, emb_dim, act, key=rp_k)
            )

    def _match_length(self, x, ref):
        """
        Make x the same spatial length as ref by cropping or zero-padding.

        Handles off-by-one mismatches from stride=2 conv/deconv.
        """
        diff = ref.shape[-1] - x.shape[-1]
        if diff == 0:
            return x
        elif diff > 0:
            # Pad on the right
            pad = [(0, 0)] * (x.ndim - 1) + [(0, diff)]
            return jnp.pad(x, pad)
        else:
            # Crop from the right
            return x[..., :ref.shape[-1]]

    def __call__(
        self,
        x: jnp.ndarray,
        t: jnp.ndarray,
        metadata: jnp.ndarray,
        key=None,
        dropout_p: float = 0.1
    ):
        """
        Forward pass: predict noise given noisy latent, timestep, and metadata.

        Parameters
        ----------
        x : jnp.ndarray
            Noisy latent, shape (C, L) - single sample
        t : jnp.ndarray
            Diffusion timestep, shape () or (1,)
        metadata : jnp.ndarray
            Observational metadata, shape (meta_dim,)
        key : jax.random.PRNGKey, optional
            Random key for classifier-free guidance dropout
        dropout_p : float
            Probability of zeroing metadata embedding (for classifier-free guidance)

        Returns
        -------
        jnp.ndarray
            Predicted noise, shape (C, L)
        """
        # Build joint conditioning embedding
        t_emb = sinusoidal_embedding(t, self.emb_dim)
        meta_emb = self.meta_head(metadata)

        # Classifier-free guidance: randomly drop metadata conditioning
        if key is not None:
            keep = jax.random.bernoulli(key, 1.0 - dropout_p)
            meta_emb = jnp.where(keep, meta_emb, jnp.zeros_like(meta_emb))

        emb = t_emb + meta_emb

        # Encoder
        x = self.lifting(x, emb)
        skips = []

        for down, left in zip(self.down_convs, self.left_blocks):
            skips.append(x)
            x = left(down(x), emb)

        # Bottleneck attention
        x = self.mid_attn(x)

        # Decoder
        for idx, (up, right) in enumerate(
            zip(reversed(self.up_convs), reversed(self.right_blocks))
        ):
            x = up(x)
            skip = skips.pop()

            # Match spatial dimensions
            x = self._match_length(x, skip)

            # Concatenate skip connection
            x = jnp.concatenate([x, skip], axis=0)
            x = right(x, emb)

            # Attention after first upsampling
            if idx == 0:
                x = self.up_attn(x)

        return self.projection(x)


# -------------------------------------------------------------------------
# Factory function and model registration
# -------------------------------------------------------------------------

def make_UNet1D_cond(
    *,
    in_ch: int,
    out_ch: int,
    meta_dim: int,
    hidden: int,
    levels: int,
    emb_dim: int,
    key=None,
) -> UNet1D_cond:
    """
    Factory function for creating a UNet1D_cond model.

    This function is used by the model loading system to instantiate models
    from saved checkpoints.

    Parameters
    ----------
    in_ch : int
        Input channels
    out_ch : int
        Output channels
    meta_dim : int
        Metadata embedding dimension
    hidden : int
        Base hidden dimension
    levels : int
        Number of encoder/decoder levels
    emb_dim : int
        Timestep/metadata embedding dimension
    key : jax.random.PRNGKey, optional
        Random key for initialization (defaults to PRNGKey(0))

    Returns
    -------
    UNet1D_cond
        Initialized UNet model
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    return UNet1D_cond(
        in_ch=in_ch,
        out_ch=out_ch,
        meta_dim=meta_dim,
        hidden=hidden,
        levels=levels,
        emb_dim=emb_dim,
        key=key
    )


# Register the LDM dark model for automatic loading
register_model(
    "ldm_dark",
    ModelSpec(constructor=make_UNet1D_cond, resource="ldm_dark.eqx")
)

# Register the LDM moon model for automatic loading
register_model(
    "ldm_moon",
    ModelSpec(constructor=make_UNet1D_cond, resource="ldm_moon.eqx")
)

# Register the LDM twilight model for automatic loading
register_model(
    "ldm_twilight",
    ModelSpec(constructor=make_UNet1D_cond, resource="ldm_twilight.eqx")
)
