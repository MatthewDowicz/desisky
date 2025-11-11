"""
Unit tests for SkyVAE model.

Tests cover:
- Model initialization
- Encoding sky spectra
- Sampling from latent space
- Decoding latent vectors
- Full forward pass
- Batch processing with vmap
- Loading pretrained weights
- Shape consistency
"""

import pytest
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from pathlib import Path
import tempfile
import json

from desisky.models.vae import SkyVAE, make_SkyVAE
from desisky.io.model_io import save, load, load_builtin


class TestSkyVAEConstruction:
    """Test VAE model construction and initialization."""

    def test_init_default_key(self):
        """Test VAE initialization with default key."""
        vae = SkyVAE(in_channels=100, latent_dim=8)
        assert vae.in_channels == 100
        assert vae.latent_dim == 8

    def test_init_custom_key(self):
        """Test VAE initialization with custom key."""
        key = jr.PRNGKey(42)
        vae = SkyVAE(in_channels=100, latent_dim=8, key=key)
        assert vae.in_channels == 100
        assert vae.latent_dim == 8

    def test_make_SkyVAE_constructor(self):
        """Test the make_SkyVAE constructor function."""
        vae = make_SkyVAE(in_channels=100, latent_dim=8)
        assert isinstance(vae, SkyVAE)
        assert vae.in_channels == 100
        assert vae.latent_dim == 8

    def test_different_architectures(self):
        """Test VAE with different architecture sizes."""
        configs = [
            (50, 4),
            (100, 8),
            (7781, 8),  # DESI size
            (1000, 16),
        ]
        for in_channels, latent_dim in configs:
            vae = SkyVAE(in_channels=in_channels, latent_dim=latent_dim)
            assert vae.in_channels == in_channels
            assert vae.latent_dim == latent_dim


class TestSkyVAEEncode:
    """Test VAE encoding functionality."""

    @pytest.fixture
    def vae(self):
        """Create a small VAE for testing."""
        return SkyVAE(in_channels=100, latent_dim=8, key=jr.PRNGKey(0))

    def test_encode_single_spectrum(self, vae):
        """Test encoding a single spectrum."""
        x = jnp.ones(100) * 10.0
        mean, logvar = vae.encode(x)

        assert mean.shape == (8,)
        assert logvar.shape == (8,)
        assert jnp.all(jnp.isfinite(mean))
        assert jnp.all(jnp.isfinite(logvar))

    def test_encode_different_inputs(self, vae):
        """Test encoding with different input values."""
        inputs = [
            jnp.ones(100),
            jnp.ones(100) * 10.0,
            jnp.linspace(0, 100, 100),
            jnp.exp(jnp.linspace(-2, 2, 100)),
        ]

        for x in inputs:
            mean, logvar = vae.encode(x)
            assert mean.shape == (8,)
            assert logvar.shape == (8,)
            assert jnp.all(jnp.isfinite(mean))
            assert jnp.all(jnp.isfinite(logvar))

    def test_encode_deterministic(self, vae):
        """Test that encoding is deterministic."""
        x = jnp.ones(100) * 10.0
        mean1, logvar1 = vae.encode(x)
        mean2, logvar2 = vae.encode(x)

        assert jnp.allclose(mean1, mean2)
        assert jnp.allclose(logvar1, logvar2)

    def test_encode_batch_with_vmap(self, vae):
        """Test batch encoding using vmap."""
        batch = jnp.ones((5, 100)) * 10.0
        batch_means, batch_logvars = jax.vmap(vae.encode)(batch)

        assert batch_means.shape == (5, 8)
        assert batch_logvars.shape == (5, 8)
        assert jnp.all(jnp.isfinite(batch_means))
        assert jnp.all(jnp.isfinite(batch_logvars))


class TestSkyVAESample:
    """Test VAE sampling from latent space."""

    @pytest.fixture
    def vae(self):
        """Create a small VAE for testing."""
        return SkyVAE(in_channels=100, latent_dim=8, key=jr.PRNGKey(0))

    def test_sample_basic(self, vae):
        """Test basic sampling from latent distribution."""
        mean = jnp.zeros(8)
        logvar = jnp.zeros(8)
        key = jr.PRNGKey(42)

        latent = vae.sample(mean, logvar, key)

        assert latent.shape == (8,)
        assert jnp.all(jnp.isfinite(latent))

    def test_sample_different_keys(self, vae):
        """Test that different keys produce different samples."""
        mean = jnp.zeros(8)
        logvar = jnp.zeros(8)

        latent1 = vae.sample(mean, logvar, jr.PRNGKey(0))
        latent2 = vae.sample(mean, logvar, jr.PRNGKey(1))

        assert not jnp.allclose(latent1, latent2)

    def test_sample_reparameterization(self, vae):
        """Test reparameterization: z = mean + std * epsilon."""
        mean = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        logvar = jnp.log(jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]))
        key = jr.PRNGKey(42)

        # Sample many times and check statistics
        keys = jr.split(key, 1000)
        samples = jax.vmap(lambda k: vae.sample(mean, logvar, k))(keys)

        # Check empirical mean is close to specified mean
        empirical_mean = jnp.mean(samples, axis=0)
        assert jnp.allclose(empirical_mean, mean, atol=0.1)

        # Check empirical std is close to specified std
        empirical_std = jnp.std(samples, axis=0)
        expected_std = jnp.exp(0.5 * logvar)
        assert jnp.allclose(empirical_std, expected_std, atol=0.1)

    def test_sample_batch_with_vmap(self, vae):
        """Test batch sampling with vmap."""
        batch_size = 5
        means = jnp.zeros((batch_size, 8))
        logvars = jnp.zeros((batch_size, 8))
        keys = jr.split(jr.PRNGKey(0), batch_size)

        latents = jax.vmap(vae.sample)(means, logvars, keys)

        assert latents.shape == (batch_size, 8)
        assert jnp.all(jnp.isfinite(latents))


class TestSkyVAEDecode:
    """Test VAE decoding functionality."""

    @pytest.fixture
    def vae(self):
        """Create a small VAE for testing."""
        return SkyVAE(in_channels=100, latent_dim=8, key=jr.PRNGKey(0))

    def test_decode_basic(self, vae):
        """Test basic decoding from latent vector."""
        latent = jnp.zeros(8)
        reconstructed = vae.decode(latent)

        assert reconstructed.shape == (100,)
        assert jnp.all(jnp.isfinite(reconstructed))

    def test_decode_different_latents(self, vae):
        """Test decoding different latent vectors produces different outputs."""
        latent1 = jnp.zeros(8)
        latent2 = jnp.ones(8)

        recon1 = vae.decode(latent1)
        recon2 = vae.decode(latent2)

        assert not jnp.allclose(recon1, recon2)

    def test_decode_deterministic(self, vae):
        """Test that decoding is deterministic."""
        latent = jnp.ones(8)
        recon1 = vae.decode(latent)
        recon2 = vae.decode(latent)

        assert jnp.allclose(recon1, recon2)

    def test_decode_batch_with_vmap(self, vae):
        """Test batch decoding with vmap."""
        batch_latents = jnp.ones((5, 8))
        batch_reconstructed = jax.vmap(vae.decode)(batch_latents)

        assert batch_reconstructed.shape == (5, 100)
        assert jnp.all(jnp.isfinite(batch_reconstructed))


class TestSkyVAEForwardPass:
    """Test VAE full forward pass."""

    @pytest.fixture
    def vae(self):
        """Create a small VAE for testing."""
        return SkyVAE(in_channels=100, latent_dim=8, key=jr.PRNGKey(0))

    def test_forward_basic(self, vae):
        """Test basic forward pass."""
        x = jnp.ones(100) * 10.0
        key = jr.PRNGKey(42)

        result = vae(x, key)

        assert isinstance(result, dict)
        assert set(result.keys()) == {'mean', 'logvar', 'latent', 'output'}
        assert result['mean'].shape == (8,)
        assert result['logvar'].shape == (8,)
        assert result['latent'].shape == (8,)
        assert result['output'].shape == (100,)

        # Check all values are finite
        for key_name, value in result.items():
            assert jnp.all(jnp.isfinite(value)), f"{key_name} contains non-finite values"

    def test_forward_reconstruction(self, vae):
        """Test that forward pass produces reasonable reconstruction."""
        x = jnp.ones(100) * 10.0
        key = jr.PRNGKey(42)

        result = vae(x, key)
        reconstructed = result['output']

        # Reconstruction should be somewhat close to input
        # (this is a weak test since model is untrained)
        assert jnp.all(jnp.isfinite(reconstructed))
        assert reconstructed.shape == x.shape

    def test_forward_different_keys(self, vae):
        """Test that different keys produce different latents but same mean/logvar."""
        x = jnp.ones(100) * 10.0

        result1 = vae(x, jr.PRNGKey(0))
        result2 = vae(x, jr.PRNGKey(1))

        # Mean and logvar should be the same (deterministic encoding)
        assert jnp.allclose(result1['mean'], result2['mean'])
        assert jnp.allclose(result1['logvar'], result2['logvar'])

        # Latents should be different (stochastic sampling)
        assert not jnp.allclose(result1['latent'], result2['latent'])

    def test_forward_batch_with_vmap(self, vae):
        """Test batch forward pass with vmap."""
        batch = jnp.ones((5, 100)) * 10.0
        key = jr.PRNGKey(42)

        # vmap over batch dimension, broadcasting key
        batch_results = jax.vmap(vae, in_axes=(0, None))(batch, key)

        assert batch_results['mean'].shape == (5, 8)
        assert batch_results['logvar'].shape == (5, 8)
        assert batch_results['latent'].shape == (5, 8)
        assert batch_results['output'].shape == (5, 100)


class TestSkyVAEReconstructionQuality:
    """Test VAE reconstruction quality."""

    @pytest.fixture
    def vae(self):
        """Create a small VAE for testing."""
        return SkyVAE(in_channels=100, latent_dim=8, key=jr.PRNGKey(0))

    def test_encode_decode_consistency(self, vae):
        """Test that encode -> decode maintains shape consistency."""
        x = jnp.ones(100) * 10.0

        # Encode
        mean, logvar = vae.encode(x)

        # Sample (use mean for deterministic test)
        latent = mean  # Skip stochasticity

        # Decode
        reconstructed = vae.decode(latent)

        assert reconstructed.shape == x.shape

    def test_reconstruction_is_finite(self, vae):
        """Test that reconstructions don't produce NaN or Inf."""
        inputs = [
            jnp.ones(100),
            jnp.ones(100) * 100.0,
            jnp.linspace(0, 100, 100),
            jnp.exp(jnp.linspace(-2, 2, 100)),
        ]

        key = jr.PRNGKey(0)
        for x in inputs:
            result = vae(x, key)
            assert jnp.all(jnp.isfinite(result['output']))


class TestSkyVAEIOIntegration:
    """Test VAE integration with model I/O system."""

    @pytest.fixture
    def vae(self):
        """Create a small VAE for testing."""
        return make_SkyVAE(in_channels=100, latent_dim=8, key=jr.PRNGKey(0))

    def test_save_and_load(self, vae, tmp_path):
        """Test saving and loading a VAE."""
        save_path = tmp_path / "test_vae.eqx"

        # Prepare metadata
        meta = {
            "schema": 1,
            "arch": {
                "in_channels": 100,
                "latent_dim": 8,
            }
        }

        # Save
        save(save_path, vae, meta)
        assert save_path.exists()

        # Load
        loaded_vae, loaded_meta = load(save_path, constructor=make_SkyVAE)

        assert loaded_meta["arch"]["in_channels"] == 100
        assert loaded_meta["arch"]["latent_dim"] == 8

        # Test that loaded model produces same output
        x = jnp.ones(100) * 10.0
        key = jr.PRNGKey(42)

        orig_result = vae(x, key)
        loaded_result = loaded_vae(x, key)

        assert jnp.allclose(orig_result['mean'], loaded_result['mean'])
        assert jnp.allclose(orig_result['logvar'], loaded_result['logvar'])
        assert jnp.allclose(orig_result['latent'], loaded_result['latent'])
        assert jnp.allclose(orig_result['output'], loaded_result['output'])

    def test_load_builtin_vae(self):
        """Test loading the pretrained VAE weights."""
        vae, meta = load_builtin("vae")

        # Check metadata
        assert "arch" in meta
        assert "in_channels" in meta["arch"]
        assert "latent_dim" in meta["arch"]

        # Should be DESI sky spectra size
        assert meta["arch"]["in_channels"] == 7781
        assert meta["arch"]["latent_dim"] == 8

        # Test that it can process a spectrum
        x = jnp.ones(7781) * 10.0
        key = jr.PRNGKey(0)
        result = vae(x, key)

        assert result['output'].shape == (7781,)
        assert jnp.all(jnp.isfinite(result['output']))


class TestSkyVAEDESISize:
    """Test VAE with DESI sky spectra dimensions."""

    @pytest.fixture
    def vae_desi(self):
        """Create a VAE with DESI dimensions."""
        return SkyVAE(in_channels=7781, latent_dim=8, key=jr.PRNGKey(0))

    def test_desi_spectrum_encoding(self, vae_desi):
        """Test encoding DESI-sized spectrum."""
        spectrum = jnp.ones(7781) * 10.0
        mean, logvar = vae_desi.encode(spectrum)

        assert mean.shape == (8,)
        assert logvar.shape == (8,)

    def test_desi_spectrum_reconstruction(self, vae_desi):
        """Test full reconstruction of DESI-sized spectrum."""
        spectrum = jnp.ones(7781) * 10.0
        key = jr.PRNGKey(42)

        result = vae_desi(spectrum, key)

        assert result['output'].shape == (7781,)
        assert jnp.all(jnp.isfinite(result['output']))

    def test_desi_batch_processing(self, vae_desi):
        """Test batch processing of DESI spectra."""
        batch_size = 10
        batch = jnp.ones((batch_size, 7781)) * 10.0

        # Batch encode
        batch_means, batch_logvars = jax.vmap(vae_desi.encode)(batch)
        assert batch_means.shape == (batch_size, 8)

        # Batch sample
        keys = jr.split(jr.PRNGKey(0), batch_size)
        batch_latents = jax.vmap(vae_desi.sample)(batch_means, batch_logvars, keys)
        assert batch_latents.shape == (batch_size, 8)

        # Batch decode
        batch_reconstructed = jax.vmap(vae_desi.decode)(batch_latents)
        assert batch_reconstructed.shape == (batch_size, 7781)


class TestSkyVAEEdgeCases:
    """Test VAE with edge cases and corner conditions."""

    @pytest.fixture
    def vae(self):
        """Create a small VAE for testing."""
        return SkyVAE(in_channels=100, latent_dim=8, key=jr.PRNGKey(0))

    def test_zero_input(self, vae):
        """Test with all-zero input."""
        x = jnp.zeros(100)
        key = jr.PRNGKey(0)
        result = vae(x, key)

        assert jnp.all(jnp.isfinite(result['output']))

    def test_large_input_values(self, vae):
        """Test with large input values."""
        x = jnp.ones(100) * 1000.0
        key = jr.PRNGKey(0)
        result = vae(x, key)

        assert jnp.all(jnp.isfinite(result['output']))

    def test_small_input_values(self, vae):
        """Test with very small input values."""
        x = jnp.ones(100) * 1e-6
        key = jr.PRNGKey(0)
        result = vae(x, key)

        assert jnp.all(jnp.isfinite(result['output']))

    def test_negative_input(self, vae):
        """Test with negative input values."""
        x = jnp.ones(100) * -10.0
        key = jr.PRNGKey(0)
        result = vae(x, key)

        assert jnp.all(jnp.isfinite(result['output']))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
