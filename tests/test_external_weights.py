"""
Unit tests for external weights auto-download functionality.

Tests cover:
- External weights registry configuration
- Integration with load_builtin()
- Caching behavior
"""

import pytest
from pathlib import Path

from desisky.io.model_io import load_builtin, EXTERNAL_WEIGHTS


class TestExternalWeightsRegistry:
    """Test EXTERNAL_WEIGHTS configuration."""

    def test_vae_in_external_weights(self):
        """Test that VAE is registered in EXTERNAL_WEIGHTS."""
        assert "vae" in EXTERNAL_WEIGHTS

    def test_vae_config_structure(self):
        """Test VAE external weights config has required fields."""
        vae_config = EXTERNAL_WEIGHTS["vae"]
        assert "url" in vae_config
        assert "sha256" in vae_config
        assert "size_mb" in vae_config

    def test_vae_url_is_huggingface(self):
        """Test VAE weights are hosted on Hugging Face."""
        vae_config = EXTERNAL_WEIGHTS["vae"]
        assert "huggingface.co" in vae_config["url"]
        assert "vae_weights.eqx" in vae_config["url"]

    def test_vae_has_valid_checksum(self):
        """Test VAE config includes SHA-256 checksum."""
        vae_config = EXTERNAL_WEIGHTS["vae"]
        assert vae_config["sha256"] is not None
        assert len(vae_config["sha256"]) == 64  # SHA-256 is 64 hex chars
        # Should be valid hex
        assert all(c in '0123456789abcdef' for c in vae_config["sha256"].lower())

    def test_vae_size_reasonable(self):
        """Test VAE size is reasonable (not negative, not impossibly large)."""
        vae_config = EXTERNAL_WEIGHTS["vae"]
        assert 0 < vae_config["size_mb"] < 1000  # Between 0 and 1GB


class TestLoadBuiltinIntegration:
    """Test load_builtin() integration with external weights."""

    def test_broadband_loads_from_package(self):
        """Test that broadband (small model) loads from package, not download."""
        # Broadband should NOT be in EXTERNAL_WEIGHTS
        assert "broadband" not in EXTERNAL_WEIGHTS

        # Should load successfully from packaged weights
        model, meta = load_builtin("broadband")
        assert meta['arch']['in_size'] == 6
        assert meta['arch']['out_size'] == 4

    def test_vae_loads_successfully(self):
        """
        Test that VAE loads (from cache or download).

        This test assumes either:
        1. Weights are already cached from a previous run
        2. Network is available to download from Hugging Face

        If this test fails with network errors, it's expected in
        offline environments.
        """
        try:
            vae, meta = load_builtin("vae")

            # Verify architecture
            assert meta['arch']['in_channels'] == 7781
            assert meta['arch']['latent_dim'] == 8

            # Verify can run inference
            import jax.numpy as jnp
            import jax.random as jr

            x = jnp.ones(7781) * 10.0
            result = vae(x, jr.PRNGKey(0))

            assert result['output'].shape == (7781,)
            assert result['latent'].shape == (8,)
            assert result['mean'].shape == (8,)
            assert result['logvar'].shape == (8,)

        except Exception as e:
            # If download fails due to network, skip
            if "HTTPError" in str(type(e)) or "ConnectionError" in str(type(e)):
                pytest.skip(f"Network error (expected in offline env): {e}")
            raise

    def test_vae_cache_location(self):
        """Test that VAE weights are cached in expected location."""
        from desisky.data._core import default_root

        # Load VAE (from cache or download)
        try:
            vae, meta = load_builtin("vae")

            # Check cache exists
            cache_dir = default_root() / "models" / "vae"
            cache_file = cache_dir / "vae_weights.eqx"

            assert cache_file.exists(), "VAE weights should be cached after first load"
            assert cache_file.stat().st_size > 1_000_000, "Cache file should be >1MB"

        except Exception as e:
            if "HTTPError" in str(type(e)) or "ConnectionError" in str(type(e)):
                pytest.skip(f"Network error: {e}")
            raise


class TestVAEFunctionality:
    """Test VAE model functionality after loading."""

    def test_vae_encode_decode(self):
        """Test VAE encode and decode operations."""
        try:
            vae, _ = load_builtin("vae")

            import jax.numpy as jnp

            # Test encoding
            x = jnp.ones(7781) * 10.0
            mean, logvar = vae.encode(x)

            assert mean.shape == (8,)
            assert logvar.shape == (8,)
            assert jnp.all(jnp.isfinite(mean))
            assert jnp.all(jnp.isfinite(logvar))

            # Test decoding
            reconstructed = vae.decode(mean)
            assert reconstructed.shape == (7781,)
            assert jnp.all(jnp.isfinite(reconstructed))

        except Exception as e:
            if "HTTPError" in str(type(e)) or "ConnectionError" in str(type(e)):
                pytest.skip(f"Network error: {e}")
            raise

    def test_vae_batch_processing(self):
        """Test VAE with batch processing via vmap."""
        try:
            vae, _ = load_builtin("vae")

            import jax
            import jax.numpy as jnp

            # Batch of 5 spectra
            batch = jnp.ones((5, 7781)) * 10.0

            # Batch encode
            batch_means, batch_logvars = jax.vmap(vae.encode)(batch)

            assert batch_means.shape == (5, 8)
            assert batch_logvars.shape == (5, 8)

            # Batch decode
            batch_reconstructed = jax.vmap(vae.decode)(batch_means)
            assert batch_reconstructed.shape == (5, 7781)

        except Exception as e:
            if "HTTPError" in str(type(e)) or "ConnectionError" in str(type(e)):
                pytest.skip(f"Network error: {e}")
            raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
