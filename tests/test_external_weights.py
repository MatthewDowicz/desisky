"""
Unit tests for external weights auto-download functionality.

Tests cover:
- External weights registry configuration (all 5 models)
- Integration with load_builtin() (download/cache/inference)
"""

import pytest
from pathlib import Path

from desisky.io.model_io import load_builtin, EXTERNAL_WEIGHTS


# ---------- Registry configuration ----------

class TestExternalWeightsRegistry:
    """Test EXTERNAL_WEIGHTS configuration for all models."""

    def test_all_models_registered(self):
        """All 5 models should be in EXTERNAL_WEIGHTS."""
        expected = {"broadband", "vae", "ldm_dark", "ldm_moon", "ldm_twilight"}
        assert set(EXTERNAL_WEIGHTS.keys()) == expected

    @pytest.mark.parametrize("kind", list(EXTERNAL_WEIGHTS.keys()))
    def test_config_has_required_fields(self, kind):
        """Each entry needs url, sha256, and size_mb."""
        config = EXTERNAL_WEIGHTS[kind]
        assert "url" in config
        assert "sha256" in config and len(config["sha256"]) == 64
        assert "size_mb" in config and config["size_mb"] > 0


# ---------- Integration tests (require network or cached weights) ----------

def _skip_on_network_error(func):
    """Decorator to skip tests that fail due to network errors."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if "HTTPError" in str(type(e)) or "ConnectionError" in str(type(e)):
                pytest.skip(f"Network error (expected in offline env): {e}")
            raise
    return wrapper


class TestBroadbandIntegration:
    """Test broadband model loading and inference."""

    @_skip_on_network_error
    def test_broadband_loads_and_runs(self):
        """Load broadband from HuggingFace (or cache) and run inference."""
        import jax.numpy as jnp

        model, meta = load_builtin("broadband")
        assert meta['arch']['in_size'] == 6
        assert meta['arch']['out_size'] == 4

        y = model(jnp.ones(6))
        assert y.shape == (4,)


class TestVAEIntegration:
    """Test VAE model loading and inference."""

    @_skip_on_network_error
    def test_vae_loads_and_runs(self):
        """Load VAE from HuggingFace (or cache) and run full forward pass."""
        import jax.numpy as jnp
        import jax.random as jr

        vae, meta = load_builtin("vae")
        assert meta['arch']['in_channels'] == 7781
        assert meta['arch']['latent_dim'] == 8

        result = vae(jnp.ones(7781) * 10.0, jr.PRNGKey(0))
        assert result['output'].shape == (7781,)
        assert result['latent'].shape == (8,)

    @_skip_on_network_error
    def test_vae_encode_decode(self):
        """Test VAE encode and decode separately."""
        import jax.numpy as jnp

        vae, _ = load_builtin("vae")

        mean, logvar = vae.encode(jnp.ones(7781) * 10.0)
        assert mean.shape == (8,)
        assert jnp.all(jnp.isfinite(mean))

        reconstructed = vae.decode(mean)
        assert reconstructed.shape == (7781,)
        assert jnp.all(jnp.isfinite(reconstructed))

    @_skip_on_network_error
    def test_vae_batch_processing(self):
        """Test VAE with batch processing via vmap."""
        import jax
        import jax.numpy as jnp

        vae, _ = load_builtin("vae")
        batch_means, _ = jax.vmap(vae.encode)(jnp.ones((5, 7781)) * 10.0)
        assert batch_means.shape == (5, 8)


class TestLDMIntegration:
    """Test LDM model loading and inference."""

    @_skip_on_network_error
    def test_ldm_dark_loads_and_runs(self):
        """Load LDM dark from HuggingFace (or cache) and run forward pass."""
        import jax.numpy as jnp

        ldm, meta = load_builtin("ldm_dark")
        assert meta['arch']['meta_dim'] == 8

        # Forward: (x_noisy, sigma, metadata)
        x = jnp.zeros((1, 8))       # (in_ch=1, latent_dim=8)
        t = jnp.array(1.0)          # diffusion timestep
        cond = jnp.ones(8)          # 8 conditioning features
        y = ldm(x, t, cond)
        assert y.shape == (1, 8)

    @_skip_on_network_error
    def test_ldm_moon_loads_and_runs(self):
        """Load LDM moon from HuggingFace (or cache) and run forward pass."""
        import jax.numpy as jnp

        ldm, meta = load_builtin("ldm_moon")
        assert meta['arch']['meta_dim'] == 6

        y = ldm(jnp.zeros((1, 8)), jnp.array(1.0), jnp.ones(6))
        assert y.shape == (1, 8)

    @_skip_on_network_error
    def test_ldm_twilight_loads_and_runs(self):
        """Load LDM twilight from HuggingFace (or cache) and run forward pass."""
        import jax.numpy as jnp

        ldm, meta = load_builtin("ldm_twilight")
        assert meta['arch']['meta_dim'] == 4

        y = ldm(jnp.zeros((1, 8)), jnp.array(1.0), jnp.ones(4))
        assert y.shape == (1, 8)


class TestCacheLocation:
    """Test that weights are cached correctly."""

    @_skip_on_network_error
    def test_weights_cached_after_load(self):
        """Weights file should exist on disk after load_builtin()."""
        from desisky.data._core import default_root

        load_builtin("broadband")

        cache_file = default_root() / "models" / "broadband" / "broadband_weights.eqx"
        assert cache_file.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
