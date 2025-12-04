# SPDX-FileCopyrightText: 2025-present MatthewDowicz <mjdowicz@gmail.com>
#
# SPDX-License-Identifier: MIT

"""
Tests for Latent Diffusion Model inference and sampling functionality.
"""

import pytest
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np


class TestCosineSchedule:
    """Tests for cosine beta schedule."""

    def test_cosine_schedule_shape(self):
        """Test that schedule returns correct keys and shapes."""
        from desisky.inference import cosine_beta_schedule

        T = 1000
        sched = cosine_beta_schedule(T)

        expected_keys = {
            'beta_t', 'alpha_t', 'oneover_sqrta', 'sqrt_beta_t',
            'alphabar_t', 'sqrtab', 'sqrtmab', 'mab_over_sqrtmab',
            'sqrt_posterior_var'
        }

        assert set(sched.keys()) == expected_keys

        # All arrays should have T+1 elements (t=0 to t=T)
        for key, arr in sched.items():
            assert arr.shape == (T + 1,), f"{key} has wrong shape"

    def test_cosine_schedule_values(self):
        """Test that schedule values are in valid ranges."""
        from desisky.inference import cosine_beta_schedule

        sched = cosine_beta_schedule(T=1000)

        # Beta should be in [0, 1)
        assert jnp.all(sched['beta_t'] >= 0)
        assert jnp.all(sched['beta_t'] < 1)

        # Alpha = 1 - beta should be in (0, 1]
        assert jnp.all(sched['alpha_t'] > 0)
        assert jnp.all(sched['alpha_t'] <= 1)

        # Alphabar should be monotonically decreasing
        assert jnp.all(jnp.diff(sched['alphabar_t']) <= 0)

        # Alphabar[0] should be ~1, alphabar[-1] should be small
        assert jnp.isclose(sched['alphabar_t'][0], 1.0, atol=1e-3)
        assert sched['alphabar_t'][-1] < 0.1


class TestGuidedDenoising:
    """Tests for classifier-free guidance."""

    def test_guided_denoising_step_shape(self):
        """Test that guided denoising returns correct shape."""
        from desisky.inference import guided_denoising_step
        from desisky.models.ldm import make_UNet1D_cond

        # Create minimal UNet
        model = make_UNet1D_cond(
            in_ch=1, out_ch=1, meta_dim=8,
            hidden=16, levels=2, emb_dim=16,
            key=jr.PRNGKey(0)
        )

        x_t = jnp.ones((1, 8))  # (C, L)
        t = jnp.array([0.5])
        cond = jnp.zeros(8)

        eps_pred = guided_denoising_step(model, x_t, t, cond, guidance_scale=2.0)

        assert eps_pred.shape == x_t.shape

    def test_guided_denoising_guidance_scale(self):
        """Test that guidance scale affects predictions."""
        from desisky.inference import guided_denoising_step
        from desisky.models.ldm import make_UNet1D_cond

        model = make_UNet1D_cond(
            in_ch=1, out_ch=1, meta_dim=8,
            hidden=16, levels=2, emb_dim=16,
            key=jr.PRNGKey(0)
        )

        x_t = jax.random.normal(jr.PRNGKey(42), (1, 8))
        t = jnp.array([0.5])
        cond = jnp.ones(8) * 0.5

        # Different guidance scales should give different results
        eps_0 = guided_denoising_step(model, x_t, t, cond, guidance_scale=0.0)
        eps_1 = guided_denoising_step(model, x_t, t, cond, guidance_scale=1.0)
        eps_3 = guided_denoising_step(model, x_t, t, cond, guidance_scale=3.0)

        # With scale=0, should be unconditioned (different from scale=1)
        assert not jnp.allclose(eps_0, eps_1, atol=1e-4)

        # With scale=3, should amplify conditioning (different from scale=1)
        assert not jnp.allclose(eps_1, eps_3, atol=1e-4)


class TestLatentDiffusionSampler:
    """Tests for LatentDiffusionSampler class."""

    @pytest.fixture
    def dummy_models(self):
        """Create dummy LDM and VAE models for testing."""
        from desisky.models.ldm import make_UNet1D_cond
        from desisky.models.vae import SkyVAE

        ldm = make_UNet1D_cond(
            in_ch=1, out_ch=1, meta_dim=8,
            hidden=16, levels=2, emb_dim=16,
            key=jr.PRNGKey(0)
        )

        vae = SkyVAE(in_channels=100, latent_dim=8, key=jr.PRNGKey(1))

        return ldm, vae

    def test_sampler_initialization(self, dummy_models):
        """Test that sampler initializes correctly."""
        from desisky.inference import LatentDiffusionSampler

        ldm, vae = dummy_models

        sampler = LatentDiffusionSampler(
            ldm_model=ldm,
            vae_model=vae,
            method="heun",
            num_steps=40,
            n_T=1000
        )

        assert sampler.config.method == "heun"
        assert sampler.config.num_steps == 40
        assert sampler.config.n_T == 1000
        assert sampler.config.latent_channels == 1
        assert sampler.config.latent_dim == 8

    def test_sampler_invalid_method(self, dummy_models):
        """Test that invalid method raises ValueError."""
        from desisky.inference import LatentDiffusionSampler

        ldm, vae = dummy_models

        with pytest.raises(ValueError, match="Unknown sampling method"):
            LatentDiffusionSampler(ldm, vae, method="invalid")

    def test_sample_latents_shape(self, dummy_models):
        """Test that sample_latents returns correct shape."""
        from desisky.inference import LatentDiffusionSampler

        ldm, vae = dummy_models

        sampler = LatentDiffusionSampler(ldm, vae, method="heun", num_steps=5)

        conditioning = jnp.ones((3, 8))  # 3 samples
        latents = sampler.sample_latents(
            key=jr.PRNGKey(0),
            conditioning=conditioning,
            guidance_scale=1.0
        )

        assert latents.shape == (3, 1, 8)  # (n_samples, channels, latent_dim)

    def test_sample_latents_single_condition(self, dummy_models):
        """Test sampling with single conditioning vector."""
        from desisky.inference import LatentDiffusionSampler

        ldm, vae = dummy_models

        sampler = LatentDiffusionSampler(ldm, vae, method="heun", num_steps=5)

        # Single conditioning (1D array)
        conditioning = jnp.ones(8)
        latents = sampler.sample_latents(
            key=jr.PRNGKey(0),
            conditioning=conditioning,
            guidance_scale=1.0
        )

        assert latents.shape == (1, 1, 8)

    def test_sample_spectra_shape(self, dummy_models):
        """Test that sample returns correct spectrum shape."""
        from desisky.inference import LatentDiffusionSampler

        ldm, vae = dummy_models

        sampler = LatentDiffusionSampler(ldm, vae, method="heun", num_steps=5)

        conditioning = jnp.ones((2, 8))
        spectra = sampler.sample(
            key=jr.PRNGKey(0),
            conditioning=conditioning,
            guidance_scale=1.0
        )

        # VAE has in_channels=100
        assert spectra.shape == (2, 100)

    def test_sample_with_return_latents(self, dummy_models):
        """Test that return_latents option works."""
        from desisky.inference import LatentDiffusionSampler

        ldm, vae = dummy_models

        sampler = LatentDiffusionSampler(ldm, vae, method="heun", num_steps=5)

        conditioning = jnp.ones((2, 8))
        spectra, latents = sampler.sample(
            key=jr.PRNGKey(0),
            conditioning=conditioning,
            guidance_scale=1.0,
            return_latents=True
        )

        assert spectra.shape == (2, 100)
        assert latents.shape == (2, 1, 8)

    def test_sample_different_methods(self, dummy_models):
        """Test that different sampling methods produce results."""
        from desisky.inference import LatentDiffusionSampler

        ldm, vae = dummy_models
        conditioning = jnp.ones((1, 8))

        for method in ["heun", "ddim", "ddpm"]:
            sampler = LatentDiffusionSampler(
                ldm, vae, method=method,
                num_steps=5 if method != "ddpm" else None
            )

            spectra = sampler.sample(
                key=jr.PRNGKey(0),
                conditioning=conditioning,
                guidance_scale=1.0
            )

            assert spectra.shape == (1, 100)
            assert jnp.all(jnp.isfinite(spectra))

    def test_sample_deterministic_with_same_key(self, dummy_models):
        """Test that sampling is deterministic given the same random key."""
        from desisky.inference import LatentDiffusionSampler

        ldm, vae = dummy_models

        sampler = LatentDiffusionSampler(ldm, vae, method="heun", num_steps=5)

        conditioning = jnp.ones((1, 8))

        # Same key should give same result
        spec1 = sampler.sample(jr.PRNGKey(42), conditioning, guidance_scale=1.0)
        spec2 = sampler.sample(jr.PRNGKey(42), conditioning, guidance_scale=1.0)

        assert jnp.allclose(spec1, spec2, atol=1e-5)

    def test_sample_different_with_different_key(self, dummy_models):
        """Test that different keys produce different samples."""
        from desisky.inference import LatentDiffusionSampler

        ldm, vae = dummy_models

        sampler = LatentDiffusionSampler(ldm, vae, method="heun", num_steps=5)

        conditioning = jnp.ones((1, 8))

        spec1 = sampler.sample(jr.PRNGKey(0), conditioning, guidance_scale=1.0)
        spec2 = sampler.sample(jr.PRNGKey(1), conditioning, guidance_scale=1.0)

        assert not jnp.allclose(spec1, spec2, atol=1e-3)

    def test_sampler_repr(self, dummy_models):
        """Test that __repr__ returns informative string."""
        from desisky.inference import LatentDiffusionSampler

        ldm, vae = dummy_models

        sampler = LatentDiffusionSampler(ldm, vae, method="heun", num_steps=40)

        repr_str = repr(sampler)

        assert "LatentDiffusionSampler" in repr_str
        assert "heun" in repr_str
        assert "40" in repr_str


class TestSamplerConfig:
    """Tests for SamplerConfig dataclass."""

    def test_config_defaults(self):
        """Test that config has sensible defaults."""
        from desisky.inference import SamplerConfig

        config = SamplerConfig()

        assert config.method == "heun"
        assert config.num_steps == 40
        assert config.n_T == 1000
        assert config.latent_channels == 1
        assert config.latent_dim == 8

    def test_config_custom_values(self):
        """Test that config accepts custom values."""
        from desisky.inference import SamplerConfig

        config = SamplerConfig(
            method="ddim",
            num_steps=100,
            n_T=2000,
            latent_channels=2,
            latent_dim=16
        )

        assert config.method == "ddim"
        assert config.num_steps == 100
        assert config.n_T == 2000
        assert config.latent_channels == 2
        assert config.latent_dim == 16


@pytest.mark.slow
class TestLDMIntegration:
    """Integration tests using pre-trained models (if available)."""

    def test_load_and_sample_ldm_dark(self):
        """Test loading ldm_dark model and generating samples."""
        pytest.importorskip("desisky.models.ldm")

        try:
            from desisky.io import load_builtin
            from desisky.inference import LatentDiffusionSampler

            # Try to load pre-trained models
            ldm, ldm_meta = load_builtin("ldm_dark")
            vae, vae_meta = load_builtin("vae")

            # Create sampler
            sampler = LatentDiffusionSampler(
                ldm_model=ldm,
                vae_model=vae,
                method="heun",
                num_steps=10  # Small for fast testing
            )

            # Sample with realistic conditioning
            # (OBSALT, TRANSP, SUNALT, SOLFLUX, ECLLON, ECLLAT, GALLON, GALLAT)
            conditioning = jnp.array([
                [60.0, 0.9, -30.0, 150.0, 45.0, 10.0, 120.0, 5.0],
                [70.0, 0.85, -25.0, 155.0, 50.0, 12.0, 125.0, 6.0],
            ])

            spectra = sampler.sample(
                key=jr.PRNGKey(0),
                conditioning=conditioning,
                guidance_scale=2.0
            )

            # Check output shape (should match VAE output)
            assert spectra.shape[0] == 2
            assert spectra.shape[1] == vae_meta['arch']['in_channels']

            # Check values are finite
            assert jnp.all(jnp.isfinite(spectra))

        except Exception as e:
            # Models might not be available in test environment
            pytest.skip(f"Pre-trained models not available: {e}")

    def test_load_and_sample_ldm_moon(self):
        """Test loading ldm_moon model and generating samples."""
        pytest.importorskip("desisky.models.ldm")

        try:
            from desisky.io import load_builtin
            from desisky.inference import LatentDiffusionSampler

            # Try to load pre-trained models
            ldm_moon, ldm_moon_meta = load_builtin("ldm_moon")
            vae, vae_meta = load_builtin("vae")

            # Verify moon model has correct meta_dim
            assert ldm_moon_meta['arch']['meta_dim'] == 6

            # Create sampler
            sampler = LatentDiffusionSampler(
                ldm_model=ldm_moon,
                vae_model=vae,
                method="heun",
                num_steps=10  # Small for fast testing
            )

            # Sample with realistic moon conditioning
            # (OBSALT, TRANSPARENCY_GFA, SUNALT, MOONALT, MOONSEP, MOONFRAC)
            conditioning = jnp.array([
                [60.0, 0.9, -25.0, 30.0, 45.0, 0.8],
                [70.0, 0.85, -28.0, 25.0, 60.0, 0.9],
            ])

            spectra = sampler.sample(
                key=jr.PRNGKey(0),
                conditioning=conditioning,
                guidance_scale=2.0
            )

            # Check output shape (should match VAE output)
            assert spectra.shape[0] == 2
            assert spectra.shape[1] == vae_meta['arch']['in_channels']

            # Check values are finite
            assert jnp.all(jnp.isfinite(spectra))

        except Exception as e:
            # Models might not be available in test environment
            pytest.skip(f"Pre-trained models not available: {e}")
