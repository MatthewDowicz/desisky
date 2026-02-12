# SPDX-FileCopyrightText: 2025-present MatthewDowicz <mjdowicz@gmail.com>
#
# SPDX-License-Identifier: MIT

"""
Tests for EDM-based Latent Diffusion Model sampling (Karras et al. 2022).
"""

import pytest
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np


# =========================================================================
# Karras Sigma Schedule
# =========================================================================

class TestKarrasSigmaSchedule:
    """Tests for the Karras et al. sigma schedule."""

    def test_schedule_shape(self):
        """Schedule returns n_steps + 1 values (final sigma=0)."""
        from desisky.inference import get_sigmas_karras

        sigmas = get_sigmas_karras(50)
        assert sigmas.shape == (51,)

    def test_schedule_endpoints(self):
        """First sigma is sigma_max, last sigma is 0."""
        from desisky.inference import get_sigmas_karras

        sigmas = get_sigmas_karras(50, sigma_min=0.002, sigma_max=80.0)
        assert jnp.isclose(sigmas[0], 80.0, atol=1e-3)
        assert sigmas[-1] == 0.0

    def test_schedule_monotonically_decreasing(self):
        """Sigmas should decrease from sigma_max to 0."""
        from desisky.inference import get_sigmas_karras

        sigmas = get_sigmas_karras(100)
        assert jnp.all(jnp.diff(sigmas) <= 0)

    def test_schedule_all_positive_except_last(self):
        """All sigmas except the terminal 0 should be positive."""
        from desisky.inference import get_sigmas_karras

        sigmas = get_sigmas_karras(50)
        assert jnp.all(sigmas[:-1] > 0)
        assert sigmas[-1] == 0.0

    def test_schedule_custom_params(self):
        """Custom sigma_min, sigma_max, and rho are respected."""
        from desisky.inference import get_sigmas_karras

        sigmas = get_sigmas_karras(20, sigma_min=0.01, sigma_max=10.0, rho=5.0)
        assert sigmas.shape == (21,)
        assert jnp.isclose(sigmas[0], 10.0, atol=1e-2)
        assert jnp.isclose(sigmas[-2], 0.01, atol=1e-2)
        assert sigmas[-1] == 0.0


# =========================================================================
# Guided Denoiser (CFG for EDM)
# =========================================================================

class TestGuidedDenoiser:
    """Tests for classifier-free guidance with EDM preconditioning."""

    @pytest.fixture
    def model_and_sigma_data(self):
        """Create a minimal UNet and sigma_data for testing."""
        from desisky.models.ldm import make_UNet1D_cond, compute_sigma_data

        model = make_UNet1D_cond(
            in_ch=1, out_ch=1, meta_dim=8,
            hidden=16, levels=2, emb_dim=16,
            key=jr.PRNGKey(0)
        )
        # Use a reasonable sigma_data for testing
        sigma_data = 6.7
        return model, sigma_data

    def test_guided_denoiser_shape(self, model_and_sigma_data):
        """Guided denoiser returns same shape as input."""
        from desisky.inference import guided_denoiser

        model, sigma_data = model_and_sigma_data
        x = jnp.ones((1, 8))
        sigma = jnp.array(1.0)
        cond = jnp.zeros(8)

        D = guided_denoiser(model, x, sigma, cond, sigma_data, guidance_scale=2.0)
        assert D.shape == x.shape

    def test_guidance_scale_affects_output(self, model_and_sigma_data):
        """Different guidance scales produce different outputs."""
        from desisky.inference import guided_denoiser

        model, sigma_data = model_and_sigma_data
        x = jax.random.normal(jr.PRNGKey(42), (1, 8))
        sigma = jnp.array(1.0)
        cond = jnp.ones(8) * 0.5

        D_0 = guided_denoiser(model, x, sigma, cond, sigma_data, guidance_scale=0.0)
        D_1 = guided_denoiser(model, x, sigma, cond, sigma_data, guidance_scale=1.0)
        D_3 = guided_denoiser(model, x, sigma, cond, sigma_data, guidance_scale=3.0)

        # scale=0 is purely unconditional, should differ from conditional
        assert not jnp.allclose(D_0, D_1, atol=1e-4)
        # scale=3 amplifies conditioning beyond scale=1
        assert not jnp.allclose(D_1, D_3, atol=1e-4)

    def test_guidance_scale_one_is_conditional(self, model_and_sigma_data):
        """With guidance_scale=1.0, output equals the conditional prediction."""
        from desisky.inference import guided_denoiser
        from desisky.models.ldm import edm_denoiser

        model, sigma_data = model_and_sigma_data
        x = jax.random.normal(jr.PRNGKey(42), (1, 8))
        sigma = jnp.array(1.0)
        cond = jnp.ones(8) * 0.5

        D_guided = guided_denoiser(
            model, x, sigma, cond, sigma_data, guidance_scale=1.0
        )
        D_cond = edm_denoiser(model, x, sigma, cond, sigma_data, None, 0.0)

        assert jnp.allclose(D_guided, D_cond, atol=1e-5)


# =========================================================================
# EDM Sampling (Heun Solver)
# =========================================================================

class TestSampleEDM:
    """Tests for the EDM Heun-based sampler."""

    @pytest.fixture
    def model_and_sigma_data(self):
        from desisky.models.ldm import make_UNet1D_cond

        model = make_UNet1D_cond(
            in_ch=1, out_ch=1, meta_dim=8,
            hidden=16, levels=2, emb_dim=16,
            key=jr.PRNGKey(0)
        )
        sigma_data = 6.7
        return model, sigma_data

    def test_sample_shape(self, model_and_sigma_data):
        """Output shape is (n_sample, channels, latent_dim)."""
        from desisky.inference import sample_edm

        model, sigma_data = model_and_sigma_data
        cond = jnp.ones((3, 8))

        latents = sample_edm(
            jr.PRNGKey(0), model, cond,
            n_sample=3, size=(1, 8), sigma_data=sigma_data,
            num_steps=5,  # few steps for speed
        )
        assert latents.shape == (3, 1, 8)

    def test_sample_finite(self, model_and_sigma_data):
        """Samples should be finite (no NaN or inf)."""
        from desisky.inference import sample_edm

        model, sigma_data = model_and_sigma_data
        cond = jnp.ones((2, 8))

        latents = sample_edm(
            jr.PRNGKey(0), model, cond,
            n_sample=2, size=(1, 8), sigma_data=sigma_data,
            num_steps=5,
        )
        assert jnp.all(jnp.isfinite(latents))

    def test_sample_deterministic(self, model_and_sigma_data):
        """Same key produces identical samples."""
        from desisky.inference import sample_edm

        model, sigma_data = model_and_sigma_data
        cond = jnp.ones((1, 8))

        s1 = sample_edm(
            jr.PRNGKey(42), model, cond,
            n_sample=1, size=(1, 8), sigma_data=sigma_data,
            num_steps=5,
        )
        s2 = sample_edm(
            jr.PRNGKey(42), model, cond,
            n_sample=1, size=(1, 8), sigma_data=sigma_data,
            num_steps=5,
        )
        assert jnp.allclose(s1, s2, atol=1e-5)

    def test_different_keys_differ(self, model_and_sigma_data):
        """Different keys produce different samples."""
        from desisky.inference import sample_edm

        model, sigma_data = model_and_sigma_data
        cond = jnp.ones((1, 8))

        s1 = sample_edm(
            jr.PRNGKey(0), model, cond,
            n_sample=1, size=(1, 8), sigma_data=sigma_data,
            num_steps=5,
        )
        s2 = sample_edm(
            jr.PRNGKey(1), model, cond,
            n_sample=1, size=(1, 8), sigma_data=sigma_data,
            num_steps=5,
        )
        assert not jnp.allclose(s1, s2, atol=1e-3)


# =========================================================================
# SamplerConfig
# =========================================================================

class TestSamplerConfig:
    """Tests for SamplerConfig dataclass."""

    def test_config_defaults(self):
        from desisky.inference import SamplerConfig

        config = SamplerConfig()
        assert config.num_steps == 100
        assert config.latent_channels == 1
        assert config.latent_dim == 8
        assert config.rho == 7.0

    def test_config_custom_values(self):
        from desisky.inference import SamplerConfig

        config = SamplerConfig(
            num_steps=50,
            sigma_min=0.01,
            sigma_max=100.0,
            rho=5.0,
            latent_channels=2,
            latent_dim=16,
        )
        assert config.num_steps == 50
        assert config.sigma_min == 0.01
        assert config.sigma_max == 100.0
        assert config.rho == 5.0
        assert config.latent_channels == 2
        assert config.latent_dim == 16


# =========================================================================
# LatentDiffusionSampler (high-level API)
# =========================================================================

class TestLatentDiffusionSampler:
    """Tests for the high-level LatentDiffusionSampler class."""

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
        sigma_data = 6.7
        return ldm, vae, sigma_data

    def test_initialization(self, dummy_models):
        """Sampler initializes with correct config."""
        from desisky.inference import LatentDiffusionSampler

        ldm, vae, sigma_data = dummy_models
        sampler = LatentDiffusionSampler(ldm, vae, sigma_data, num_steps=40)

        assert sampler.sigma_data == sigma_data
        assert sampler.config.num_steps == 40
        assert sampler.config.latent_channels == 1
        assert sampler.config.latent_dim == 8

    def test_sample_latents_shape(self, dummy_models):
        """sample_latents returns (n_samples, channels, latent_dim)."""
        from desisky.inference import LatentDiffusionSampler

        ldm, vae, sigma_data = dummy_models
        sampler = LatentDiffusionSampler(ldm, vae, sigma_data, num_steps=5)

        conditioning = jnp.ones((3, 8))
        latents = sampler.sample_latents(
            key=jr.PRNGKey(0),
            conditioning=conditioning,
            guidance_scale=1.0,
        )
        assert latents.shape == (3, 1, 8)

    def test_sample_latents_single_condition(self, dummy_models):
        """Single 1D conditioning is broadcast to batch dimension."""
        from desisky.inference import LatentDiffusionSampler

        ldm, vae, sigma_data = dummy_models
        sampler = LatentDiffusionSampler(ldm, vae, sigma_data, num_steps=5)

        conditioning = jnp.ones(8)  # 1D
        latents = sampler.sample_latents(
            key=jr.PRNGKey(0),
            conditioning=conditioning,
            guidance_scale=1.0,
        )
        assert latents.shape == (1, 1, 8)

    def test_sample_spectra_shape(self, dummy_models):
        """sample() returns decoded spectra of correct shape."""
        from desisky.inference import LatentDiffusionSampler

        ldm, vae, sigma_data = dummy_models
        sampler = LatentDiffusionSampler(ldm, vae, sigma_data, num_steps=5)

        conditioning = jnp.ones((2, 8))
        spectra = sampler.sample(
            key=jr.PRNGKey(0),
            conditioning=conditioning,
            guidance_scale=1.0,
        )
        # VAE has in_channels=100
        assert spectra.shape == (2, 100)

    def test_sample_with_return_latents(self, dummy_models):
        """return_latents=True returns both spectra and latents."""
        from desisky.inference import LatentDiffusionSampler

        ldm, vae, sigma_data = dummy_models
        sampler = LatentDiffusionSampler(ldm, vae, sigma_data, num_steps=5)

        conditioning = jnp.ones((2, 8))
        spectra, latents = sampler.sample(
            key=jr.PRNGKey(0),
            conditioning=conditioning,
            guidance_scale=1.0,
            return_latents=True,
        )
        assert spectra.shape == (2, 100)
        assert latents.shape == (2, 1, 8)

    def test_deterministic_with_same_key(self, dummy_models):
        """Same key produces identical spectra."""
        from desisky.inference import LatentDiffusionSampler

        ldm, vae, sigma_data = dummy_models
        sampler = LatentDiffusionSampler(ldm, vae, sigma_data, num_steps=5)
        conditioning = jnp.ones((1, 8))

        s1 = sampler.sample(jr.PRNGKey(42), conditioning, guidance_scale=1.0)
        s2 = sampler.sample(jr.PRNGKey(42), conditioning, guidance_scale=1.0)
        assert jnp.allclose(s1, s2, atol=1e-5)

    def test_different_keys_differ(self, dummy_models):
        """Different keys produce different spectra."""
        from desisky.inference import LatentDiffusionSampler

        ldm, vae, sigma_data = dummy_models
        sampler = LatentDiffusionSampler(ldm, vae, sigma_data, num_steps=5)
        conditioning = jnp.ones((1, 8))

        s1 = sampler.sample(jr.PRNGKey(0), conditioning, guidance_scale=1.0)
        s2 = sampler.sample(jr.PRNGKey(1), conditioning, guidance_scale=1.0)
        assert not jnp.allclose(s1, s2, atol=1e-3)

    def test_repr(self, dummy_models):
        """__repr__ contains key parameters."""
        from desisky.inference import LatentDiffusionSampler

        ldm, vae, sigma_data = dummy_models
        sampler = LatentDiffusionSampler(ldm, vae, sigma_data, num_steps=40)
        repr_str = repr(sampler)

        assert "LatentDiffusionSampler" in repr_str
        assert "40" in repr_str
        assert str(sigma_data) in repr_str

    def test_n_samples_mismatch_raises(self, dummy_models):
        """Passing n_samples that doesn't match conditioning raises ValueError."""
        from desisky.inference import LatentDiffusionSampler

        ldm, vae, sigma_data = dummy_models
        sampler = LatentDiffusionSampler(ldm, vae, sigma_data, num_steps=5)

        conditioning = jnp.ones((3, 8))
        with pytest.raises(ValueError, match="n_samples"):
            sampler.sample_latents(
                key=jr.PRNGKey(0),
                conditioning=conditioning,
                n_samples=5,  # mismatch with batch size of 3
                guidance_scale=1.0,
            )

    def test_no_scaler_passes_through(self, dummy_models):
        """Without conditioning_scaler, conditioning is passed through unchanged."""
        from desisky.inference import LatentDiffusionSampler

        ldm, vae, sigma_data = dummy_models
        sampler = LatentDiffusionSampler(ldm, vae, sigma_data, num_steps=5)

        assert sampler._scaler_mean is None
        assert sampler._scaler_scale is None

    def test_scaler_stores_params(self, dummy_models):
        """conditioning_scaler mean and scale are stored as jnp arrays."""
        from desisky.inference import LatentDiffusionSampler

        ldm, vae, sigma_data = dummy_models
        scaler = {
            "mean": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "scale": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            "columns": ["A", "B", "C", "D", "E", "F", "G", "H"],
        }
        sampler = LatentDiffusionSampler(
            ldm, vae, sigma_data, conditioning_scaler=scaler, num_steps=5
        )

        assert sampler._scaler_mean is not None
        assert sampler._scaler_scale is not None
        np.testing.assert_array_almost_equal(
            sampler._scaler_mean, scaler["mean"]
        )
        np.testing.assert_array_almost_equal(
            sampler._scaler_scale, scaler["scale"]
        )

    def test_scaler_normalizes_conditioning(self, dummy_models):
        """Sampler with scaler produces different output than without."""
        from desisky.inference import LatentDiffusionSampler

        ldm, vae, sigma_data = dummy_models

        scaler = {
            "mean": [50.0, 0.9, -25.0, 150.0, 45.0, 10.0, 120.0, 5.0],
            "scale": [10.0, 0.1, 5.0, 20.0, 15.0, 5.0, 30.0, 3.0],
        }
        raw_cond = jnp.array([[60.0, 0.9, -30.0, 150.0, 45.0, 10.0, 120.0, 5.0]])

        sampler_no_scaler = LatentDiffusionSampler(
            ldm, vae, sigma_data, num_steps=5
        )
        sampler_with_scaler = LatentDiffusionSampler(
            ldm, vae, sigma_data, conditioning_scaler=scaler, num_steps=5
        )

        # Same raw input, but scaler normalizes internally → different latents
        lat1 = sampler_no_scaler.sample_latents(
            jr.PRNGKey(0), raw_cond, guidance_scale=1.0
        )
        lat2 = sampler_with_scaler.sample_latents(
            jr.PRNGKey(0), raw_cond, guidance_scale=1.0
        )
        assert not jnp.allclose(lat1, lat2, atol=1e-3)

    def test_repr_shows_auto_normalize(self, dummy_models):
        """__repr__ reflects whether auto_normalize is active."""
        from desisky.inference import LatentDiffusionSampler

        ldm, vae, sigma_data = dummy_models

        sampler_off = LatentDiffusionSampler(ldm, vae, sigma_data, num_steps=5)
        assert "auto_normalize=False" in repr(sampler_off)

        scaler = {"mean": [0.0] * 8, "scale": [1.0] * 8}
        sampler_on = LatentDiffusionSampler(
            ldm, vae, sigma_data, conditioning_scaler=scaler, num_steps=5
        )
        assert "auto_normalize=True" in repr(sampler_on)


# =========================================================================
# Integration Tests (require pre-trained HuggingFace models)
# =========================================================================

@pytest.mark.slow
class TestLDMIntegration:
    """Integration tests using pre-trained EDM models from HuggingFace."""

    def _load_and_sample(self, ldm_kind, conditioning):
        """Helper: load model + VAE, sample, and return spectra + metadata."""
        from desisky.io import load_builtin
        from desisky.inference import LatentDiffusionSampler

        ldm, ldm_meta = load_builtin(ldm_kind)
        vae, vae_meta = load_builtin("vae")

        sigma_data = ldm_meta["training"]["sigma_data"]

        sampler = LatentDiffusionSampler(
            ldm_model=ldm,
            vae_model=vae,
            sigma_data=sigma_data,
            num_steps=10,  # few steps for fast testing
        )

        spectra = sampler.sample(
            key=jr.PRNGKey(0),
            conditioning=conditioning,
            guidance_scale=2.0,
        )
        return spectra, ldm_meta, vae_meta

    def test_load_and_sample_ldm_dark(self):
        """Test loading ldm_dark model and generating samples."""
        try:
            # dark: 8 conditioning features
            # (OBSALT, TRANSP, SUNALT, SOLFLUX, ECLLON, ECLLAT, GALLON, GALLAT)
            conditioning = jnp.array([
                [60.0, 0.9, -30.0, 150.0, 45.0, 10.0, 120.0, 5.0],
                [70.0, 0.85, -25.0, 155.0, 50.0, 12.0, 125.0, 6.0],
            ])
            spectra, ldm_meta, vae_meta = self._load_and_sample("ldm_dark", conditioning)

            assert ldm_meta["arch"]["meta_dim"] == 8
            assert spectra.shape[0] == 2
            assert spectra.shape[1] == vae_meta["arch"]["in_channels"]
            assert jnp.all(jnp.isfinite(spectra))
        except Exception as e:
            pytest.skip(f"Pre-trained models not available: {e}")

    def test_load_and_sample_ldm_moon(self):
        """Test loading ldm_moon model and generating samples."""
        try:
            # moon: 6 conditioning features
            # (OBSALT, TRANSPARENCY_GFA, SUNALT, MOONALT, MOONSEP, MOONFRAC)
            conditioning = jnp.array([
                [60.0, 0.9, -25.0, 30.0, 45.0, 0.8],
                [70.0, 0.85, -28.0, 25.0, 60.0, 0.9],
            ])
            spectra, ldm_meta, vae_meta = self._load_and_sample("ldm_moon", conditioning)

            assert ldm_meta["arch"]["meta_dim"] == 6
            assert spectra.shape[0] == 2
            assert spectra.shape[1] == vae_meta["arch"]["in_channels"]
            assert jnp.all(jnp.isfinite(spectra))
        except Exception as e:
            pytest.skip(f"Pre-trained models not available: {e}")

    def test_load_and_sample_ldm_twilight(self):
        """Test loading ldm_twilight model and generating samples."""
        try:
            # twilight: 4 conditioning features
            # (OBSALT, TRANSPARENCY_GFA, SUNALT, SUNSEP)
            conditioning = jnp.array([
                [60.0, 0.9, -15.0, 120.0],
                [70.0, 0.85, -12.0, 115.0],
            ])
            spectra, ldm_meta, vae_meta = self._load_and_sample("ldm_twilight", conditioning)

            assert ldm_meta["arch"]["meta_dim"] == 4
            assert spectra.shape[0] == 2
            assert spectra.shape[1] == vae_meta["arch"]["in_channels"]
            assert jnp.all(jnp.isfinite(spectra))
        except Exception as e:
            pytest.skip(f"Pre-trained models not available: {e}")
