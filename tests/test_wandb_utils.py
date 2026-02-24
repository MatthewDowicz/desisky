# SPDX-FileCopyrightText: 2025-present MatthewDowicz <mjdowicz@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Tests for wandb integration utilities.

All tests mock the wandb module so they run without a network connection
or wandb account.  The ``@patch`` decorator temporarily replaces the real
``wandb`` module (or specific attributes) with a ``MagicMock`` object.
When the test finishes, the original module is restored.  This lets us
verify that our code calls wandb in the right way without side effects.
"""

import pytest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass

from desisky.training.wandb_utils import (
    WandbConfig,
    init_wandb_run,
    log_epoch_metrics,
    log_figure,
    finish_wandb_run,
)


# ---------- Fixtures ----------


@dataclass
class _DummyTrainingConfig:
    """Minimal dataclass to stand in for a training config."""
    epochs: int = 10
    learning_rate: float = 1e-4


@pytest.fixture
def wconfig():
    return WandbConfig(project="test-project", run_name="test-run", tags=["ci"])


@pytest.fixture
def train_config():
    return _DummyTrainingConfig()


# ---------- init_wandb_run ----------


class TestInitWandbRun:
    """Test sweep-awareness and config merging in init_wandb_run."""

    @patch("desisky.training.wandb_utils.wandb")
    def test_sweep_reuses_existing_run(self, mock_wandb, wconfig, train_config):
        """When wandb.run is already active (sweep), init should NOT call wandb.init."""
        existing_run = MagicMock()
        mock_wandb.run = existing_run

        result = init_wandb_run(wconfig, train_config)

        assert result is existing_run
        mock_wandb.init.assert_not_called()

    @patch("desisky.training.wandb_utils.wandb")
    def test_creates_new_run_when_none_active(self, mock_wandb, wconfig, train_config):
        """When no run is active, init should call wandb.init with correct args."""
        mock_wandb.run = None
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run

        result = init_wandb_run(wconfig, train_config)

        assert result is mock_run
        mock_wandb.init.assert_called_once()
        call_kwargs = mock_wandb.init.call_args[1]
        assert call_kwargs["project"] == "test-project"
        assert call_kwargs["name"] == "test-run"
        assert call_kwargs["tags"] == ["ci"]
        assert "training" in call_kwargs["config"]

    @patch("desisky.training.wandb_utils.wandb")
    def test_model_config_merged(self, mock_wandb, wconfig, train_config):
        """Model architecture dict should appear under config['model']."""
        mock_wandb.run = None
        mock_wandb.init.return_value = MagicMock()
        model_config = {"hidden": 64, "levels": 3}

        init_wandb_run(wconfig, train_config, model_config=model_config)

        config_arg = mock_wandb.init.call_args[1]["config"]
        assert config_arg["model"] == model_config


# ---------- log_epoch_metrics ----------


class TestLogEpochMetrics:
    """Test that metrics are logged with correct prefix and step."""

    @patch("desisky.training.wandb_utils.wandb")
    def test_metrics_prefixed_and_stepped(self, mock_wandb):
        """Metric keys should be prefixed and epoch used as step."""
        mock_wandb.run = MagicMock()  # active run

        log_epoch_metrics({"loss": 0.5, "recon": 0.3}, epoch=10, prefix="train/")

        mock_wandb.log.assert_called_once()
        payload = mock_wandb.log.call_args[0][0]
        assert payload["train/loss"] == 0.5
        assert payload["train/recon"] == 0.3
        assert payload["epoch"] == 10
        assert mock_wandb.log.call_args[1]["step"] == 10

    @patch("desisky.training.wandb_utils.wandb")
    def test_no_op_when_no_active_run(self, mock_wandb):
        """Should silently return when wandb.run is None."""
        mock_wandb.run = None

        log_epoch_metrics({"loss": 0.5}, epoch=1)

        mock_wandb.log.assert_not_called()


# ---------- log_figure ----------


class TestLogFigure:
    """Test figure logging."""

    @patch("desisky.training.wandb_utils.wandb")
    def test_figure_logged_as_image(self, mock_wandb):
        """Figure should be wrapped in wandb.Image and logged."""
        mock_wandb.run = MagicMock()
        fake_fig = MagicMock()

        log_figure("val/cdf", fake_fig, epoch=5)

        mock_wandb.Image.assert_called_once_with(fake_fig)
        mock_wandb.log.assert_called_once()
        assert mock_wandb.log.call_args[1]["step"] == 5


# ---------- finish_wandb_run ----------


class TestFinishWandbRun:
    """Test safe cleanup."""

    @patch("desisky.training.wandb_utils.wandb")
    def test_finish_calls_wandb_finish(self, mock_wandb):
        mock_wandb.run = MagicMock()

        finish_wandb_run()

        mock_wandb.finish.assert_called_once()

    @patch("desisky.training.wandb_utils.wandb")
    def test_finish_safe_when_no_run(self, mock_wandb):
        mock_wandb.run = None

        finish_wandb_run()  # should not raise

        mock_wandb.finish.assert_not_called()


# ---------- Trainer new-param acceptance ----------


class TestTrainerParamAcceptance:
    """Verify trainers accept wandb_config and on_epoch_end without breaking."""

    def test_vae_trainer_accepts_new_params(self):
        """VAETrainer.__init__ should accept wandb_config and on_epoch_end."""
        import jax.random as jr
        from desisky.models.vae import make_SkyVAE
        from desisky.training import VAETrainer, VAETrainingConfig

        model = make_SkyVAE(in_channels=100, latent_dim=4, key=jr.PRNGKey(0))
        config = VAETrainingConfig(epochs=1, learning_rate=1e-3, save_best=False)

        # Should not raise
        trainer = VAETrainer(
            model, config,
            wandb_config=WandbConfig(project="test"),
            on_epoch_end=lambda model, history, epoch: None,
        )
        assert trainer.wandb_config is not None
        assert trainer.on_epoch_end is not None

    def test_ldm_trainer_accepts_new_params(self):
        """LatentDiffusionTrainer.__init__ should accept wandb_config and on_epoch_end."""
        import jax.random as jr
        from desisky.models.ldm import make_UNet1D_cond
        from desisky.training import LatentDiffusionTrainer, LDMTrainingConfig

        model = make_UNet1D_cond(
            in_ch=1, out_ch=1, meta_dim=4, hidden=16,
            levels=2, emb_dim=16, key=jr.PRNGKey(0),
        )
        config = LDMTrainingConfig(
            epochs=1, learning_rate=1e-3, meta_dim=4,
            sigma_data=1.0, save_best=False,
        )

        trainer = LatentDiffusionTrainer(
            model, config,
            wandb_config=WandbConfig(project="test"),
            on_epoch_end=lambda m, ema, h, e: None,
        )
        assert trainer.wandb_config is not None
        assert trainer.on_epoch_end is not None


# ---------- New LDM config/history fields ----------


class TestLDMNewFields:
    """Verify new fields added for EMA early stopping."""

    def test_early_stop_on_ema_defaults_true(self):
        from desisky.training import LDMTrainingConfig

        config = LDMTrainingConfig(
            epochs=10, learning_rate=1e-4, meta_dim=4, sigma_data=1.0,
        )
        assert config.early_stop_on_ema is True

    def test_ema_val_losses_in_history(self):
        from desisky.training import LDMTrainingHistory

        history = LDMTrainingHistory()
        assert hasattr(history, "ema_val_losses")
        assert history.ema_val_losses == []
