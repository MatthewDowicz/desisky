# SPDX-FileCopyrightText: 2025-present MatthewDowicz <mjdowicz@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Unit tests for CLI script argument parsing, constants, and helpers."""

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest


# ---------- --help smoke tests ----------


SCRIPTS = [
    "desisky.scripts.train_broadband",
    "desisky.scripts.train_vae",
    "desisky.scripts.train_ldm",
    "desisky.scripts.infer_broadband",
    "desisky.scripts.infer_vae",
    "desisky.scripts.infer_ldm",
]


@pytest.mark.parametrize("module", SCRIPTS)
def test_help_exits_zero(module):
    """Each CLI script's --help should exit with code 0."""
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr


# ---------- Argument parsing defaults ----------


class TestTrainBroadbandArgs:
    def test_defaults(self):
        from desisky.scripts.train_broadband import parse_args
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("sys.argv", ["prog"])
            args = parse_args()
        assert args.epochs == 500
        assert args.learning_rate == 1e-4
        assert args.batch_size == 64
        assert args.loss == "huber"
        assert args.huber_delta == 0.25
        assert args.val_split == 0.3
        assert args.width == 128
        assert args.depth == 5
        assert args.run_name == "broadband_moon"
        assert args.wandb is False
        assert args.seed == 42

    def test_custom_args(self):
        from desisky.scripts.train_broadband import parse_args
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("sys.argv", [
                "prog", "--epochs", "10", "--loss", "l2",
                "--width", "64", "--depth", "3", "--wandb",
            ])
            args = parse_args()
        assert args.epochs == 10
        assert args.loss == "l2"
        assert args.width == 64
        assert args.depth == 3
        assert args.wandb is True


class TestTrainVAEArgs:
    def test_defaults(self):
        from desisky.scripts.train_vae import parse_args
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("sys.argv", ["prog"])
            args = parse_args()
        assert args.epochs == 100
        assert args.latent_dim == 8
        assert args.beta == 1e-3
        assert args.lam == 4.0
        assert args.kernel_sigma == "auto"
        assert args.val_split == 0.1
        assert args.run_name == "sky_vae"

    def test_custom_args(self):
        from desisky.scripts.train_vae import parse_args
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("sys.argv", [
                "prog", "--epochs", "5", "--latent-dim", "4",
                "--beta", "0.01", "--clip-gradients",
            ])
            args = parse_args()
        assert args.epochs == 5
        assert args.latent_dim == 4
        assert args.beta == 0.01
        assert args.clip_gradients is True


class TestTrainLDMArgs:
    def test_defaults(self):
        from desisky.scripts.train_ldm import parse_args
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("sys.argv", ["prog"])
            args = parse_args()
        assert args.variant == "dark"
        assert args.epochs == 200
        assert args.hidden == 32
        assert args.levels == 3
        assert args.emb_dim == 32
        assert args.dropout_p == 0.1
        assert args.ema_decay == 0.9999

    def test_variant_choices(self):
        from desisky.scripts.train_ldm import parse_args
        for variant in ["dark", "moon", "twilight"]:
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr("sys.argv", ["prog", "--variant", variant])
                args = parse_args()
            assert args.variant == variant

    def test_invalid_variant_exits(self):
        from desisky.scripts.train_ldm import parse_args
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("sys.argv", ["prog", "--variant", "invalid"])
            with pytest.raises(SystemExit):
                parse_args()


class TestInferBroadbandArgs:
    def test_defaults(self):
        from desisky.scripts.infer_broadband import parse_args
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("sys.argv", ["prog"])
            args = parse_args()
        assert args.output == "broadband_predictions.csv"
        assert args.output_format == "csv"
        assert args.n_samples is None


class TestInferVAEArgs:
    def test_defaults(self):
        from desisky.scripts.infer_vae import parse_args
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("sys.argv", ["prog"])
            args = parse_args()
        assert args.subset == "full"
        assert args.output == "vae_output.npz"
        assert args.seed == 42

    def test_subset_choices(self):
        from desisky.scripts.infer_vae import parse_args
        for subset in ["full", "dark", "moon", "twilight"]:
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr("sys.argv", ["prog", "--subset", subset])
                args = parse_args()
            assert args.subset == subset


class TestInferLDMArgs:
    def test_defaults(self):
        from desisky.scripts.infer_ldm import parse_args
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("sys.argv", ["prog"])
            args = parse_args()
        assert args.variant == "dark"
        assert args.n_samples == 100
        assert args.guidance_scale == 1.0
        assert args.num_steps == 250
        assert args.output == "ldm_generated.npz"


# ---------- Constants ----------


class TestConditioningCols:
    def test_train_ldm_conditioning_cols(self):
        from desisky.scripts.train_ldm import CONDITIONING_COLS
        assert set(CONDITIONING_COLS.keys()) == {"dark", "moon", "twilight"}
        assert len(CONDITIONING_COLS["dark"]) == 8
        assert len(CONDITIONING_COLS["moon"]) == 6
        assert len(CONDITIONING_COLS["twilight"]) == 4

    def test_infer_ldm_conditioning_cols(self):
        from desisky.scripts.infer_ldm import CONDITIONING_COLS
        assert set(CONDITIONING_COLS.keys()) == {"dark", "moon", "twilight"}
        assert len(CONDITIONING_COLS["dark"]) == 8
        assert len(CONDITIONING_COLS["moon"]) == 6
        assert len(CONDITIONING_COLS["twilight"]) == 4

    def test_train_and_infer_cols_match(self):
        """Train and infer scripts must use identical conditioning columns."""
        from desisky.scripts.train_ldm import CONDITIONING_COLS as TRAIN_COLS
        from desisky.scripts.infer_ldm import CONDITIONING_COLS as INFER_COLS
        assert TRAIN_COLS == INFER_COLS

    def test_broadband_input_features(self):
        from desisky.scripts.train_broadband import INPUT_FEATURES, IN_SIZE, OUT_SIZE
        assert len(INPUT_FEATURES) == IN_SIZE == 6
        assert OUT_SIZE == 4

    def test_broadband_infer_features_match(self):
        from desisky.scripts.train_broadband import INPUT_FEATURES as TRAIN_FEATS
        from desisky.scripts.infer_broadband import INPUT_FEATURES as INFER_FEATS
        assert TRAIN_FEATS == INFER_FEATS


# ---------- Helpers ----------


class TestClassifySkyCondition:
    def test_dark(self):
        from desisky.scripts.train_vae import classify_sky_condition
        meta = pd.DataFrame({
            "SUNALT": [-25.0], "MOONALT": [-10.0], "MOONFRAC": [0.0], "MOONSEP": [180.0],
        })
        result = classify_sky_condition(meta)
        assert result[0] == "dark"

    def test_moon(self):
        from desisky.scripts.train_vae import classify_sky_condition
        meta = pd.DataFrame({
            "SUNALT": [-25.0], "MOONALT": [10.0], "MOONFRAC": [0.8], "MOONSEP": [45.0],
        })
        result = classify_sky_condition(meta)
        assert result[0] == "moon"

    def test_twilight(self):
        from desisky.scripts.train_vae import classify_sky_condition
        meta = pd.DataFrame({
            "SUNALT": [-10.0], "MOONALT": [-10.0], "MOONFRAC": [0.0], "MOONSEP": [180.0],
        })
        result = classify_sky_condition(meta)
        assert result[0] == "twilight"

    def test_other(self):
        from desisky.scripts.train_vae import classify_sky_condition
        meta = pd.DataFrame({
            "SUNALT": [-10.0], "MOONALT": [30.0], "MOONFRAC": [0.8], "MOONSEP": [45.0],
        })
        result = classify_sky_condition(meta)
        assert result[0] == "other"
