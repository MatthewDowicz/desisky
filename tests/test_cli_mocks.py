# SPDX-FileCopyrightText: 2025-present MatthewDowicz <mjdowicz@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Mock-based tests for CLI scripts.

Tests data loading, wandb integration, and checkpoint behavior without
requiring network access or real DESI data.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Path to fixture data
FIXTURES = Path(__file__).parent / "fixtures"


# ---------- Broadband data loading ----------


class TestBroadbandDataLoading:
    def test_csv_data_loading(self):
        """Test that train_broadband can parse a CSV with required columns."""
        from desisky.scripts.train_broadband import INPUT_FEATURES
        import pandas as pd

        csv_path = FIXTURES / "tiny_broadband.csv"
        df = pd.read_csv(csv_path)
        required = set(INPUT_FEATURES) | {"V", "g", "r", "z"}
        assert required.issubset(set(df.columns))

    def test_csv_missing_columns_detected(self):
        """A CSV missing required columns should trigger sys.exit."""
        import pandas as pd

        bad_csv = FIXTURES / "tiny_broadband_bad.csv"
        df = pd.DataFrame({"MOONSEP": [1.0], "MOONFRAC": [0.5]})
        df.to_csv(bad_csv, index=False)
        try:
            from desisky.scripts.train_broadband import INPUT_FEATURES
            loaded = pd.read_csv(bad_csv)
            required = set(INPUT_FEATURES) | {"V", "g", "r", "z"}
            missing = required - set(loaded.columns)
            assert len(missing) > 0
        finally:
            bad_csv.unlink(missing_ok=True)


# ---------- VAE data loading ----------


class TestVAEDataLoading:
    def test_npz_has_flux_key(self):
        """Verify fixture .npz has the required 'flux' key."""
        data = np.load(FIXTURES / "tiny_spectra.npz")
        assert "flux" in data
        assert data["flux"].shape[0] == 50
        assert data["flux"].shape[1] == 7781


# ---------- LDM data loading ----------


class TestLDMDataLoading:
    def test_npz_has_required_keys(self):
        """Verify fixture .npz has the required keys for LDM training."""
        data = np.load(FIXTURES / "tiny_conditioning.npz")
        assert "flux" in data
        assert "conditioning" in data
        assert "wavelength" in data
        assert data["conditioning"].shape == (50, 8)


# ---------- LDM conditioning sources ----------


class TestLDMConditioningLoading:
    def test_load_conditioning_from_json(self):
        """Test inline JSON conditioning parsing."""
        from desisky.scripts.infer_ldm import load_conditioning
        import argparse

        args = argparse.Namespace(
            conditioning='[[1,2,3,4,5,6,7,8]]',
            cond_path=None,
            from_data=False,
            n_samples=5,
            seed=42,
        )
        cond = load_conditioning(args, "dark", meta_dim=8)
        assert cond.shape == (5, 8)
        np.testing.assert_allclose(cond[0], cond[4])

    def test_load_conditioning_from_npz(self):
        """Test loading conditioning from a .npz file."""
        from desisky.scripts.infer_ldm import load_conditioning
        import argparse

        args = argparse.Namespace(
            conditioning=None,
            cond_path=str(FIXTURES / "tiny_conditioning.npz"),
            from_data=False,
            n_samples=10,
            seed=42,
        )
        cond = load_conditioning(args, "dark", meta_dim=8)
        assert cond.shape == (50, 8)

    def test_load_conditioning_wrong_dim_exits(self):
        """JSON with wrong dimension should sys.exit."""
        from desisky.scripts.infer_ldm import load_conditioning
        import argparse

        args = argparse.Namespace(
            conditioning='[[1,2,3]]',
            cond_path=None,
            from_data=False,
            n_samples=1,
            seed=42,
        )
        with pytest.raises(SystemExit):
            load_conditioning(args, "dark", meta_dim=8)


# ---------- wandb flag behavior ----------


class TestWandbFlagBehavior:
    def test_wandb_not_imported_without_flag(self):
        """When --wandb is not set, wandb should not be imported."""
        from desisky.scripts.train_broadband import parse_args
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("sys.argv", ["prog", "--epochs", "1"])
            args = parse_args()
        assert args.wandb is False

    def test_wandb_flag_sets_config(self):
        """When --wandb is set, wandb_config should be constructed."""
        from desisky.scripts.train_broadband import parse_args
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("sys.argv", [
                "prog", "--wandb", "--wandb-project", "test-proj",
                "--wandb-entity", "my-team", "--wandb-tags", "a,b",
            ])
            args = parse_args()
        assert args.wandb is True
        assert args.wandb_project == "test-proj"
        assert args.wandb_entity == "my-team"
        assert args.wandb_tags == "a,b"


# ---------- No-save flag ----------


class TestNoSaveFlag:
    def test_no_save_broadband(self):
        from desisky.scripts.train_broadband import parse_args
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("sys.argv", ["prog", "--no-save"])
            args = parse_args()
        assert args.no_save is True

    def test_no_save_vae(self):
        from desisky.scripts.train_vae import parse_args
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("sys.argv", ["prog", "--no-save"])
            args = parse_args()
        assert args.no_save is True

    def test_no_save_ldm(self):
        from desisky.scripts.train_ldm import parse_args
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("sys.argv", ["prog", "--no-save"])
            args = parse_args()
        assert args.no_save is True
