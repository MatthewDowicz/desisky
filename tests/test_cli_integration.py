# SPDX-FileCopyrightText: 2025-present MatthewDowicz <mjdowicz@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Integration tests for CLI scripts.

These tests run the actual training/inference pipelines with small fixture
datasets and minimal epochs. Marked @pytest.mark.slow since they involve
real JAX computation.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


# ---------- Training integration tests ----------


@pytest.mark.slow
class TestTrainBroadbandIntegration:
    def test_train_with_csv_data(self, tmp_path):
        """Train broadband for 2 epochs on fixture CSV data."""
        result = subprocess.run(
            [
                sys.executable, "-m", "desisky.scripts.train_broadband",
                "--data-path", str(FIXTURES / "tiny_broadband.csv"),
                "--epochs", "2",
                "--batch-size", "16",
                "--no-save",
                "--print-every", "1",
            ],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, f"STDERR:\n{result.stderr}"
        assert "Best test loss" in result.stdout


@pytest.mark.slow
class TestTrainVAEIntegration:
    def test_train_with_npz_data(self, tmp_path):
        """Train VAE for 2 epochs on fixture spectra."""
        result = subprocess.run(
            [
                sys.executable, "-m", "desisky.scripts.train_vae",
                "--data-path", str(FIXTURES / "tiny_spectra.npz"),
                "--epochs", "2",
                "--batch-size", "16",
                "--no-save",
                "--print-every", "1",
                "--latent-dim", "4",
            ],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, f"STDERR:\n{result.stderr}"
        assert "Best test loss" in result.stdout


@pytest.mark.slow
class TestTrainLDMIntegration:
    @pytest.mark.parametrize("variant,meta_dim", [
        ("dark", 8),
        ("moon", 6),
        ("twilight", 4),
    ])
    def test_train_variant(self, tmp_path, variant, meta_dim):
        """Train LDM for 2 epochs on fixture data (all variants)."""
        # Create variant-specific fixture with correct meta_dim
        rng = np.random.default_rng(42)
        flux = rng.normal(1.0, 0.1, (50, 7781)).astype(np.float32)
        conditioning = rng.normal(0, 1, (50, meta_dim)).astype(np.float32)
        wavelength = np.linspace(3600, 9824, 7781, dtype=np.float32)
        fixture_path = tmp_path / f"tiny_{variant}.npz"
        np.savez(fixture_path, flux=flux, conditioning=conditioning, wavelength=wavelength)

        result = subprocess.run(
            [
                sys.executable, "-m", "desisky.scripts.train_ldm",
                "--variant", variant,
                "--data-path", str(fixture_path),
                "--epochs", "2",
                "--batch-size", "16",
                "--no-save",
                "--print-every", "1",
                "--hidden", "8",
                "--levels", "2",
                "--emb-dim", "8",
            ],
            capture_output=True, text=True, timeout=300,
        )
        assert result.returncode == 0, f"STDERR:\n{result.stderr}"
        assert "Best val loss" in result.stdout


# ---------- Inference integration tests ----------


@pytest.mark.slow
class TestInferBroadbandIntegration:
    def test_infer_csv_output(self, tmp_path):
        """Run broadband inference and check CSV output."""
        out_path = tmp_path / "preds.csv"
        result = subprocess.run(
            [
                sys.executable, "-m", "desisky.scripts.infer_broadband",
                "--data-path", str(FIXTURES / "tiny_broadband.csv"),
                "--output", str(out_path),
                "--output-format", "csv",
                "--n-samples", "10",
            ],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, f"STDERR:\n{result.stderr}"
        assert out_path.exists()
        df = pd.read_csv(out_path)
        assert len(df) == 10
        assert "V_pred" in df.columns

    def test_infer_npz_output(self, tmp_path):
        """Run broadband inference and check npz output."""
        out_path = tmp_path / "preds.npz"
        result = subprocess.run(
            [
                sys.executable, "-m", "desisky.scripts.infer_broadband",
                "--data-path", str(FIXTURES / "tiny_broadband.csv"),
                "--output", str(out_path),
                "--output-format", "npz",
                "--n-samples", "5",
            ],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, f"STDERR:\n{result.stderr}"
        assert out_path.exists()
        data = np.load(out_path)
        assert "predictions" in data
        assert data["predictions"].shape == (5, 4)


@pytest.mark.slow
class TestInferVAEIntegration:
    def test_infer_npz_output(self, tmp_path):
        """Run VAE inference and check output keys."""
        out_path = tmp_path / "vae_out.npz"
        result = subprocess.run(
            [
                sys.executable, "-m", "desisky.scripts.infer_vae",
                "--data-path", str(FIXTURES / "tiny_spectra.npz"),
                "--output", str(out_path),
                "--n-samples", "5",
            ],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, f"STDERR:\n{result.stderr}"
        assert out_path.exists()
        data = np.load(out_path)
        assert "latents" in data
        assert "reconstructed" in data
        assert "recon_error" in data
        assert data["latents"].shape[0] == 5


@pytest.mark.slow
class TestInferLDMIntegration:
    def test_infer_with_json_conditioning(self, tmp_path):
        """Run LDM inference with inline JSON conditioning."""
        out_path = tmp_path / "ldm_out.npz"
        cond = "[[1,2,3,4,5,6,7,8]]"
        result = subprocess.run(
            [
                sys.executable, "-m", "desisky.scripts.infer_ldm",
                "--variant", "dark",
                "--conditioning", cond,
                "--n-samples", "2",
                "--num-steps", "5",
                "--output", str(out_path),
            ],
            capture_output=True, text=True, timeout=300,
        )
        assert result.returncode == 0, f"STDERR:\n{result.stderr}"
        assert out_path.exists()
        data = np.load(out_path)
        assert "spectra" in data
        assert data["spectra"].shape[0] == 2
