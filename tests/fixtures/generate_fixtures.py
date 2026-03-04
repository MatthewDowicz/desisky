#!/usr/bin/env python
"""Generate small synthetic fixture datasets for CLI tests.

Run this once to create the fixture files:
    python tests/fixtures/generate_fixtures.py

The fixtures are synthetic (random) data with realistic shapes and column names,
suitable for testing data loading paths, argument parsing, and pipeline wiring.
They are NOT suitable for testing model accuracy.
"""

import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
N = 50
N_WAVE = 7781
SEED = 42

rng = np.random.default_rng(SEED)


def generate_tiny_spectra():
    """Create tiny_spectra.npz: flux (N, 7781) + wavelength (7781,)."""
    wavelength = np.linspace(3600.0, 9824.0, N_WAVE, dtype=np.float32)
    flux = rng.normal(1.0, 0.1, (N, N_WAVE)).astype(np.float32)
    np.savez(HERE / "tiny_spectra.npz", flux=flux, wavelength=wavelength)
    print(f"  tiny_spectra.npz: flux {flux.shape}, wavelength {wavelength.shape}")


def generate_tiny_broadband():
    """Create tiny_broadband.csv: broadband input features + target magnitudes."""
    df = pd.DataFrame({
        "EXPID": np.arange(N),
        "MOONSEP": rng.uniform(10, 90, N),
        "MOONFRAC": rng.uniform(0.5, 1.0, N),
        "MOONALT": rng.uniform(5, 60, N),
        "OBSALT": rng.uniform(30, 80, N),
        "TRANSPARENCY_GFA": rng.uniform(0.5, 1.0, N),
        "ECLIPSE_FRAC": rng.uniform(0, 0.3, N),
        "V": rng.uniform(18, 21, N),
        "g": rng.uniform(18, 21, N),
        "r": rng.uniform(18, 21, N),
        "z": rng.uniform(18, 21, N),
    })
    df.to_csv(HERE / "tiny_broadband.csv", index=False)
    print(f"  tiny_broadband.csv: {len(df)} rows")


def generate_tiny_conditioning():
    """Create tiny_conditioning.npz: conditioning (N, 8) for dark variant."""
    conditioning = rng.normal(0, 1, (N, 8)).astype(np.float32)
    wavelength = np.linspace(3600.0, 9824.0, N_WAVE, dtype=np.float32)
    flux = rng.normal(1.0, 0.1, (N, N_WAVE)).astype(np.float32)
    np.savez(
        HERE / "tiny_conditioning.npz",
        conditioning=conditioning,
        wavelength=wavelength,
        flux=flux,
    )
    print(f"  tiny_conditioning.npz: conditioning {conditioning.shape}, flux {flux.shape}")


if __name__ == "__main__":
    print("Generating test fixtures...")
    generate_tiny_spectra()
    generate_tiny_broadband()
    generate_tiny_conditioning()
    print("Done.")
