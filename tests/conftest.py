import os
import pytest
from desisky.io.model_io import REGISTRY

# Force JAX to use CPU for all tests to avoid CUDA out of memory errors
# This must be set BEFORE importing JAX
os.environ['JAX_PLATFORM_NAME'] = 'cpu'


@pytest.fixture(scope="session", autouse=True)
def _ensure_vac_downloaded():
    """
    Ensure VAC file is downloaded once at the start of the test session.

    This allows tests to use download=False while still working in CI.
    The file is downloaded once and reused by all tests in the session.

    Downloads are skipped if:
    - Data dependencies (fitsio, pandas) are not installed
    - File already exists (download_file checks and skips)
    """
    try:
        from desisky.data import SkySpecVAC
        # Download if not present (no-op if already exists)
        vac = SkySpecVAC(version="v1.0", download=True)
        # Verify it exists
        assert vac.path.exists(), f"VAC file not found at {vac.path}"
    except ImportError:
        # Skip if data dependencies not installed (tests will be skipped too)
        pass
    yield


@pytest.fixture(scope="session", autouse=True)
def _clear_registry_at_end():
    """
    Clear the model registry at the end of the test session.

    We use session scope so that models registered during testing (like broadband)
    stay registered throughout the entire test suite. This matches real-world usage
    where models are registered once on import and remain registered.

    Individual tests that need to manipulate the registry (like testing registration
    logic) should save/restore the registry state themselves.
    """
    yield
    REGISTRY.clear()