import pytest
from desisky.io.model_io import REGISTRY

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