import pytest
from desisky.io.model_io import REGISTRY

@pytest.fixture(autouse=True)
def _clear_registry():
    REGISTRY.clear()
    yield
    REGISTRY.clear()