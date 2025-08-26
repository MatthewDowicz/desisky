from pathlib import Path
from typing import Any, Callable, Optional, Tuple

from .model_io import (
    REGISTRY, ModelSpec, get_user_model_dir, save, load,
    load_builtin, load_or_builtin, register_model,
)

def _ensure_registered(kind: str) -> None:
    """Lazy-register known models on first use to avoid import cycles."""
    if kind in REGISTRY:
        return
    if kind == "broadband":
        # Importing this module should call register_model(...) as a side effect.
        from ..models import broadband  # noqa: F401
    else:
        # Optional: convention-based best effort, e.g. desisky.models.<kind>
        try:
            __import__(f"{__package__[:-3]}.models.{kind}")  # ".." back to package
        except Exception:
            # leave it to load_or_builtin to raise a nice KeyError
            pass

def load_model(
    kind: str,
    path: Optional[str | Path] = None,
    constructor: Optional[Callable[..., Any]] = None,
) -> Tuple[Any, dict]:
    """Load a user checkpoint (if `path` given) or the packaged/builtin weights for `kind`."""
    _ensure_registered(kind)
    return load_or_builtin(kind, path=path, constructor=constructor)

__all__ = [
    "ModelSpec",
    "REGISTRY",
    "get_user_model_dir",
    "save", "load", "load_builtin", "load_or_builtin",
    "register_model",
    "load_model",
]
