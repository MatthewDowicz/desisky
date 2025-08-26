# SPDX-FileCopyrightText: 2025-present MatthewDowicz <mjdowicz@gmail.com>
#
# SPDX-License-Identifier: MIT

from .io.model_io import REGISTRY, load_or_builtin, get_user_model_dir  # re-export

def _ensure_registered(kind: str) -> None:
    """Lazy-register known models on first use to avoid heavy imports."""
    if kind == "broadband" and "broadband" not in REGISTRY:
        # Importing this submodule runs `register_model(...)` as a side-effect.
        from .models import broadband  # noqa: F401

def load_model(kind: str, path=None, constructor=None):
    """Convenience wrapper that ensures registration, then loads."""
    _ensure_registered(kind)
    return load_or_builtin(kind, path=path, constructor=constructor)