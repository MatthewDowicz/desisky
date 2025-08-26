# SPDX-FileCopyrightText: 2025-present MatthewDowicz <mjdowicz@gmail.com>
#
# SPDX-License-Identifier: MIT

# SPDX…

import importlib

__all__ = ["io"]

def __getattr__(name):
    if name == "io":
        mod = importlib.import_module(".io", __name__)
        globals()["io"] = mod  # cache for future lookups
        return mod
    raise AttributeError(name)
