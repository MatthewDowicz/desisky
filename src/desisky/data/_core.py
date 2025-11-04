# SPDX-FileCopyrightText: 2025-present MatthewDowicz <mjdowicz@gmail.com>
#
# SPDX-License-Identifier: MIT

from __future__ import annotations
from pathlib import Path
import os
import hashlib
import tempfile
import shutil
import requests
from typing import Optional

ENV_VAR = "DESISKY_DATA_DIR"


def default_root() -> Path:
    """
    Return the default root directory for DESI sky data.

    Checks the environment variable ``DESISKY_DATA_DIR``. If set, that path is
    expanded (e.g., ``~`` -> home directory) and resolved (absolute path).
    Otherwise, falls back to the default: ``~/.desisky/data``.

    Returns
    -------
    Path
        Absolute path to the dataset root.
    """
    val = os.getenv(ENV_VAR)
    if val:
        # Use user-specified path, expanding `~` and resolving symlinks
        return Path(val).expanduser().resolve()
    return Path.home() / ".desisky" / "data"


def ensure_dir(p: Path) -> Path:
    """
    Ensure that a directory exists, creating it if necessary.

    Parameters
    ----------
    p : Path
        Directory path to create.

    Returns
    -------
    Path
        The same path, guaranteed to exist as a directory.
    """
    p.mkdir(parents=True, exist_ok=True)
    return p


def sha256sum(path: Path, chunk: int = 1024 * 1024) -> str:
    """
    Compute the SHA-256 checksum of a file.

    Parameters
    ----------
    path : Path
        Path to the file to hash.
    chunk : int, optional
        Number of bytes to read at a time. Defaults to 1 MB. Larger chunks
        reduce overhead, smaller chunks reduce memory usage.

    Returns
    -------
    str
        The hexadecimal SHA-256 digest of file contents.

    Notes
    -----
    Useful for verifying file integrity after download. If the file differs
    even by a single byte, the checksum will change.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def download_file(
    url: str,
    dest: Path,
    expected_sha256: str | None = None,
    force: bool = False,
    timeout: int = 60,
) -> Path:
    """
    Download a file to ``dest``, optionally verifying its SHA-256 checksum.

    The download streams to a temporary file in the same directory as ``dest``
    and then renames it into place to avoid leaving partial files on failure.
    If ``dest`` already exists, it is reused unless ``force=True``, and when
    ``expected_sha256`` is provided its integrity is checked first.

    Parameters
    ----------
    url : str
        Source URL to download.
    dest : Path
        Destination file path (may include ``~``, which will be expanded).
    expected_sha256 : str, optional
        Expected hex-encoded SHA-256 digest. If provided, a mismatch raises
        ``ValueError``.
    force : bool, default False
        If True, re-download even if ``dest`` already exists.
    timeout : int, default 60
        Timeout in seconds for the HTTP request.

    Returns
    -------
    Path
        The path to the downloaded file (``dest``).

    Raises
    ------
    ValueError
        If the computed SHA-256 digest does not match ``expected_sha256``.
    requests.HTTPError
        If the HTTP request fails (via ``raise_for_status()``).
    """
    # Normalize the destination (expand `~`) and ensure its parent exists.
    dest = dest.expanduser()
    ensure_dir(dest.parent)

    # Reuse existing file when allowed. If a checksum is provided, verify it.
    if dest.exists() and not force:
        if expected_sha256:
            if sha256sum(dest).lower() == expected_sha256.lower():
                return dest
        else:
            return dest

    # Stream response to temp file (in same directory as `dest`)
    # so final move is atomic on the same filesystem.
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, dir=str(dest.parent)) as tmp:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    tmp.write(chunk)
            tmp_path = Path(tmp.name)

    # If an expected checksum was provided, verify the downloaded bytes.
    if expected_sha256:
        got = sha256sum(tmp_path)
        if got.lower() != expected_sha256.lower():
            tmp_path.unlink(missing_ok=True)
            raise ValueError(
                f"SHA256 mismatch for {dest.name}:\n"
                f"  expected {expected_sha256}\n"
                f"  got      {got}"
            )

    shutil.move(str(tmp_path), str(dest))
    return dest
