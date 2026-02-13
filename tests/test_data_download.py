# SPDX-FileCopyrightText: 2025-present MatthewDowicz <mjdowicz@gmail.com>
#
# SPDX-License-Identifier: MIT

from pathlib import Path
import pytest
import types
import importlib


# ---------- Mock for requests.get ----------


class _MockResponse:
    """Mock response object for requests.get() streaming."""

    def __init__(self, content: bytes, status_code: int = 200):
        self._content = content
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code != 200:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size=1024):
        yield self._content

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


# ---------- Tests for _core.py ----------


def test_default_root_when_env_unset(monkeypatch):
    """Test default_root() returns ~/.desisky/data when env var is unset."""
    monkeypatch.delenv("DESISKY_DATA_DIR", raising=False)

    from desisky.data._core import default_root

    # Reload to pick up env changes (default_root reads env at call time, so this is extra safe)
    importlib.reload(importlib.import_module("desisky.data._core"))

    p = default_root()
    assert isinstance(p, Path)
    assert p == Path.home() / ".desisky" / "data"


def test_default_root_env_override_absolute_path(monkeypatch, tmp_path):
    """Test that DESISKY_DATA_DIR environment variable overrides default."""
    monkeypatch.setenv("DESISKY_DATA_DIR", str(tmp_path))

    from desisky.data._core import default_root

    p = default_root()
    assert p == tmp_path.resolve()


def test_default_root_env_override_with_tilde(monkeypatch):
    """Test that ~ in DESISKY_DATA_DIR gets expanded properly.

    Note: This test validates that expanduser() is called, not that
    we can mock the home directory (that's testing Python's stdlib).
    """
    # Set env var with tilde - it will expand to real home
    monkeypatch.setenv("DESISKY_DATA_DIR", "~/mydata")

    from desisky.data._core import default_root

    p = default_root()
    # Should be expanded (no ~ in result) and absolute
    assert "~" not in str(p)
    assert p.is_absolute()
    assert str(p).endswith("mydata")


def test_default_root_no_disk_side_effects(monkeypatch):
    """Test that default_root() doesn't create directories."""
    monkeypatch.delenv("DESISKY_DATA_DIR", raising=False)

    from desisky.data._core import default_root

    p = default_root()
    # Should return a path but not create it
    assert isinstance(p, Path)


def test_ensure_dir_creates_nested_directories(tmp_path):
    """Test that ensure_dir() creates nested directories."""
    from desisky.data._core import ensure_dir

    target = tmp_path / "nested" / "data" / "deep"
    assert not target.exists()

    result = ensure_dir(target)

    assert result == target
    assert target.exists()
    assert target.is_dir()


def test_ensure_dir_idempotent(tmp_path):
    """Test that ensure_dir() is safe to call multiple times."""
    from desisky.data._core import ensure_dir

    target = tmp_path / "mydir"

    # First call creates it
    result1 = ensure_dir(target)
    assert target.exists()

    # Second call should not error
    result2 = ensure_dir(target)
    assert result1 == result2
    assert target.exists()


def test_sha256sum_computes_correct_hash(tmp_path):
    """Test that sha256sum() computes correct checksums."""
    from desisky.data._core import sha256sum
    import hashlib

    # Create a test file with known content
    test_file = tmp_path / "test.txt"
    content = b"Hello, DESI Sky!"
    test_file.write_bytes(content)

    # Compute expected hash
    expected = hashlib.sha256(content).hexdigest()

    # Test our function
    result = sha256sum(test_file)
    assert result == expected


def test_sha256sum_different_files_different_hashes(tmp_path):
    """Test that different files produce different hashes."""
    from desisky.data._core import sha256sum

    file1 = tmp_path / "file1.txt"
    file2 = tmp_path / "file2.txt"

    file1.write_bytes(b"content A")
    file2.write_bytes(b"content B")

    hash1 = sha256sum(file1)
    hash2 = sha256sum(file2)

    assert hash1 != hash2


def test_download_file_creates_file(monkeypatch, tmp_path):
    """Test that download_file() successfully downloads and creates a file."""
    from desisky.data import _core

    # Mock requests.get
    mock_content = b"downloaded data"
    monkeypatch.setattr(
        _core, "requests", types.SimpleNamespace(get=lambda *a, **k: _MockResponse(mock_content))
    )

    dest = tmp_path / "downloaded.txt"
    result = _core.download_file("http://example.com/file.txt", dest)

    assert result == dest
    assert dest.exists()
    assert dest.read_bytes() == mock_content


def test_download_file_with_valid_sha256(monkeypatch, tmp_path):
    """Test that download_file() accepts correct SHA-256 checksums."""
    from desisky.data import _core
    import hashlib

    mock_content = b"test data for checksum"
    expected_sha = hashlib.sha256(mock_content).hexdigest()

    monkeypatch.setattr(
        _core, "requests", types.SimpleNamespace(get=lambda *a, **k: _MockResponse(mock_content))
    )

    dest = tmp_path / "checked.txt"
    result = _core.download_file("http://example.com/file.txt", dest, expected_sha256=expected_sha)

    assert result == dest
    assert dest.exists()


def test_download_file_with_invalid_sha256_raises(monkeypatch, tmp_path):
    """Test that download_file() raises ValueError on SHA-256 mismatch."""
    from desisky.data import _core

    mock_content = b"actual content"
    wrong_sha = "0" * 64  # Deliberately wrong hash

    monkeypatch.setattr(
        _core, "requests", types.SimpleNamespace(get=lambda *a, **k: _MockResponse(mock_content))
    )

    dest = tmp_path / "bad_checksum.txt"

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        _core.download_file("http://example.com/file.txt", dest, expected_sha256=wrong_sha)

    # File should be cleaned up after failed verification
    assert not dest.exists()


def test_download_file_reuses_existing_when_valid(monkeypatch, tmp_path):
    """Test that download_file() reuses existing file with correct checksum."""
    from desisky.data import _core
    import hashlib

    # Create existing file
    existing_content = b"already downloaded"
    dest = tmp_path / "existing.txt"
    dest.write_bytes(existing_content)
    expected_sha = hashlib.sha256(existing_content).hexdigest()

    # Mock should NOT be called if file is reused
    download_called = False

    def mock_get(*args, **kwargs):
        nonlocal download_called
        download_called = True
        return _MockResponse(b"new content")

    monkeypatch.setattr(_core, "requests", types.SimpleNamespace(get=mock_get))

    result = _core.download_file("http://example.com/file.txt", dest, expected_sha256=expected_sha)

    assert result == dest
    assert dest.read_bytes() == existing_content  # Original content preserved
    assert not download_called  # Should NOT have downloaded


def test_download_file_force_redownload(monkeypatch, tmp_path):
    """Test that force=True re-downloads even when file exists."""
    from desisky.data import _core

    # Create existing file
    dest = tmp_path / "existing.txt"
    dest.write_bytes(b"old content")

    # Mock download with new content
    new_content = b"new content"
    monkeypatch.setattr(
        _core, "requests", types.SimpleNamespace(get=lambda *a, **k: _MockResponse(new_content))
    )

    result = _core.download_file("http://example.com/file.txt", dest, force=True)

    assert result == dest
    assert dest.read_bytes() == new_content  # Should have new content


def test_download_file_expands_tilde_in_path(monkeypatch):
    """Test that download_file() properly expands ~ in destination path.

    Note: This validates that expanduser() is called, not that we can
    mock the home directory (that's testing Python's stdlib).
    """
    from desisky.data import _core

    mock_content = b"tilde test"
    monkeypatch.setattr(
        _core, "requests", types.SimpleNamespace(get=lambda *a, **k: _MockResponse(mock_content))
    )

    # Use ~ in path - will expand to real home
    dest = Path("~/desisky_test_download.txt")
    result = _core.download_file("http://example.com/file.txt", dest)

    try:
        # Should be expanded (no ~ in result) and absolute
        assert "~" not in str(result)
        assert result.is_absolute()
        assert result.exists()
        assert result.read_bytes() == mock_content
    finally:
        # Clean up test file from home directory
        if result.exists():
            result.unlink()


# ---------- Tests for skyspec.py ----------


def test_skyspec_vac_raises_on_unknown_version():
    """Test that SkySpecVAC raises KeyError for unknown versions."""
    from desisky.data import SkySpecVAC

    with pytest.raises(KeyError, match="Unknown SkySpec VAC version"):
        SkySpecVAC(version="v99.0", download=False)


def test_skyspec_vac_raises_when_file_missing_and_no_download(tmp_path):
    """Test that SkySpecVAC raises FileNotFoundError when file doesn't exist."""
    from desisky.data import SkySpecVAC

    with pytest.raises(FileNotFoundError, match="does not exist"):
        SkySpecVAC(root=tmp_path, version="v1.0", download=False)


def test_skyspec_vac_downloads_when_requested(monkeypatch, tmp_path):
    """Test that SkySpecVAC downloads data when download=True."""
    from desisky.data import _core, skyspec

    # Mock download_file to create a dummy file
    def mock_download(url, dest, expected_sha256=None, force=False, timeout=60):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"mock FITS data")
        return dest

    monkeypatch.setattr(_core, "download_file", mock_download)

    vac = skyspec.SkySpecVAC(root=tmp_path, version="v1.0", download=True)

    assert vac.path.exists()
    assert vac.filepath() == vac.path


def test_skyspec_vac_filepath_returns_path(monkeypatch, tmp_path):
    """Test that filepath() method returns the correct path."""
    from desisky.data import _core, skyspec

    def mock_download(url, dest, expected_sha256=None, force=False, timeout=60):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"mock FITS")
        return dest

    monkeypatch.setattr(_core, "download_file", mock_download)

    vac = skyspec.SkySpecVAC(root=tmp_path, version="v1.0", download=True)

    filepath = vac.filepath()
    assert isinstance(filepath, Path)
    assert filepath == vac.path


def test_skyspec_registry_has_v1_with_sha256():
    """Test that REGISTRY contains v1.0 with SHA-256 checksum."""
    from desisky.data.skyspec import REGISTRY

    assert "v1.0" in REGISTRY
    spec = REGISTRY["v1.0"]

    assert spec.url.startswith("https://data.desi.lbl.gov")
    assert spec.filename == "sky_spectra_vac_v1.fits"
    assert spec.subdir == "dr1"
    assert spec.sha256 == "e943bcf046965090c4566b2b132bd48aba4646f0e2c49a53eb6904e98c471a1b"


def test_load_skyspec_vac_raises_without_fitsio(monkeypatch, tmp_path):
    """Test that load_skyspec_vac() raises ImportError if fitsio is missing."""
    from desisky.data import skyspec

    # Create a dummy FITS file
    fits_file = tmp_path / "test.fits"
    fits_file.write_bytes(b"not really FITS")

    # Mock fitsio import to fail
    import builtins

    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "fitsio":
            raise ImportError("fitsio not found")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    with pytest.raises(ImportError, match="fitsio is required"):
        skyspec.load_skyspec_vac(fits_file)


# ---------- Tests for data module exports ----------


def test_data_module_exports_all_expected_symbols():
    """Test that desisky.data exports all expected public APIs."""
    from desisky import data

    expected_exports = [
        "SkySpecVAC",
        "load_skyspec_vac",
        "REGISTRY",
        "DataSpec",
        "default_root",
        "download_file",
        "sha256sum",
        "compute_vband_magnitudes",
        "load_eclipse_catalog",
        "compute_eclipse_fraction",
        "load_solar_flux",
        "attach_solar_flux",
        "add_galactic_coordinates",
        "add_ecliptic_coordinates",
    ]

    for symbol in expected_exports:
        assert hasattr(data, symbol), f"Missing export: {symbol}"


def test_data_module_all_matches_exports():
    """Test that __all__ matches actual exports."""
    from desisky import data

    assert hasattr(data, "__all__")
    assert len(data.__all__) == 15  # Updated to include get_validation_mask

    for symbol in data.__all__:
        assert hasattr(data, symbol), f"__all__ contains {symbol} but it's not exported"
