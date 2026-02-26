import json
from pathlib import Path
from dataclasses import dataclass
import pytest
from typing import Callable, Dict, Any
import equinox as eqx
import jax.numpy as jnp

from desisky.io.model_io import (
    save, load, load_or_builtin, load_builtin,
    register_model, ModelSpec, REGISTRY, EXTERNAL_WEIGHTS,
    get_user_model_dir, _resolve_cache_dir,
)
from desisky.io import load_model

# ---------- Model-agnostic test case spec ----------

@dataclass(frozen=True)
class ModelCase:
    kind: str
    constructor: Callable[..., Any]
    arch: Dict[str, Any]
    resource: str
    make_input: Callable[[], Any]
    check_output: Callable[[Any], None]

def _broadband_case():
    from desisky.models.broadband import make_broadbandMLP
    arch = dict(in_size=6, out_size=4, width_size=128, depth=5)
    def make_input():
        return jnp.ones((arch["in_size"],))
    def check_output(y):
        assert y.shape == (arch["out_size"],)
    return ModelCase(
        kind="broadband",
        constructor=make_broadbandMLP,
        arch=arch,
        resource="broadband_weights.eqx",
        make_input=make_input,
        check_output=check_output,
    )

def _vae_case():
    from desisky.models.vae import make_SkyVAE
    import jax.random as jr
    arch = dict(in_channels=7781, latent_dim=8)
    def make_input():
        x = jnp.ones((arch["in_channels"],))
        key = jr.PRNGKey(0)
        return (x, key)
    def check_output(result):
        assert isinstance(result, dict)
        assert result['mean'].shape == (arch["latent_dim"],)
        assert result['output'].shape == (arch["in_channels"],)
    return ModelCase(
        kind="vae",
        constructor=make_SkyVAE,
        arch=arch,
        resource="vae_weights.eqx",
        make_input=make_input,
        check_output=check_output,
    )

def _ldm_case(kind, meta_dim, resource):
    from desisky.models.ldm import make_UNet1D_cond
    arch = dict(in_ch=1, out_ch=1, meta_dim=meta_dim, hidden=32, levels=3, emb_dim=32)
    def make_input():
        # UNet1D_cond forward: (x_noisy, sigma, metadata)
        x = jnp.zeros((arch["in_ch"], 8))   # (C, L)
        t = jnp.array(1.0)                  # scalar timestep
        meta = jnp.ones((arch["meta_dim"],))
        return (x, t, meta)
    def check_output(y):
        assert y.shape == (1, 8)
    return ModelCase(
        kind=kind,
        constructor=make_UNet1D_cond,
        arch=arch,
        resource=resource,
        make_input=make_input,
        check_output=check_output,
    )

def _cases():
    return [
        _broadband_case(),
        _vae_case(),
        _ldm_case("ldm_dark", meta_dim=8, resource="ldm_dark.eqx"),
        _ldm_case("ldm_moon", meta_dim=6, resource="ldm_moon.eqx"),
        _ldm_case("ldm_twilight", meta_dim=4, resource="ldm_twilight.eqx"),
    ]


# ---------- helpers ----------

def _write_nested_header_ckpt(path: Path, model, arch: dict, schema=1, extra=None):
    meta = {"schema": schema, "arch": arch.copy()}
    if extra:
        meta.update(extra)
    save(path, model, meta)

def _mock_cache_with_checkpoint(monkeypatch, tmp_path, case, schema=2, extra=None):
    """
    Set up a mocked cache directory with a pre-written checkpoint.

    Uses the DESISKY_CACHE_DIR env var so _resolve_cache_dir() returns
    tmp_path/cache/<kind>/ as the cache directory.
    """
    if extra is None:
        extra = {"source": "test"}
    cache_base = tmp_path / "cache"
    cache_dir = cache_base / case.kind
    cache_dir.mkdir(parents=True)

    model0 = case.constructor(**case.arch)
    ckpt = cache_dir / case.resource
    _write_nested_header_ckpt(ckpt, model0, case.arch, schema=schema, extra=extra)

    monkeypatch.setenv("DESISKY_CACHE_DIR", str(cache_base))
    register_model(case.kind, ModelSpec(case.constructor, case.resource))
    return ckpt


# ---------- Core save/load ----------

@pytest.mark.parametrize("case", _cases(), ids=lambda c: c.kind)
def test_roundtrip_save_load(case, tmp_path):
    """Save a model and load it back; verify architecture and metadata."""
    ckpt = tmp_path / "rt.eqx"
    model0 = case.constructor(**case.arch)
    _write_nested_header_ckpt(ckpt, model0, case.arch, schema=1, extra={"note": "rt"})

    model1, meta1 = load(ckpt, constructor=case.constructor)
    assert isinstance(model1, eqx.Module)
    assert meta1["schema"] == 1
    assert meta1["arch"] == case.arch
    assert meta1["note"] == "rt"

@pytest.mark.parametrize("case", _cases(), ids=lambda c: c.kind)
def test_constructor_mismatch_raises(case, tmp_path):
    """Missing arch kwargs should raise TypeError."""
    ckpt = tmp_path / "bad.eqx"
    bad_arch = case.arch.copy()
    bad_arch.pop(next(iter(bad_arch)))
    ckpt.write_text(json.dumps({"schema": 1, "arch": bad_arch}) + "\n")
    with pytest.raises(TypeError):
        load(ckpt, constructor=case.constructor)


# ---------- Builtin loading (mocked cache) ----------

@pytest.mark.parametrize("case", _cases(), ids=lambda c: c.kind)
def test_builtin_load_and_inference(case, monkeypatch, tmp_path):
    """Load builtin weights from mocked cache and run inference."""
    _mock_cache_with_checkpoint(monkeypatch, tmp_path, case)

    model, meta = load_builtin(case.kind)
    assert meta["arch"] == case.arch

    x = case.make_input()
    y = model(*x) if isinstance(x, tuple) else model(x)
    case.check_output(y)

@pytest.mark.parametrize("case", _cases(), ids=lambda c: c.kind)
def test_path_precedence_over_builtin(case, monkeypatch, tmp_path):
    """User-provided path should take precedence over builtin weights."""
    _mock_cache_with_checkpoint(monkeypatch, tmp_path, case)

    userfile = tmp_path / "user" / f"{case.kind}_override.eqx"
    userfile.parent.mkdir(parents=True)
    _write_nested_header_ckpt(userfile, case.constructor(**case.arch), case.arch, schema=1)

    model_user, _ = load_or_builtin(case.kind, path=userfile)
    x = case.make_input()
    y = model_user(*x) if isinstance(x, tuple) else model_user(x)
    case.check_output(y)


# ---------- Public API: load_model() ----------

@pytest.mark.parametrize("case", _cases(), ids=lambda c: c.kind)
def test_load_model_with_lazy_registration(case, monkeypatch, tmp_path):
    """load_model() triggers lazy registration and loads from cache."""
    _mock_cache_with_checkpoint(monkeypatch, tmp_path, case)

    model, meta = load_model(case.kind)
    assert meta["arch"] == case.arch
    assert case.kind in REGISTRY

@pytest.mark.parametrize("case", _cases(), ids=lambda c: c.kind)
def test_load_model_from_user_checkpoint(case, tmp_path):
    """load_model(kind, path=...) loads a user-trained checkpoint."""
    ckpt = tmp_path / "my_model.eqx"
    model0 = case.constructor(**case.arch)
    _write_nested_header_ckpt(ckpt, model0, case.arch, schema=1, extra={"note": "user-trained"})

    model, meta = load_model(case.kind, path=ckpt)
    assert meta["note"] == "user-trained"
    assert meta["arch"] == case.arch


# ---------- Error handling ----------

def test_load_nonexistent_file(tmp_path):
    """Loading a nonexistent file raises FileNotFoundError."""
    from desisky.models.broadband import make_broadbandMLP
    with pytest.raises(FileNotFoundError):
        load(tmp_path / "nope.eqx", constructor=make_broadbandMLP)

def test_load_builtin_unknown_kind():
    """Loading an unregistered kind raises KeyError."""
    with pytest.raises(KeyError, match="Unknown model kind"):
        load_builtin("nonexistent_model_type")

@pytest.mark.parametrize("case", _cases(), ids=lambda c: c.kind)
def test_corrupted_checkpoint_empty_file(case, tmp_path):
    """Empty file raises JSONDecodeError or ValueError."""
    corrupted = tmp_path / "corrupted.eqx"
    corrupted.write_bytes(b"")
    with pytest.raises((json.JSONDecodeError, ValueError)):
        load(corrupted, constructor=case.constructor)

@pytest.mark.parametrize("case", _cases(), ids=lambda c: c.kind)
def test_missing_arch_in_header(case, tmp_path):
    """Empty JSON header (no arch) raises ValueError."""
    bad_ckpt = tmp_path / "bad.eqx"
    bad_ckpt.write_text(json.dumps({}) + "\n")
    with pytest.raises(ValueError, match="missing 'arch'"):
        load(bad_ckpt, constructor=case.constructor)


# ---------- Metadata round-tripping ----------

@pytest.mark.parametrize("case", _cases(), ids=lambda c: c.kind)
def test_schema_versioning(case, tmp_path):
    """Schema versions are preserved across save/load."""
    for v in [0, 1, 2, 100]:
        ckpt = tmp_path / f"schema_v{v}.eqx"
        _write_nested_header_ckpt(ckpt, case.constructor(**case.arch), case.arch, schema=v)
        _, meta = load(ckpt, constructor=case.constructor)
        assert meta["schema"] == v

@pytest.mark.parametrize("case", _cases(), ids=lambda c: c.kind)
def test_extra_metadata_preserved(case, tmp_path):
    """Arbitrary extra metadata survives a round-trip."""
    ckpt = tmp_path / "with_metadata.eqx"
    extra = {"training": {"date": "2025-01-15", "loss": 0.042}, "note": "best"}
    _write_nested_header_ckpt(ckpt, case.constructor(**case.arch), case.arch, extra=extra)

    _, meta = load(ckpt, constructor=case.constructor)
    assert meta["training"] == extra["training"]
    assert meta["note"] == extra["note"]


# ---------- Cache directory resolution ----------

def test_resolve_cache_dir_env_var(monkeypatch, tmp_path):
    """DESISKY_CACHE_DIR env var overrides the default cache location."""
    monkeypatch.setenv("DESISKY_CACHE_DIR", str(tmp_path))
    assert _resolve_cache_dir("vae") == tmp_path / "vae"

def test_resolve_cache_dir_default(monkeypatch):
    """Without env var, falls back to default_root() / models / kind."""
    from desisky.data._core import default_root
    monkeypatch.delenv("DESISKY_CACHE_DIR", raising=False)
    assert _resolve_cache_dir("broadband") == default_root() / "models" / "broadband"

def test_get_user_model_dir():
    """get_user_model_dir() returns expected paths."""
    assert get_user_model_dir() == Path.home() / ".cache" / "desisky" / "saved_models"
    assert get_user_model_dir("vae") == Path.home() / ".cache" / "desisky" / "saved_models" / "vae"
