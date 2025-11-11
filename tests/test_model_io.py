import json
from pathlib import Path
from dataclasses import dataclass
import pytest
from typing import Callable, Dict, Any      
import equinox as eqx
import jax.numpy as jnp

from desisky.io.model_io import (
    save, load, load_or_builtin, load_builtin,
    register_model, ModelSpec, REGISTRY, get_user_model_dir
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
        # VAE forward pass needs (x, key)
        x = jnp.ones((arch["in_channels"],))
        key = jr.PRNGKey(0)
        return (x, key)
    def check_output(result):
        # VAE forward returns dict with 'mean', 'logvar', 'latent', 'output'
        assert isinstance(result, dict)
        assert result['mean'].shape == (arch["latent_dim"],)
        assert result['logvar'].shape == (arch["latent_dim"],)
        assert result['latent'].shape == (arch["latent_dim"],)
        assert result['output'].shape == (arch["in_channels"],)
    return ModelCase(
        kind="vae",
        constructor=make_SkyVAE,
        arch=arch,
        resource="vae_weights.eqx",
        make_input=make_input,
        check_output=check_output,
    )

# def _diffusion_case():  # example; adjust to your UNet/diffuser API
#     dm = pytest.importorskip("desisky.models.diffusion")
#     arch = dict(signal_dim=1024, base_width=64, depth=4)
#     def make_input():
#         # e.g., a (signal, timestep) pair if your forward uses t
#         x = jnp.ones((arch["signal_dim"],))
#         t = jnp.array(10, dtype=jnp.int32)
#         return (x, t)
#     def check_output(y):
#         # e.g., predicted noise same shape as x
#         assert y.shape == (arch["signal_dim"],)
#     return ModelCase(
#         kind="diffusion",
#         constructor=dm.make_unet,   # or your top-level constructor
#         arch=arch,
#         resource="diffusion_weights.eqx",
#         make_input=make_input,
#         check_output=check_output,
#     )

def _cases():
    cases = [_broadband_case(), _vae_case()]
    # Uncomment once we have these models
    # cases.append(_diffusion_case())
    return cases 


# ---------- helpers ----------

def _write_nested_header_ckpt(path: Path, model, arch: dict, schema=1, extra=None):
    meta = {"schema": schema, "arch": arch.copy()}
    if extra:
        meta.update(extra)
    save(path, model, meta)


# ---------- Parametrized tests ----------

@pytest.mark.parametrize("case", _cases(), ids=lambda c: c.kind)
def test_roundtrip_save_load(case, tmp_path):
    ckpt = tmp_path / case.kind / "rt.eqx"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    model0 = case.constructor(**case.arch)
    _write_nested_header_ckpt(ckpt, model0, case.arch, schema=1, extra={"note": "rt"})

    model1, meta1 = load(ckpt, constructor=case.constructor)
    assert isinstance(model1, eqx.Module)
    assert meta1["schema"] == 1
    assert meta1["arch"] == case.arch 
    assert meta1["note"] == "rt"

@pytest.mark.parametrize("case", _cases(), ids=lambda c: c.kind)
def test_constructor_mismatch_raises(case, tmp_path):
    ckpt = tmp_path / case.kind / "bad.eqx"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    # Remove one required kwarg from arch
    bad_arch = case.arch.copy()
    bad_arch.pop(next(iter(bad_arch))) # drop first key 
    ckpt.write_text( json.dumps({"schema": 1, "arch": bad_arch}) + "\n")
    with pytest.raises(TypeError):
        load(ckpt, constructor=case.constructor)

@pytest.mark.parametrize("case", _cases(), ids=lambda c: c.kind)
def test_packaged_load_via_registry(case, monkeypatch, tmp_path):
    """
    Test loading pre-packaged weights via the registry.

    This test manually registers the model (since we're using a temp directory
    instead of the real desisky.weights package). In production, models auto-register
    themselves on import (see broadband.py).
    """
    pkg_dir = tmp_path / "pkg"
    packaged = pkg_dir / case.resource
    model0 = case.constructor(**case.arch)
    _write_nested_header_ckpt(packaged, model0, case.arch, schema=2, extra={"source": "test"})

    # Manually register for this test since we're mocking the package location
    register_model(case.kind, ModelSpec(case.constructor, case.resource))
    # Point importlib.resources.files("desisky.weights") to our temp dir
    monkeypatch.setattr("desisky.io.model_io.res.files", lambda _pkg: pkg_dir)

    model, meta = load_builtin(case.kind)
    assert meta["schema"] == 2 and meta["arch"] == case.arch and meta["source"] == "test"

@pytest.mark.parametrize("case", _cases(), ids=lambda c: c.kind)
def test_path_precedence_over_packaged(case, monkeypatch, tmp_path):
    """
    Test that user-provided checkpoint paths take precedence over packaged weights.

    This ensures users can override packaged models with their own fine-tuned versions.
    """
    pkg_dir = tmp_path / "pkg"
    pkg_dir.mkdir()
    user_dir = tmp_path / "user"
    user_dir.mkdir()
    packaged = pkg_dir / case.resource
    userfile = user_dir / f"{case.kind}_override.eqx"

    _write_nested_header_ckpt(packaged, case.constructor(**case.arch), case.arch, schema=1)
    _write_nested_header_ckpt(userfile, case.constructor(**case.arch), case.arch, schema=1)

    # Manually register for this test
    register_model(case.kind, ModelSpec(case.constructor, case.resource))
    monkeypatch.setattr("desisky.io.model_io.res.files", lambda _pkg: pkg_dir)

    model_user, meta_user = load_or_builtin(case.kind, path=userfile)
    x = case.make_input()
    # Unpack tuple inputs if model expects multiple
    y = model_user(*x) if isinstance(x, tuple) else model_user(x)
    case.check_output(y)

@pytest.mark.parametrize("case", _cases(), ids=lambda c: c.kind)
def test_inference_smoke_from_packaged(case, monkeypatch, tmp_path):
    """
    End-to-end smoke test: load packaged weights and run inference.

    This verifies the full pipeline works: registry lookup -> weight loading -> inference.
    """
    pkg_dir = tmp_path / "pkg"
    pkg_dir.mkdir()
    ckpt = pkg_dir / case.resource
    _write_nested_header_ckpt(ckpt, case.constructor(**case.arch), case.arch, schema=1)

    # Manually register for this test
    register_model(case.kind, ModelSpec(case.constructor, case.resource))
    monkeypatch.setattr("desisky.io.model_io.res.files", lambda _pkg: pkg_dir)

    model, meta = load_builtin(case.kind)
    x = case.make_input()
    y = model(*x) if isinstance(x, tuple) else model(x)
    case.check_output(y)


# ---------- New tests: lazy registration, error cases, edge cases ----------

@pytest.mark.parametrize("case", _cases(), ids=lambda c: c.kind)
def test_load_model_with_lazy_registration(case, monkeypatch, tmp_path):
    """
    Test the user-facing load_model() API with lazy registration.

    This is the main function users call (from desisky.io.load_model).
    It should trigger lazy registration via _ensure_registered() on first use.

    The lazy registration mechanism imports the model module, which causes
    the model to auto-register itself (e.g., broadband.py registers on import).

    Note: In Python, modules are only executed once on first import. So if
    the module was already imported elsewhere in the test suite, importing
    it again won't re-run the registration code. This test validates that
    load_model() works when the registry already has the model (the common case).
    """
    pkg_dir = tmp_path / "pkg"
    pkg_dir.mkdir()
    ckpt = pkg_dir / case.resource
    _write_nested_header_ckpt(ckpt, case.constructor(**case.arch), case.arch, schema=1)

    # Mock the package weights location
    monkeypatch.setattr("desisky.io.model_io.res.files", lambda _pkg: pkg_dir)

    # Use the high-level load_model() API (this is what users call)
    # The broadband module should have auto-registered when it was first imported
    model, meta = load_model(case.kind)

    # Verify it worked
    assert meta["arch"] == case.arch
    x = case.make_input()
    y = model(*x) if isinstance(x, tuple) else model(x)
    case.check_output(y)

    # Verify the model is in the registry (may have been there already or just registered)
    assert case.kind in REGISTRY


@pytest.mark.parametrize("case", _cases(), ids=lambda c: c.kind)
def test_load_model_from_user_checkpoint(case, tmp_path):
    """
    Test load_model() with a user-provided checkpoint path.

    This tests the common workflow: train a model, save it, load it later.
    When loading from a path, the constructor is taken from the registry,
    so we need to ensure the model is registered first.
    """
    ckpt = tmp_path / "my_model.eqx"
    model0 = case.constructor(**case.arch)
    _write_nested_header_ckpt(ckpt, model0, case.arch, schema=1, extra={"note": "user-trained"})

    # Load using the high-level API with explicit path
    # load_model() will trigger lazy registration to get the constructor
    model, meta = load_model(case.kind, path=ckpt)

    assert meta["note"] == "user-trained"
    assert meta["arch"] == case.arch

    # Verify it works
    x = case.make_input()
    y = model(*x) if isinstance(x, tuple) else model(x)
    case.check_output(y)


def test_load_nonexistent_file(tmp_path):
    """
    Test that loading a nonexistent file raises a clear error.
    """
    from desisky.models.broadband import make_broadbandMLP

    nonexistent = tmp_path / "does_not_exist.eqx"

    with pytest.raises(FileNotFoundError):
        load(nonexistent, constructor=make_broadbandMLP)


def test_load_builtin_unknown_kind():
    """
    Test that loading an unregistered model kind raises KeyError with helpful message.
    """
    with pytest.raises(KeyError, match="Unknown model kind"):
        load_builtin("nonexistent_model_type")


@pytest.mark.parametrize("case", _cases(), ids=lambda c: c.kind)
def test_corrupted_checkpoint_empty_file(case, tmp_path):
    """
    Test that loading an empty/corrupted checkpoint raises a clear error.
    """
    corrupted = tmp_path / "corrupted.eqx"
    corrupted.write_bytes(b"")  # Empty file

    with pytest.raises((json.JSONDecodeError, ValueError)):
        load(corrupted, constructor=case.constructor)


@pytest.mark.parametrize("case", _cases(), ids=lambda c: c.kind)
def test_missing_arch_in_header(case, tmp_path):
    """
    Test that a checkpoint with missing/empty 'arch' dict raises ValueError.

    The header must contain either:
    1. A nested structure with 'arch' key: {"schema": 1, "arch": {...}}
    2. A flat structure where top-level keys are arch params (legacy)

    An empty header should fail since there's no arch information.
    """
    bad_ckpt = tmp_path / "bad.eqx"
    # Write an empty JSON dict - no arch, no params
    bad_ckpt.write_text(json.dumps({}) + "\n")

    with pytest.raises(ValueError, match="missing 'arch'"):
        load(bad_ckpt, constructor=case.constructor)


def test_get_user_model_dir():
    """
    Test that get_user_model_dir() returns expected paths.
    """
    # Without kind argument
    base_dir = get_user_model_dir()
    assert base_dir == Path.home() / ".cache" / "desisky" / "saved_models"

    # With kind argument
    kind_dir = get_user_model_dir("broadband")
    assert kind_dir == Path.home() / ".cache" / "desisky" / "saved_models" / "broadband"

    # With different kind
    vae_dir = get_user_model_dir("vae")
    assert vae_dir == Path.home() / ".cache" / "desisky" / "saved_models" / "vae"


@pytest.mark.parametrize("case", _cases(), ids=lambda c: c.kind)
def test_schema_versioning(case, tmp_path):
    """
    Test that schema versions are preserved across save/load cycles.

    This ensures future schema changes can be handled gracefully.
    """
    for schema_version in [0, 1, 2, 100]:
        ckpt = tmp_path / f"schema_v{schema_version}.eqx"
        model0 = case.constructor(**case.arch)
        _write_nested_header_ckpt(ckpt, model0, case.arch, schema=schema_version)

        model1, meta = load(ckpt, constructor=case.constructor)
        assert meta["schema"] == schema_version, f"Schema version {schema_version} not preserved"
        assert meta["arch"] == case.arch


@pytest.mark.parametrize("case", _cases(), ids=lambda c: c.kind)
def test_extra_metadata_preserved(case, tmp_path):
    """
    Test that arbitrary extra metadata in the header is preserved.

    Users might want to store training info, git commits, dates, etc.
    """
    ckpt = tmp_path / "with_metadata.eqx"
    model0 = case.constructor(**case.arch)

    extra_meta = {
        "training": {
            "date": "2025-01-15",
            "commit": "abc123",
            "loss": 0.042,
        },
        "note": "best model so far",
        "author": "test_suite",
    }

    _write_nested_header_ckpt(ckpt, model0, case.arch, schema=1, extra=extra_meta)

    model1, meta = load(ckpt, constructor=case.constructor)

    # Verify all extra metadata is preserved
    assert meta["training"] == extra_meta["training"]
    assert meta["note"] == extra_meta["note"]
    assert meta["author"] == extra_meta["author"]
    # Standard fields should also be present
    assert meta["schema"] == 1
    assert meta["arch"] == case.arch
