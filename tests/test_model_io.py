import json
from pathlib import Path
from dataclasses import dataclass
import pytest
from typing import Callable, Dict, Any      
import equinox as eqx
import jax.numpy as jnp

from desisky.io.model_io import (
    save, load, load_or_builtin, load_builtin, 
    register_model, ModelSpec, REGISTRY
)

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

# # Optional future examples — keep them commented or mark-skipped until available.
# def _vae_case():  # example shape contracts; adjust to your real API
#     vae = pytest.importorskip("desisky.models.vae")        # skip if not present
#     arch = dict(in_size=1024, latent_dim=64, hidden=256, depth=4)
#     def make_input():
#         return jnp.ones((arch["in_size"],))
#     def check_output(y):
#         # suppose forward returns (recon, mu, logvar)
#         recon, mu, logvar = y
#         assert recon.shape == (arch["in_size"],)
#         assert mu.shape == (arch["latent_dim"],)
#         assert logvar.shape == (arch["latent_dim"],)
#     return ModelCase(
#         kind="vae",
#         constructor=vae.make_vae,
#         arch=arch,
#         resource="vae_weights.eqx",
#         make_input=make_input,
#         check_output=check_output,
#     )

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
    cases = [_broadband_case()]
    # Uncomment once we have these models
    # cases.append(_vae_case())
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
    pkg_dir = tmp_path / "pkg"
    packaged = pkg_dir / case.resource 
    model0 = case.constructor(**case.arch)
    _write_nested_header_ckpt(packaged, model0, case.arch, schema=2, extra={"source": "test"})

    register_model(case.kind, ModelSpec(case.constructor, case.resource))
    # Point importlib.resources.files("desisky.weights") to our temp dir
    monkeypatch.setattr("desisky.io.model_io.res.files", lambda _pkg: pkg_dir)

    model, meta = load_builtin(case.kind)
    assert meta["schema"] == 2 and meta["arch"] == case.arch and meta["source"] == "test"

@pytest.mark.parametrize("case", _cases(), ids=lambda c: c.kind)
def test_path_precedence_over_packaged(case, monkeypatch, tmp_path):
    pkg_dir = tmp_path / "pkg"
    pkg_dir.mkdir()
    user_dir = tmp_path / "user"
    user_dir.mkdir()
    packaged = pkg_dir / case.resource 
    userfile = user_dir / f"{case.kind}_override.eqx"

    _write_nested_header_ckpt(packaged, case.constructor(**case.arch), case.arch, schema=1)
    _write_nested_header_ckpt(userfile, case.constructor(**case.arch), case.arch, schema=1)

    register_model(case.kind, ModelSpec(case.constructor, case.resource))
    monkeypatch.setattr("desisky.io.model_io.res.files", lambda _pkg: pkg_dir)

    model_user, meta_user = load_or_builtin(case.kind, path=userfile)
    x = case.make_input()
    # Unpack tuple inputs if model expects multiple
    y = model_user(*x) if isinstance(x, tuple) else model_user(x)
    case.check_output(y)

@pytest.mark.parametrize("case", _cases(), ids=lambda c: c.kind)
def test_inference_smoke_from_packaged(case, monkeypatch, tmp_path):
    pkg_dir = tmp_path / "pkg"
    pkg_dir.mkdir()
    ckpt = pkg_dir / case.resource
    _write_nested_header_ckpt(ckpt, case.constructor(**case.arch), case.arch, schema=1)

    register_model(case.kind, ModelSpec(case.constructor, case.resource))
    monkeypatch.setattr("desisky.io.model_io.res.files", lambda _pkg: pkg_dir)

    model, meta = load_builtin(case.kind)
    x = case.make_input()
    y = model(*x) if isinstance(x, tuple) else model(x)
    case.check_output(y)
