from pathlib import Path
from typing import Any, Dict, Tuple, Callable, Optional
import json
import equinox as eqx
from dataclasses import dataclass



@dataclass(frozen=True)
class ModelSpec:
    """
    Specification for a registered model type.

    Attributes:
    -----------
    constructor : Callable[..., Any]
        Callable that instantiates an *uninitialized* model from keyword args in
        the header's ``arch`` dict (e.g., ``in_size``, ``out_size``, ``width_size``, ``depth``).
        The returned object must have the same PyTree structure that the checkpoint expects
        so that ``equinox.tree_deserialise_leaves`` can load parameters into it.
    resource : str
        Filename for the weights file, used as the HuggingFace resource name
        and local cache filename (e.g., ``"broadband_weights.eqx"``).
    """
    constructor: Callable[..., Any]
    resource: str 

# Global registry; populated at import-time in desisky/__init__.py 
REGISTRY: Dict[str, ModelSpec] = {}


# -------------------------
# Paths & small utils
# -------------------------
def get_user_model_dir(kind: str | None = None) -> Path:
    """
    Return the default directory for user-trained checkpoints.

    The default is ``~/.cache/desisky/saved_models[/<kind>]``. This is a convenience
    location only; callers are free to save/load from any path (e.g., project folders,
    NERSC ``$SCRATCH``/``$CFS``).

    Parameters
    ----------
    kind : str | None
        Optional subfolder name (e.g., ``"broadband"``, ``"vae"``, ``"unet"``).

    Returns
    -------
    Path
        The directory path.
    """
    base = Path.home() / ".cache" / "desisky" / "saved_models"
    return base / kind if kind else base 

def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def _read_header_and_deserialize(fp, constructor) -> Tuple[Any, Dict[str, Any]]:
    """
    Read one JSON header line, construct the model, then deserialize leaves.

    File format
    -----------
    The checkpoint format is:

    1. One line of JSON (UTF-8) terminated by ``\\n``. Recommended shape:
       ``{"schema": <int>, "arch": {...}, ...}``.
       For legacy files, a *flat* header where the top-level keys are the arch kwargs
       is also accepted.
    2. The Equinox payload written by :func:`equinox.tree_serialise_leaves`.

    Parameters
    ----------
    fp : BinaryIO
        Opened binary file-like object positioned at the start of the checkpoint.
    constructor : Callable[..., Any]
        Callable used to build an uninitialized model from the header's ``arch`` kwargs.

    Returns
    -------
    (model, meta) : tuple[Any, dict[str, Any]]
        ``model`` is the deserialized Equinox module. ``meta`` is a normalized dict
        that always contains at least ``{"schema": int, "arch": dict}`` and preserves
        any additional top-level header keys.

    Raises
    ------
    ValueError
        If the header is missing/empty or does not provide ``arch`` (nested or flat).
    json.JSONDecodeError
        If the first line is not valid JSON.
    """
    header_line = fp.readline().decode()
    header = json.loads(header_line) if header_line else {}

    arch = header.get("arch")
    if arch is None:
        if isinstance(header, dict) and len(header) > 0:
            arch = header # Accept legacy flat header
        else:
            raise ValueError("Checkpoint header missing 'arch' & does not look like a flat arch dict.")

    model = constructor(**arch)
    model = eqx.tree_deserialise_leaves(fp, model)

    # Normalize meta so we always return at least {"schema": int, "arch: {...}}
    meta: Dict[str, Any] = {"schema": int(header.get("schema", 0)), "arch": arch}
    for k, v in header.items():
        if k not in ("arch", "schema"):
            meta[k] = v
    return model, meta 


# -------------------------
# Public API: save / load (user checkpoints)
# -------------------------

def save(path: str | Path, model: Any, meta: Dict[str, Any]) -> Path:
    """
    Save an Equinox model with a one-line JSON header.

    The written file has:
      1) a JSON header line (``meta``) then
      2) the payload from :func:`equinox.tree_serialise_leaves`.

    Required keys in ``meta``:
      - ``"schema"`` : int
      - ``"arch"``   : dict of constructor kwargs (e.g., sizes/depth)

    Parameters
    ----------
    path : str | Path
        Destination path. Parent directories are created if needed.
    model : Any
        The Equinox module to serialize.
    meta : dict[str, Any]
        JSON-serializable header. Consider adding non-arch info like
        ``{"training": {"date": "...", "commit": "..."}}``.

    Returns
    -------
    Path
        The path that was written.
    """
    p = Path(path)
    _ensure_parent(p)
    with p.open("wb") as f:
        f.write((json.dumps(meta) + "\n").encode())
        eqx.tree_serialise_leaves(f, model)
    return p 

def load(path: str | Path, constructor) -> Tuple[Any, Dict[str, Any]]:
    """
    Load an Equinox model from a filesystem path created by :func:`save`.

    Parameters
    ----------
    path : str | Path
        Path to the checkpoint file.
    constructor : Callable[..., Any]
        Callable used to reconstruct the uninitialized model from ``meta['arch']``.

    Returns
    -------
    (model, meta) : tuple[Any, dict[str, Any]]
        The model and the normalized header dict.
    """
    p = Path(path)
    with p.open("rb") as f:
        return _read_header_and_deserialize(f, constructor)
    

# -------------------------
# External weights management (pre-trained models on HuggingFace)
# -------------------------

# Download URLs for pre-trained model weights hosted on HuggingFace
EXTERNAL_WEIGHTS = {
    "broadband": {
        "url": "https://huggingface.co/datasets/mjdowicz/desisky/resolve/main/broadband_weights.eqx",
        "sha256": "ed355fd0577db3c022dd31ffa79acbca5774d558e19b2bdd6dee269af6394cf1",
        "size_mb": 0.27,
    },
    "vae": {
        "url": "https://huggingface.co/datasets/mjdowicz/desisky/resolve/main/vae_weights.eqx",
        "sha256": "638948543d7c2dcadd42efbc62c9558f9f9779440aa97bd47220a3c91e42d607",
        "size_mb": 76.2,
    },
    "ldm_dark": {
        "url": "https://huggingface.co/datasets/mjdowicz/desisky/resolve/main/ldm_dark.eqx",
        "sha256": "b3aefd0fa9a59dc65bcbded954122d4d21f95cdea43ca34111dc51a18109e0f0",
        "size_mb": 1.33,
    },
    "ldm_moon": {
        "url": "https://huggingface.co/datasets/mjdowicz/desisky/resolve/main/ldm_moon.eqx",
        "sha256": "a6f875f743923cedea78ce451bcfac9b94733bc6f542defbb6be3cb208503ac9",
        "size_mb": 1.32,
    },
    "ldm_twilight": {
        "url": "https://huggingface.co/datasets/mjdowicz/desisky/resolve/main/ldm_twilight.eqx",
        "sha256": "d269d78dfa09c745ac1c19c5081d9895d9ad8b5e6c080675455d633cb89179a0",
        "size_mb": 1.31,
    }
}

def _download_model_weights(kind: str, dest: Path) -> None:
    """
    Download model weights from external storage with checksum verification.

    For Hugging Face datasets, set the HF_TOKEN environment variable if the
    repository is private:
        export HF_TOKEN=hf_your_token_here

    Parameters
    ----------
    kind : str
        Model kind (e.g., 'vae')
    dest : Path
        Destination path for downloaded weights

    Raises
    ------
    ValueError
        If checksum verification fails
    requests.HTTPError
        If download fails (e.g., 401 for private repos without HF_TOKEN)
    """
    import os
    import requests
    import tempfile
    from desisky.data._core import sha256sum

    if kind not in EXTERNAL_WEIGHTS:
        raise ValueError(f"No external weights configured for '{kind}'")

    info = EXTERNAL_WEIGHTS[kind]
    url = info["url"]
    expected_sha = info.get("sha256")

    print(f"Downloading {kind} weights ({info['size_mb']}MB, first time only)...")
    print(f"Source: {url}")

    # Check for Hugging Face token (for private repos)
    headers = {}
    hf_token = os.getenv("HF_TOKEN")
    if hf_token and "huggingface.co" in url:
        headers["Authorization"] = f"Bearer {hf_token}"

    dest.parent.mkdir(parents=True, exist_ok=True)

    # Download with streaming to handle large files
    try:
        with requests.get(url, stream=True, headers=headers, timeout=300) as r:
            r.raise_for_status()

            # Stream to temporary file
            with tempfile.NamedTemporaryFile(delete=False, dir=str(dest.parent)) as tmp:
                for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        tmp.write(chunk)
                tmp_path = Path(tmp.name)

        # Verify checksum
        if expected_sha:
            print("Verifying checksum...")
            actual = sha256sum(tmp_path)
            if actual != expected_sha:
                tmp_path.unlink()
                raise ValueError(
                    f"Checksum mismatch for {kind} weights.\n"
                    f"Expected: {expected_sha}\n"
                    f"Got:      {actual}\n"
                    f"The download may be corrupted. Please try again."
                )

        # Move to final destination
        tmp_path.replace(dest)
        print(f"✓ {kind} weights downloaded and verified successfully")

    except requests.HTTPError as e:
        if e.response.status_code == 401:
            raise ValueError(
                f"Unauthorized access to {kind} weights.\n"
                f"This repository may be private. If so, set your Hugging Face token:\n"
                f"  export HF_TOKEN=hf_your_token_here\n"
                f"Get your token at: https://huggingface.co/settings/tokens"
            ) from e
        raise

# -------------------------
# Public API: builtin weights
# -------------------------

def _resolve_cache_dir(kind: str) -> Path:
    """
    Resolve the cache directory for downloading pre-trained weights.

    Priority order:
    1. ``DESISKY_CACHE_DIR`` environment variable (if set)
    2. Default: ``~/.desisky/models/<kind>/``

    Parameters
    ----------
    kind : str
        Model kind (used as subfolder name in the default path).

    Returns
    -------
    Path
        Resolved cache directory.
    """
    import os
    env_dir = os.getenv("DESISKY_CACHE_DIR")
    if env_dir:
        return Path(env_dir) / kind

    from desisky.data._core import default_root
    return default_root() / "models" / kind


def load_builtin(kind: str) -> Tuple[Any, Dict[str, Any]]:
    """
    Load pre-trained weights for a registered ``kind``.

    Weights are downloaded from HuggingFace on first use and cached locally.
    Subsequent calls load from cache.

    The cache location defaults to ``~/.desisky/models/<kind>/`` and can be
    overridden by setting the ``DESISKY_CACHE_DIR`` environment variable::

        export DESISKY_CACHE_DIR=/scratch/weights   # shell
        os.environ["DESISKY_CACHE_DIR"] = "/scratch/weights"  # Python / notebook

    Parameters
    ----------
    kind : str
        Registered model kind (e.g., ``'broadband'``, ``'vae'``, ``'ldm_dark'``)

    Returns
    -------
    (model, meta) : tuple[Any, dict[str, Any]]
        Loaded model and normalized metadata dictionary

    Examples
    --------
    >>> from desisky.io import load_builtin
    >>> vae, meta = load_builtin("vae")

    Raises
    ------
    KeyError
        If ``kind`` is not registered or has no external weights configured
    ValueError
        If download fails or checksum verification fails
    """
    if kind not in REGISTRY:
        raise KeyError(f"Unknown model kind '{kind}'. Registered: {list(REGISTRY)}")

    spec = REGISTRY[kind]

    if kind not in EXTERNAL_WEIGHTS:
        raise KeyError(
            f"Model '{kind}' is registered but has no external weights configured. "
            f"Provide a checkpoint path via load_model('{kind}', path=...)"
        )

    resolved_cache = _resolve_cache_dir(kind)
    resolved_cache.mkdir(parents=True, exist_ok=True)
    weights_file = resolved_cache / spec.resource

    # Download if not already cached
    if not weights_file.exists():
        _download_model_weights(kind, weights_file)

    # Load from cache
    return load(weights_file, constructor=spec.constructor)


def load_or_builtin(kind: str,
                    path: Optional[str | Path] = None,
                    constructor: Optional[Callable[..., Any]] = None
                    ) -> Tuple[Any, Dict[str, Any]]:
    """
    Load a user checkpoint from ``path`` if provided; otherwise load the builtin weights
    registered under ``kind``.

    Parameters
    ----------
    kind : str
        Registry key for the model (used for builtin weights lookup and, by default,
        to look up the constructor).
    path : str | Path | None
        Optional filesystem path to a user checkpoint created by :func:`save`.
        If provided, this takes precedence over builtin weights.
    constructor : Callable[..., Any] | None
        Optional explicit constructor. If omitted and ``path`` is given, the constructor
        will be taken from the registry entry for ``kind``.

    Returns
    -------
    (model, meta) : tuple[Any, dict[str, Any]]

    Raises
    ------
    KeyError
        If ``path`` is provided and no constructor is supplied and ``kind`` is not registered.
    """
    if path is not None:
        ctor = constructor
        if ctor is None:
            if kind not in REGISTRY:
                raise KeyError("Provide 'constructor=' when loading from a custom path or register the kind.")
            ctor = REGISTRY[kind].constructor
        return load(path, constructor=ctor)
    return load_builtin(kind)

def register_model(kind: str, spec: ModelSpec) -> None:
    """
    Register a model kind (constructor + resource filename).

    Typical usage
    -------------
    In a model submodule (e.g., ``desisky/models/broadband.py``) do:

    >>> from desisky.io.model_io import register_model, ModelSpec
    >>> from desisky.models.broadband import make_broadbandMLP
    >>> register_model("broadband", ModelSpec(make_broadbandMLP, "broadband_weights.eqx"))
    """
    REGISTRY[kind] = spec 
    


