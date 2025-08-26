from typing import Any, Optional
import jax
import equinox as eqx

from desisky.io.model_io import register_model, ModelSpec

def make_broadbandMLP(*, in_size: int, out_size: int, width_size: int, depth: int, key: Optional[Any] = None
) -> eqx.Module:
    """
    Constructor for the broadband MLP architecture.
    
    Accepts exactly the kwargs stored under 'meta["arch"]' in the checkpoint header.
    """
    if key is None:
        key = jax.random.PRNGKey(32)
    return eqx.nn.MLP(
        in_size=in_size, out_size=out_size,
        width_size=width_size, depth=depth,
        key=key
    )

# Register this model kind with the IO registry. The resource path is relative to
# the 'desisky.weights' package (i.e., 'src/desisky/weights/broadband_weights.eqx')
register_model(
    "broadband",
    ModelSpec(constructor=make_broadbandMLP, resource="broadband_weights.eqx"),
)