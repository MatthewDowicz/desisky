# SPDX-FileCopyrightText: 2025-present MatthewDowicz <mjdowicz@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Loss functions for sky brightness model training."""

from __future__ import annotations
from typing import Literal

import jax
import jax.numpy as jnp
import equinox as eqx


def loss_l2(pred: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """
    Mean squared error (L2 loss) over batch and all outputs.

    Parameters
    ----------
    pred : jnp.ndarray
        Model predictions. Shape: (batch_size, n_outputs).
    targets : jnp.ndarray
        Ground truth targets. Shape: (batch_size, n_outputs).

    Returns
    -------
    loss : jnp.ndarray
        Scalar loss value (mean squared error).

    Examples
    --------
    >>> pred = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    >>> targets = jnp.array([[1.1, 2.1], [2.9, 3.9]])
    >>> loss = loss_l2(pred, targets)
    """
    return jnp.mean((pred - targets) ** 2)


def loss_huber(
    pred: jnp.ndarray, targets: jnp.ndarray, delta: float = 1.0
) -> jnp.ndarray:
    """
    Huber loss over batch and all outputs (robust to outliers).

    The Huber loss is quadratic for errors smaller than delta and linear for
    larger errors, making it less sensitive to outliers than L2 loss.

    Parameters
    ----------
    pred : jnp.ndarray
        Model predictions. Shape: (batch_size, n_outputs).
    targets : jnp.ndarray
        Ground truth targets. Shape: (batch_size, n_outputs).
    delta : float, default 1.0
        Threshold at which to switch from quadratic to linear loss.

    Returns
    -------
    loss : jnp.ndarray
        Scalar Huber loss value.

    Examples
    --------
    >>> pred = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    >>> targets = jnp.array([[1.1, 2.1], [2.9, 3.9]])
    >>> loss = loss_huber(pred, targets, delta=0.5)
    """
    err = pred - targets
    abs_err = jnp.abs(err)
    quadratic = jnp.minimum(abs_err, delta)
    linear = abs_err - quadratic
    return jnp.mean(0.5 * quadratic**2 + delta * linear)


def loss_func(
    model: eqx.Module,
    inputs: jnp.ndarray,
    targets: jnp.ndarray,
    loss_name: Literal["l2", "huber"] = "l2",
    huber_delta: float = 1.0,
) -> jnp.ndarray:
    """
    Forward pass + compute selected loss.

    This function combines model inference with loss computation in a single
    call, suitable for use with JAX's grad/value_and_grad functions.

    Parameters
    ----------
    model : eqx.Module
        Equinox model (e.g., eqx.nn.MLP).
    inputs : jnp.ndarray
        Batch of input features. Shape: (batch_size, n_features).
    targets : jnp.ndarray
        Batch of target outputs. Shape: (batch_size, n_outputs).
    loss_name : {"l2", "huber"}, default "l2"
        Name of the loss function to use.
    huber_delta : float, default 1.0
        Delta parameter for Huber loss (ignored if loss_name="l2").

    Returns
    -------
    loss : jnp.ndarray
        Scalar loss value.

    Raises
    ------
    ValueError
        If loss_name is not "l2" or "huber".

    Examples
    --------
    >>> import equinox as eqx
    >>> import jax
    >>> model = eqx.nn.MLP(in_size=6, out_size=4, width_size=64, depth=3,
    ...                     key=jax.random.PRNGKey(0))
    >>> inputs = jnp.ones((16, 6))
    >>> targets = jnp.zeros((16, 4))
    >>> loss = loss_func(model, inputs, targets, loss_name="l2")
    """
    pred = jax.vmap(model)(inputs)  # (batch_size, n_outputs)

    if loss_name == "l2":
        return loss_l2(pred, targets)
    elif loss_name == "huber":
        return loss_huber(pred, targets, delta=huber_delta)
    else:
        raise ValueError(f"Unknown loss: {loss_name}. Must be 'l2' or 'huber'.")
