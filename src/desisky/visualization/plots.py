# SPDX-FileCopyrightText: 2025-present MatthewDowicz <mjdowicz@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Plotting functions for model diagnostics and training visualization."""

from __future__ import annotations
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import equinox as eqx

try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    from matplotlib.figure import Figure
except ImportError:
    raise ImportError(
        "matplotlib is required for visualization. "
        "Install with: pip install matplotlib"
    )


def plot_loss_curve(
    train_losses: Sequence[float],
    test_losses: Sequence[float],
    *,
    title: str = "Loss Curve",
    figsize: tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plot training and test loss curves.

    Parameters
    ----------
    train_losses : Sequence[float]
        Training loss at each epoch.
    test_losses : Sequence[float]
        Test/validation loss at each epoch.
    title : str, default "Loss Curve"
        Plot title.
    figsize : tuple[int, int], default (12, 5)
        Figure size in inches (width, height).
    save_path : str | None, default None
        If provided, save the figure to this path.

    Returns
    -------
    fig : Figure
        Matplotlib figure object.

    Examples
    --------
    >>> from desisky.visualization import plot_loss_curve
    >>> fig = plot_loss_curve(history.train_losses, history.test_losses,
    ...                        title="Broadband Model Training")
    >>> plt.show()
    """
    epochs = np.arange(len(train_losses))

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(epochs, train_losses, label="Training Loss", linewidth=2)
    ax.plot(epochs, test_losses, label="Test Loss", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_broadband_band_panel(
    model: eqx.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    band_idx: int,
    band_name: str,
) -> Figure:
    """
    2x2 diagnostic panel for one broadband band.

    Top-left: Train scatter (Obs vs Pred) with 1:1 line + +/-sigma_test rails.
    Bottom-left: Test scatter (same format).
    Top-right: Train residual histogram with 0 + +/-sigma_test lines.
    Bottom-right: Test residual histogram (same format).

    sigma_test = std of test residuals (the "honest" estimate of prediction
    uncertainty on unseen data; training sigma would be optimistically biased).

    Parameters
    ----------
    model : eqx.Module
        Trained broadband MLP.
    X_train, y_train : np.ndarray
        Training inputs and targets.
    X_test, y_test : np.ndarray
        Test inputs and targets.
    band_idx : int
        Index of the band (0=V, 1=g, 2=r, 3=z).
    band_name : str
        Name of the band for titles.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    pred_train = np.array(jax.vmap(model)(jnp.asarray(X_train)))
    pred_test = np.array(jax.vmap(model)(jnp.asarray(X_test)))
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    resid_train = pred_train[:, band_idx] - y_train[:, band_idx]
    resid_test = pred_test[:, band_idx] - y_test[:, band_idx]
    rmse_train = np.sqrt(np.mean(resid_train**2))
    rmse_test = np.sqrt(np.mean(resid_test**2))
    sigma_test = np.std(resid_test)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    x_min, x_max = 15.9, 22.0

    # Left column: Scatter plots (Obs vs Pred)
    for row, (y_obs, y_pred, rmse, label) in enumerate([
        (y_train[:, band_idx], pred_train[:, band_idx], rmse_train, "Train"),
        (y_test[:, band_idx], pred_test[:, band_idx], rmse_test, "Test"),
    ]):
        ax = axs[row, 0]
        ax.scatter(y_obs, y_pred, alpha=0.25, s=10)
        ax.plot([x_min, x_max], [x_min, x_max], ls="--", c="black", label="1:1")
        ax.plot([x_min, x_max], [x_min + sigma_test, x_max + sigma_test],
                ls="--", c="red", label=f"±σ_test={sigma_test:.3f}")
        ax.plot([x_min, x_max], [x_min - sigma_test, x_max - sigma_test],
                ls="--", c="red")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(x_min - 0.5, x_max + 0.5)
        ax.set_title(f"{band_name} — {label} (RMSE={rmse:.3f})")
        ax.set_xlabel("Observed Magnitude")
        ax.set_ylabel("Predicted Magnitude")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Right column: Residual histograms
    xlim = max(3 * sigma_test, 0.5)
    for row, (resid, label) in enumerate([
        (resid_train, "Train"),
        (resid_test, "Test"),
    ]):
        ax = axs[row, 1]
        ax.hist(resid, bins=30, alpha=0.75, color="gray")
        ax.axvline(0.0, color="black", ls="--", label="0 (perfect)")
        ax.axvline(+sigma_test, color="red", ls="--",
                   label=f"±σ_test={sigma_test:.3f}")
        ax.axvline(-sigma_test, color="red", ls="--")
        ax.set_xlim(-xlim, xlim)
        ax.set_title(f"{band_name} — {label} Residuals")
        ax.set_xlabel("Residual (Pred − Obs)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Broadband {band_name}-band Diagnostics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_nn_outlier_analysis(
    model: eqx.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    meta_train: pd.DataFrame,
    X_test: np.ndarray,
    y_test: np.ndarray,
    meta_test: pd.DataFrame,
    *,
    band_idx: int = 0,
    outlier_mag: float = 0.40,
    xlim: Optional[tuple[float, float]] = None,
    figsize: tuple[int, int] = (28, 12),
    save_path: Optional[str] = None,
) -> Figure:
    """
    Create 2×3 diagnostic panel (train/test) for an Equinox MLP.

    This creates a comprehensive visualization showing:
    - Residuals vs. color key (TRANSPARENCY_GFA or PWV_los)
    - Observed vs. predicted magnitudes
    - Residual histograms

    Parameters
    ----------
    model : eqx.Module
        Trained Equinox model.
    X_train : np.ndarray
        Training input features. Shape: (N_train, n_features).
    y_train : np.ndarray
        Training targets. Shape: (N_train, 4).
    meta_train : pd.DataFrame
        Training metadata.
    X_test : np.ndarray
        Test input features. Shape: (N_test, n_features).
    y_test : np.ndarray
        Test targets. Shape: (N_test, 4).
    meta_test : pd.DataFrame
        Test metadata.
    band_idx : int, default 0
        Which band to plot (0=V, 1=g, 2=r, 3=z).
    outlier_mag : float, default 0.40
        Threshold in magnitudes for marking outliers.
    xlim : tuple[float, float] | None, default None
        X-axis limits for residual plots. If None, auto-computed from data.
    figsize : tuple[int, int], default (28, 12)
        Figure size in inches (width, height).
    save_path : str | None, default None
        If provided, save the figure to this path.

    Returns
    -------
    fig : Figure
        Matplotlib figure object.

    Examples
    --------
    >>> from desisky.visualization import plot_nn_outlier_analysis
    >>> from desisky.training import gather_full_data
    >>>
    >>> # Gather data from loaders
    >>> X_train, y_train, _, meta_train = gather_full_data(train_loader)
    >>> X_test, y_test, _, meta_test = gather_full_data(test_loader)
    >>>
    >>> # Plot V-band diagnostics
    >>> fig = plot_nn_outlier_analysis(
    ...     model, X_train, y_train, meta_train,
    ...     X_test, y_test, meta_test,
    ...     band_idx=0, outlier_mag=0.40
    ... )
    >>> plt.show()
    """
    band_names = ["V", "g", "r", "z"]
    band_name = band_names[band_idx]

    # ------------------------------------------------------------------ #
    # 1. Forward pass - extract single band                              #
    # ------------------------------------------------------------------ #
    pred_train_all = np.array(jax.vmap(model)(jnp.array(X_train)))  # (N, 4)
    pred_test_all = np.array(jax.vmap(model)(jnp.array(X_test)))  # (N, 4)

    # Extract the specific band
    pred_train = pred_train_all[:, band_idx]  # (N,)
    pred_test = pred_test_all[:, band_idx]  # (N,)

    y_train_band = y_train[:, band_idx]  # (N,)
    y_test_band = y_test[:, band_idx]  # (N,)

    resid_train = pred_train - y_train_band
    resid_test = pred_test - y_test_band

    outlier_train = np.abs(resid_train) > outlier_mag
    outlier_test = np.abs(resid_test) > outlier_mag

    # Determine color key (prefer PWV_los if available, else TRANSPARENCY_GFA)
    colour_key = "PWV_los" if "PWV_los" in meta_train.columns else "TRANSPARENCY_GFA"
    colour_train = meta_train[colour_key].to_numpy()
    colour_test = meta_test[colour_key].to_numpy()

    # Normalization for color-map
    valid_colour = np.concatenate(
        [
            colour_train[np.isfinite(colour_train)],
            colour_test[np.isfinite(colour_test)],
        ]
    )
    norm = mcolors.Normalize(vmin=valid_colour.min(), vmax=valid_colour.max())
    cmap = cm.viridis

    # X-limits: either supplied or derived from the data
    if xlim is None:
        finite = valid_colour[np.isfinite(valid_colour)]
        vmin, vmax = finite.min(), finite.max()
        pad = (vmax - vmin) * 0.02
        xlim = (vmin - pad, vmax + pad)

    # ------------------------------------------------------------------ #
    # 2. Create figure                                                    #
    # ------------------------------------------------------------------ #
    fig, axs = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(f"{band_name}-band Diagnostics", fontsize=16, y=0.995)

    # Helper to plot one residual panel
    def _plot_residual(ax, x, y, out_mask, title):
        in_mask = (~out_mask) & np.isfinite(x)
        out_mask = out_mask & np.isfinite(x)

        sc_in = ax.scatter(
            x[in_mask], y[in_mask], c=x[in_mask], cmap=cmap, norm=norm, alpha=0.5
        )
        ax.scatter(
            x[out_mask],
            y[out_mask],
            c=x[out_mask],
            cmap=cmap,
            norm=norm,
            alpha=0.5,
            marker="s",
            edgecolor="black",
            label=f"Outliers (n={out_mask.sum()})",
        )

        ax.axhline(
            +outlier_mag, color="red", ls="--", lw=2, label=f"$\\pm$ {outlier_mag}"
        )
        ax.axhline(-outlier_mag, color="red", ls="--", lw=2)
        ax.axhline(0.0, color="black", ls="--", lw=2, label="Zero Residual")
        ax.set_xlim(*xlim)
        ax.set_ylim(-2, 2)
        ax.set_xlabel(colour_key, fontsize=11)
        ax.set_ylabel("Residual (Pred − Obs)", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize="small")
        return sc_in

    sc0 = _plot_residual(
        axs[0, 0],
        colour_train,
        resid_train,
        outlier_train,
        f"Residuals vs. {colour_key} (Train)",
    )
    sc3 = _plot_residual(
        axs[1, 0],
        colour_test,
        resid_test,
        outlier_test,
        f"Residuals vs. {colour_key} (Test)",
    )

    # Observed vs predicted panels
    def _plot_obs_pred(ax, y_obs, y_pred, x_colour, out_mask, title):
        in_mask = (~out_mask) & np.isfinite(x_colour)
        out_mask = out_mask & np.isfinite(x_colour)

        ax.scatter(y_obs[in_mask], y_pred[in_mask], c="green", alpha=0.2)
        ax.scatter(y_obs[out_mask], y_pred[out_mask], c="green", alpha=0.2)

        ax.plot([16, 22], [16, 22], ls="--", c="black", label="1‑to‑1")
        ax.plot(
            [16, 22],
            [16 + outlier_mag, 22 + outlier_mag],
            ls="--",
            c="red",
            label=f"±{outlier_mag} mag",
        )
        ax.plot([16, 22], [16 - outlier_mag, 22 - outlier_mag], ls="--", c="red")

        ax.set_xlim(15.9, 22)
        ax.set_ylim(15.5, 22.5)
        ax.set_xlabel("Observed Magnitudes", fontsize=11)
        ax.set_ylabel("Predicted Magnitudes", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize="small")
        return

    _plot_obs_pred(
        axs[0, 1],
        y_train_band,
        pred_train,
        colour_train,
        outlier_train,
        "Observed vs. Predicted (Train)",
    )
    _plot_obs_pred(
        axs[1, 1],
        y_test_band,
        pred_test,
        colour_test,
        outlier_test,
        "Observed vs. Predicted (Test)",
    )

    # Histograms
    for ax, data, lbl in zip(
        [axs[0, 2], axs[1, 2]], [resid_train, resid_test], ["Train", "Test"]
    ):
        ax.hist(data, bins=100, color="gray", alpha=0.7)
        ax.axvline(0.0, color="black", ls="--", label="Zero Residual")
        ax.axvline(+outlier_mag, color="red", ls="--", label=f"$\\pm$ {outlier_mag}")
        ax.axvline(-outlier_mag, color="red", ls="--")
        ax.set_xlim(-1.25, 1.25)
        ax.set_title(f"Residual Histogram ({lbl})", fontsize=12)
        ax.set_xlabel("Residual (Pred − Obs)", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize="small")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ============================================================================
# UMAP Visualization
# ============================================================================


def plot_latent_umap(
    latents: np.ndarray,
    *,
    discrete_labels: Optional[np.ndarray] = None,
    subset: Optional[Sequence[str]] = None,
    show_unclassified: bool = True,
    unclassified_token: str = "Unclassified",
    continuous_var: Optional[np.ndarray] = None,
    continuous_name: str = "",
    cmap: str = "viridis",
    palette: Optional[dict[str, str]] = None,
    alpha: float = 0.7,
    marker_size: int = 8,
    show_colorbar: bool = True,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    random_state: int = 42,
    max_samples: Optional[int] = 50_000,
    title: str = "UMAP of Latent Space",
    figsize: tuple[int, int] = (10, 8),
) -> Figure:
    """Project latent vectors to 2-D with UMAP and visualize.

    Supports coloring by discrete class labels (e.g. sky condition) or
    by a continuous variable (e.g. MOONFRAC).

    Parameters
    ----------
    latents : np.ndarray
        Latent vectors, shape ``(N, latent_dim)``.
    discrete_labels : np.ndarray | None
        Class labels for each sample, shape ``(N,)``.
        Used for filtering and/or coloring (when *continuous_var* is None).
    subset : Sequence[str] | None
        Keep only these class labels.
    show_unclassified : bool
        Whether to keep rows with label == *unclassified_token*.
    unclassified_token : str
        Label marking unclassified points.
    continuous_var : np.ndarray | None
        Continuous variable for coloring, shape ``(N,)``.
        Takes precedence over *discrete_labels* for coloring.
    continuous_name : str
        Colorbar label when using *continuous_var*.
    cmap : str
        Matplotlib colormap for continuous coloring.
    palette : dict[str, str] | None
        ``{label: color}`` for discrete coloring.
    alpha : float
        Marker transparency.
    marker_size : int
        Marker size.
    show_colorbar : bool
        Show colorbar when using *continuous_var*.
    n_neighbors, min_dist, metric, random_state
        UMAP hyperparameters.
    max_samples : int | None
        Randomly subsample to this many points for speed.
    title : str
        Figure title.
    figsize : tuple[int, int]
        Figure size in inches.

    Returns
    -------
    Figure
        Matplotlib figure with UMAP projection.
    """
    try:
        from umap import UMAP
    except ImportError:
        raise ImportError(
            "umap-learn is required for UMAP visualization. "
            "Install with: pip install umap-learn"
        )

    import itertools

    latents = np.asarray(latents)
    if latents.ndim != 2:
        raise ValueError("`latents` must have shape (N, latent_dim)")

    N = len(latents)
    rng = np.random.default_rng(random_state)

    keep = ~np.isnan(latents).any(axis=1)

    if discrete_labels is not None:
        discrete_labels = np.asarray(discrete_labels, dtype=str)
        if len(discrete_labels) != N:
            raise ValueError("`discrete_labels` length mismatch with `latents`")
        if not show_unclassified:
            keep &= discrete_labels != unclassified_token
        if subset is not None:
            keep &= np.isin(discrete_labels, list(subset))

    if continuous_var is not None:
        continuous_var = np.asarray(continuous_var, dtype=float)
        if len(continuous_var) != N:
            raise ValueError("`continuous_var` length mismatch with `latents`")
        keep &= ~np.isnan(continuous_var)

    idx = np.where(keep)[0]
    if max_samples is not None and idx.size > max_samples:
        idx = rng.choice(idx, size=max_samples, replace=False)

    latents = latents[idx]
    if discrete_labels is not None:
        discrete_labels = discrete_labels[idx]
    if continuous_var is not None:
        continuous_var = continuous_var[idx]

    z2 = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    ).fit_transform(latents)

    fig, ax = plt.subplots(figsize=figsize)

    if continuous_var is not None:
        sc = ax.scatter(
            z2[:, 0], z2[:, 1],
            c=continuous_var, cmap=cmap,
            s=marker_size, alpha=alpha, lw=0,
        )
        if show_colorbar:
            cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
            cbar.set_label(continuous_name, rotation=270, labelpad=15)
    else:
        if palette is None:
            colours = itertools.cycle(plt.get_cmap("tab10").colors)
            unique_labels = np.unique(discrete_labels) if discrete_labels is not None else []
            palette = {lab: next(colours) for lab in unique_labels}

        if discrete_labels is not None:
            for lab in np.unique(discrete_labels):
                mask = discrete_labels == lab
                ax.scatter(
                    z2[mask, 0], z2[mask, 1],
                    s=marker_size, alpha=alpha,
                    color=palette.get(lab, "black"),
                    label=f"{lab} ({mask.sum():,})",
                    lw=0,
                )
            ax.legend(frameon=True, fontsize=9, loc="upper right")
        else:
            ax.scatter(
                z2[:, 0], z2[:, 1],
                s=marker_size, alpha=alpha,
                color="steelblue", lw=0,
            )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("UMAP 1", fontsize=10)
    ax.set_ylabel("UMAP 2", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig
