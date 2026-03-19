# SPDX-FileCopyrightText: 2025-present MatthewDowicz <mjdowicz@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Visualization functions for wandb experiment tracking.

All functions return a :class:`matplotlib.figure.Figure` and are
**wandb-agnostic** — the caller is responsible for logging via
``wandb.Image(fig)``.  This keeps the plotting logic reusable
outside of wandb contexts (e.g., standalone analysis notebooks).

Typical usage inside an ``on_epoch_end`` callback::

    from desisky.visualization.wandb_plots import plot_cdf_comparison
    from desisky.training.wandb_utils import log_figure

    def on_epoch_end(model, ema_model, history, epoch):
        fig, emd = plot_cdf_comparison(real_mags, gen_mags, ["V", "g", "r", "z"])
        log_figure("val/cdf_broadband", fig, epoch)
        plt.close(fig)
"""

from __future__ import annotations
import warnings
from typing import Optional, Sequence

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.gridspec import GridSpec
except ImportError:
    raise ImportError(
        "matplotlib is required for visualization. "
        "Install with: pip install desisky[viz]"
    )

from scipy.stats import wasserstein_distance


# ============================================================================
# VAE Reconstruction Plots
# ============================================================================


def plot_vae_reconstructions(
    originals: np.ndarray,
    reconstructions: np.ndarray,
    wavelength: np.ndarray,
    n_samples: int = 5,
    figsize: tuple[int, int] = (14, 3),
) -> Figure:
    """Plot original vs reconstructed spectra.

    Parameters
    ----------
    originals : np.ndarray
        Original spectra, shape ``(N, n_wavelengths)``.
    reconstructions : np.ndarray
        Reconstructed spectra, shape ``(N, n_wavelengths)``.
    wavelength : np.ndarray
        Wavelength grid, shape ``(n_wavelengths,)``.
    n_samples : int
        Number of spectra to plot (uses the first ``n_samples``).
    figsize : tuple[int, int]
        Figure size per subplot row (width, height).

    Returns
    -------
    Figure
        Matplotlib figure with ``n_samples`` subplots.
    """
    n_samples = min(n_samples, len(originals))
    fig, axes = plt.subplots(n_samples, 1, figsize=(figsize[0], figsize[1] * n_samples))
    if n_samples == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(wavelength, originals[i], label="Original", linewidth=1.5, alpha=0.8)
        ax.plot(
            wavelength, reconstructions[i],
            label="Reconstructed", linewidth=1.5, linestyle="--", alpha=0.8,
        )
        ax.set_yscale("log")
        ax.set_ylabel(
            r"Sky Brightness"
            "\n"
            r"[$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$ arcsec$^{-2}$]",
            fontsize=9,
        )
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=9)
        if i == n_samples - 1:
            ax.set_xlabel(r"Wavelength ($\AA$)", fontsize=10)

    fig.suptitle("VAE Reconstruction", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


# ============================================================================
# Latent Corner Plot
# ============================================================================


def plot_latent_corner(
    latents: np.ndarray,
    labels: Optional[Sequence[str]] = None,
    sky_conditions: Optional[np.ndarray] = None,
    condition_names: Optional[Sequence[str]] = None,
    bins: int = 40,
    figsize: float = 10,
) -> Figure:
    """Corner/pair-plot of VAE latent dimensions.

    Histograms on the diagonal, scatter plots on the lower triangle.
    Optionally colored by sky condition (dark, moon, twilight, etc.).

    Parameters
    ----------
    latents : np.ndarray
        Latent codes, shape ``(N, latent_dim)``.
    labels : Sequence[str] | None
        Names for each latent dimension (e.g. ``["z0", "z1", ...]``).
    sky_conditions : np.ndarray | None
        Categorical labels for each sample, shape ``(N,)``.  For
        example: ``["dark", "moon", "twilight", ...]``.  If provided,
        each category is plotted in a different color.
    condition_names : Sequence[str] | None
        Display names for each category (legend labels).  If None,
        inferred from unique values in ``sky_conditions``.
    bins : int
        Number of histogram bins on the diagonal.
    figsize : float
        Figure size (square), in inches.

    Returns
    -------
    Figure
        Matplotlib corner-plot figure.
    """
    latents = np.asarray(latents)
    N, D = latents.shape

    if labels is None:
        labels = [f"z{i}" for i in range(D)]

    # Color setup
    default_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                      "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    if sky_conditions is not None:
        sky_conditions = np.asarray(sky_conditions)
        categories = condition_names or sorted(set(sky_conditions))
        color_map = {cat: default_colors[i % len(default_colors)]
                     for i, cat in enumerate(categories)}
    else:
        categories = None

    fig, axes = plt.subplots(
        D, D, figsize=(figsize, figsize),
        gridspec_kw={"hspace": 0.05, "wspace": 0.05},
    )

    for i in range(D):
        for j in range(D):
            ax = axes[i, j]

            if i == j:
                # Diagonal: 1D histograms
                if categories is not None:
                    for cat in categories:
                        mask = sky_conditions == cat
                        ax.hist(
                            latents[mask, i], bins=bins, density=True,
                            histtype="stepfilled", alpha=0.45,
                            color=color_map[cat], edgecolor=color_map[cat],
                            linewidth=0.8, label=str(cat),
                        )
                else:
                    ax.hist(latents[:, i], bins=bins, density=True,
                            histtype="stepfilled", alpha=0.6,
                            color="steelblue", edgecolor="steelblue",
                            linewidth=0.8)

            elif i > j:
                # Lower triangle: scatter
                if categories is not None:
                    for cat in categories:
                        mask = sky_conditions == cat
                        ax.scatter(
                            latents[mask, j], latents[mask, i],
                            c=color_map[cat], alpha=0.15, s=3,
                            rasterized=True,
                        )
                else:
                    ax.scatter(
                        latents[:, j], latents[:, i],
                        c="steelblue", alpha=0.15, s=3,
                        rasterized=True,
                    )
            else:
                # Upper triangle: off
                ax.axis("off")

            # Axis label management
            if j == 0 and i > 0:
                ax.set_ylabel(labels[i], fontsize=9)
            else:
                ax.set_yticklabels([])
            if i == D - 1:
                ax.set_xlabel(labels[j], fontsize=9)
            else:
                ax.set_xticklabels([])

            ax.tick_params(labelsize=6)

    # Legend from diagonal histograms
    if categories is not None:
        handles, lbls = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles, lbls, loc="upper right", fontsize=8,
            title="Sky Condition", title_fontsize=9,
        )

    fig.suptitle("Latent Space", fontsize=13, fontweight="bold")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


# ============================================================================
# Latent Corner Comparison (Real vs Generated)
# ============================================================================


def plot_latent_corner_comparison(
    real_latents: np.ndarray,
    gen_latents: np.ndarray,
    labels: Optional[Sequence[str]] = None,
    bins: int = 40,
    figsize: float = 10,
    real_color: str = "black",
    gen_color: str = "tab:green",
) -> Figure:
    """Corner/pair-plot comparing real vs generated latent distributions.

    Overlaid histograms on the diagonal, overlaid scatter plots on the
    lower triangle, with per-dimension Wasserstein-1 (EMD) annotated.

    Parameters
    ----------
    real_latents : np.ndarray
        Real latent codes, shape ``(N_real, latent_dim)``.
    gen_latents : np.ndarray
        Generated latent codes, shape ``(N_gen, latent_dim)``.
    labels : Sequence[str] | None
        Names for each latent dimension (e.g. ``["z0", "z1", ...]``).
    bins : int
        Number of histogram bins on the diagonal.
    figsize : float
        Figure size (square), in inches.
    real_color : str
        Color for the real data.
    gen_color : str
        Color for the generated data.

    Returns
    -------
    Figure
        Matplotlib corner-plot figure.
    """
    real_latents = np.asarray(real_latents)
    gen_latents = np.asarray(gen_latents)
    D = real_latents.shape[1]

    if labels is None:
        labels = [f"z{i}" for i in range(D)]

    fig, axes = plt.subplots(
        D, D, figsize=(figsize, figsize),
        gridspec_kw={"hspace": 0.05, "wspace": 0.05},
    )

    for i in range(D):
        for j in range(D):
            ax = axes[i, j]

            if i == j:
                # Diagonal: overlaid 1D histograms
                ax.hist(
                    real_latents[:, i], bins=bins, density=True,
                    alpha=0.5, color=real_color, label="Real",
                )
                ax.hist(
                    gen_latents[:, i], bins=bins, density=True,
                    alpha=0.5, color=gen_color, label="Generated",
                )

                # Per-dimension EMD
                emd = wasserstein_distance(real_latents[:, i], gen_latents[:, i])
                ax.text(
                    0.5, 1.05, f"EMD={emd:.3f}", transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=7,
                    bbox=dict(boxstyle="round", facecolor="white",
                              alpha=0.8, edgecolor="gray", linewidth=0.5),
                )

            elif i > j:
                # Lower triangle: overlaid scatter
                ax.scatter(
                    real_latents[:, j], real_latents[:, i],
                    c=real_color, alpha=0.15, s=3, rasterized=True,
                )
                ax.scatter(
                    gen_latents[:, j], gen_latents[:, i],
                    c=gen_color, alpha=0.15, s=3, rasterized=True,
                )
            else:
                # Upper triangle: off
                ax.axis("off")

            # Axis label management
            if j == 0 and i > 0:
                ax.set_ylabel(labels[i], fontsize=9)
            else:
                ax.set_yticklabels([])
            if i == D - 1:
                ax.set_xlabel(labels[j], fontsize=9)
            else:
                ax.set_xticklabels([])

            ax.tick_params(labelsize=6)

    # Legend from diagonal histograms
    handles, lbls = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, lbls, loc="upper right", fontsize=8,
        title="Distribution", title_fontsize=9,
    )

    fig.suptitle("Latent Space Comparison", fontsize=13, fontweight="bold")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


# ============================================================================
# CDF Comparison with EMD
# ============================================================================


def plot_cdf_comparison(
    real: np.ndarray,
    generated: np.ndarray,
    feature_names: Sequence[str],
    n_bins: int = 25,
    figsize_per_feature: tuple[float, float] = (14, 6),
) -> tuple[Figure, dict[str, float]]:
    """CDF + histogram comparison between real and generated features.

    For each feature, produces a two-panel figure:

    - **Left**: overlaid histograms (real vs generated).
    - **Right**: empirical CDFs with a difference (Delta CDF) subplot
      and Wasserstein-1 distance (EMD) annotated.

    Parameters
    ----------
    real : np.ndarray
        Real feature values, shape ``(N_real, n_features)``.
    generated : np.ndarray
        Generated feature values, shape ``(N_gen, n_features)``.
    feature_names : Sequence[str]
        Display names for each feature column.
    n_bins : int
        Number of histogram bins.
    figsize_per_feature : tuple[float, float]
        Figure size for each per-feature panel.

    Returns
    -------
    fig : Figure
        Multi-row figure (one row per feature).
    emd_dict : dict[str, float]
        Feature-name -> Wasserstein-1 distance.
    """
    real = np.asarray(real)
    generated = np.asarray(generated)
    n_features = len(feature_names)

    if real.ndim == 1:
        real = real[:, None]
    if generated.ndim == 1:
        generated = generated[:, None]

    emd_dict: dict[str, float] = {}
    w, h = figsize_per_feature
    fig = plt.figure(figsize=(w, h * n_features))

    for idx, name in enumerate(feature_names):
        real_col = real[:, idx]
        gen_col = generated[:, idx]

        # Clean NaN/Inf
        real_clean = real_col[np.isfinite(real_col)]
        gen_clean = gen_col[np.isfinite(gen_col)]

        # EMD
        emd = wasserstein_distance(real_clean, gen_clean)
        emd_dict[name] = float(emd)

        # GridSpec for this row
        gs = GridSpec(
            2, 2, figure=fig,
            left=0.06, right=0.96,
            top=1 - idx / n_features - 0.07,
            bottom=1 - (idx + 1) / n_features + 0.04,
            width_ratios=[1, 1],
            height_ratios=[3, 1],
            hspace=0.05, wspace=0.35,
        )

        # --- Left panel: histogram ---
        ax_hist = fig.add_subplot(gs[:, 0])
        # Shared bin edges so real & generated are comparable
        combined = np.concatenate([real_clean, gen_clean])
        shared_edges = np.linspace(combined.min(), combined.max(), n_bins + 1)
        ax_hist.hist(
            real_clean, bins=shared_edges, density=True, alpha=0.5,
            color="black", label="Real", histtype="stepfilled",
        )
        ax_hist.hist(
            gen_clean, bins=shared_edges, density=True, alpha=0.6,
            color="tab:green", label="Generated",
        )
        ax_hist.set_xlabel(name, fontsize=10)
        ax_hist.set_ylabel("Density", fontsize=10)
        ax_hist.set_title(f"{name}  (EMD = {emd:.4f})", fontsize=11, fontweight="bold")
        ax_hist.legend(fontsize=9)
        ax_hist.grid(alpha=0.3)

        # --- Right panel: CDF ---
        real_sorted = np.sort(real_clean)
        gen_sorted = np.sort(gen_clean)
        real_cdf = np.arange(1, len(real_sorted) + 1) / len(real_sorted)
        gen_cdf = np.arange(1, len(gen_sorted) + 1) / len(gen_sorted)

        ax_cdf = fig.add_subplot(gs[0, 1])
        ax_cdf.plot(real_sorted, real_cdf, "k-", linewidth=2, label="Real", alpha=0.8)
        ax_cdf.plot(
            gen_sorted, gen_cdf, color="tab:green",
            linewidth=2, label="Generated", alpha=0.8,
        )
        ax_cdf.set_ylabel("Cumulative Probability", fontsize=9)
        ax_cdf.legend(loc="lower right", fontsize=8)
        ax_cdf.grid(alpha=0.3)
        ax_cdf.tick_params(labelbottom=False)
        ax_cdf.set_title("CDF Comparison", fontsize=10, fontweight="bold")

        # --- Difference subplot ---
        ax_diff = fig.add_subplot(gs[1, 1], sharex=ax_cdf)
        lo = max(real_sorted.min(), gen_sorted.min())
        hi = min(real_sorted.max(), gen_sorted.max())
        common = np.linspace(lo, hi, 500)
        real_interp = np.interp(common, real_sorted, real_cdf)
        gen_interp = np.interp(common, gen_sorted, gen_cdf)
        diff = gen_interp - real_interp

        ax_diff.plot(common, diff, "r-", linewidth=1.5)
        ax_diff.axhline(0, color="k", linestyle="--", linewidth=1, alpha=0.5)
        ax_diff.fill_between(common, 0, diff, alpha=0.3, color="red")
        ax_diff.set_xlabel(name, fontsize=9)
        ax_diff.set_ylabel(r"$\Delta$ CDF", fontsize=9)
        ax_diff.grid(alpha=0.3)

    return fig, emd_dict


# ============================================================================
# Conditional Validation Grid
# ============================================================================


def plot_conditional_validation_grid(
    real_features: np.ndarray,
    gen_features: np.ndarray,
    cond_values: np.ndarray,
    cond_name: str,
    feature_names: Sequence[str],
    n_bins: int = 10,
    ncols: int = 4,
    figsize: tuple[int, int] = (18, 10),
) -> Figure:
    """Validation grid: feature statistics vs a conditioning variable.

    For each feature, plots the binned mean and 16th-84th percentile
    bands as a function of a conditioning variable, comparing real
    data to generated data.

    Parameters
    ----------
    real_features : np.ndarray
        Real feature values, shape ``(N_real, n_features)``.
    gen_features : np.ndarray
        Generated feature values, shape ``(N_gen, n_features)``.
    cond_values : np.ndarray
        Conditioning variable values for real data, shape ``(N_real,)``.
        Generated data is assumed to share the same conditioning
        distribution (i.e. generated from the same conditioning inputs).
    cond_name : str
        Display name of the conditioning variable (x-axis label).
    feature_names : Sequence[str]
        Display names for each feature column.
    n_bins : int
        Number of quantile-based bins for the conditioning variable.
    ncols : int
        Number of columns in the subplot grid.
    figsize : tuple[int, int]
        Overall figure size.

    Returns
    -------
    Figure
        Matplotlib figure with one subplot per feature.
    """
    real_features = np.asarray(real_features)
    gen_features = np.asarray(gen_features)
    cond_values = np.asarray(cond_values, dtype=float)
    n_features = len(feature_names)

    # Adaptive bin count: need at least ~3 samples per bin for stable stats
    min_per_bin = 3
    n_bins = max(2, min(n_bins, len(cond_values) // min_per_bin))

    # Quantile-based bins on conditioning variable
    bin_edges = np.quantile(
        cond_values[np.isfinite(cond_values)],
        np.linspace(0, 1, n_bins + 1),
    )
    bin_indices = np.digitize(cond_values, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    def _bin_stats(features, bin_idx):
        """Compute per-bin mean, q16, q84, bin_center."""
        centers, means, q16s, q84s = [], [], [], []
        for b in range(n_bins):
            mask = bin_idx == b
            if mask.sum() < min_per_bin:
                continue
            centers.append(cond_values[mask].mean())
            vals = features[mask]
            means.append(np.nanmean(vals, axis=0))
            q16s.append(np.nanpercentile(vals, 16, axis=0))
            q84s.append(np.nanpercentile(vals, 84, axis=0))
        if len(centers) == 0:
            return (
                np.empty(0),
                np.empty((0, features.shape[1])),
                np.empty((0, features.shape[1])),
                np.empty((0, features.shape[1])),
            )
        return (
            np.array(centers),
            np.array(means),
            np.array(q16s),
            np.array(q84s),
        )

    real_centers, real_means, real_q16, real_q84 = _bin_stats(
        real_features, bin_indices,
    )

    # For generated data, use the same binning (same conditioning inputs)
    gen_bin_indices = bin_indices
    if len(gen_features) != len(cond_values):
        # If generated data has different length, bin on same edges
        gen_bin_indices = np.clip(
            np.digitize(cond_values[:len(gen_features)], bin_edges) - 1,
            0, n_bins - 1,
        )
    gen_centers, gen_means, gen_q16, gen_q84 = _bin_stats(
        gen_features, gen_bin_indices,
    )

    # Plot
    nrows = (n_features + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()

    for idx, name in enumerate(feature_names):
        ax = axes_flat[idx]

        # Real
        if len(real_centers) > 0:
            ax.plot(
                real_centers, real_means[:, idx],
                "ko-", linewidth=2, markersize=5, label="Real", alpha=0.8,
            )
            ax.fill_between(
                real_centers, real_q16[:, idx], real_q84[:, idx],
                color="black", alpha=0.12, label="Real 16-84%",
            )

        # Generated
        if len(gen_centers) > 0:
            ax.plot(
                gen_centers, gen_means[:, idx],
                "s--", color="tab:green", linewidth=2, markersize=4,
                label="Generated", alpha=0.8,
            )
            ax.fill_between(
                gen_centers, gen_q16[:, idx], gen_q84[:, idx],
                color="tab:green", alpha=0.12, label="Gen 16-84%",
            )

        ax.set_xlabel(cond_name, fontsize=9)
        ax.set_ylabel(name, fontsize=9)
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.3)

    # Hide unused subplots
    for idx in range(n_features, len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle(
        f"Conditional Validation: Features vs. {cond_name}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


# ============================================================================
# Convenience CDF Visualizations (data extraction + plotting)
# ============================================================================


def plot_broadband_cdfs(
    wavelength: np.ndarray,
    real_spectra: np.ndarray,
    gen_spectra: np.ndarray,
    n_bins: int = 25,
) -> dict[str, tuple[Figure, float]]:
    """Compute broadband magnitudes and produce one CDF figure per filter.

    This is a convenience wrapper that calls
    :func:`~desisky.data.compute_broadband_mags` and then
    :func:`plot_cdf_comparison` for each filter individually.

    Parameters
    ----------
    wavelength : np.ndarray
        Wavelength grid in Angstrom, shape ``(n_wavelengths,)``.
    real_spectra : np.ndarray
        Real flux array, shape ``(N_real, n_wavelengths)``.
    gen_spectra : np.ndarray
        Generated/reconstructed flux array, shape ``(N_gen, n_wavelengths)``.
    n_bins : int
        Number of histogram bins.

    Returns
    -------
    dict[str, tuple[Figure, float]]
        ``{band_name: (figure, emd_value)}`` for each of V, g, r, z.
        Caller is responsible for ``plt.close(fig)`` after logging.
    """
    from desisky.data import compute_broadband_mags, BROADBAND_NAMES

    real_mags = compute_broadband_mags(wavelength, real_spectra)
    gen_mags = compute_broadband_mags(wavelength, gen_spectra)

    results: dict[str, tuple[Figure, float]] = {}
    for j, band_name in enumerate(BROADBAND_NAMES):
        fig, emd_dict = plot_cdf_comparison(
            real_mags[:, j : j + 1],
            gen_mags[:, j : j + 1],
            [band_name],
            n_bins=n_bins,
        )
        results[band_name] = (fig, emd_dict[band_name])

    return results


def plot_airglow_cdfs(
    wavelength: np.ndarray,
    real_spectra: np.ndarray,
    gen_spectra: np.ndarray,
    line_names: list[str] | None = None,
    n_bins: int = 25,
) -> dict[str, tuple[Figure, float]]:
    """Measure airglow intensities and produce one CDF figure per line.

    This is a convenience wrapper that calls
    :func:`~desisky.data.measure_airglow_intensities` and then
    :func:`plot_cdf_comparison` for each emission line individually.

    Parameters
    ----------
    wavelength : np.ndarray
        Wavelength grid in Angstrom, shape ``(n_wavelengths,)``.
    real_spectra : np.ndarray
        Real flux array, shape ``(N_real, n_wavelengths)``.
    gen_spectra : np.ndarray
        Generated/reconstructed flux array, shape ``(N_gen, n_wavelengths)``.
    line_names : list[str] | None
        Airglow lines to plot.  Defaults to :data:`~desisky.data.AIRGLOW_CDF_NAMES`.
    n_bins : int
        Number of histogram bins.

    Returns
    -------
    dict[str, tuple[Figure, float]]
        ``{line_name: (figure, emd_value)}`` for each requested line.
        Caller is responsible for ``plt.close(fig)`` after logging.
    """
    from desisky.data import measure_airglow_intensities, AIRGLOW_CDF_NAMES

    if line_names is None:
        line_names = list(AIRGLOW_CDF_NAMES)

    real_ag = measure_airglow_intensities(wavelength, real_spectra)
    gen_ag = measure_airglow_intensities(wavelength, gen_spectra)

    results: dict[str, tuple[Figure, float]] = {}
    for line_name in line_names:
        if line_name not in real_ag.columns:
            continue
        fig, emd_dict = plot_cdf_comparison(
            real_ag[[line_name]].to_numpy(),
            gen_ag[[line_name]].to_numpy(),
            [line_name],
            n_bins=n_bins,
        )
        results[line_name] = (fig, emd_dict[line_name])

    return results
