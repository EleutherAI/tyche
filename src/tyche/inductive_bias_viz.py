import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Literal, List, Union, Dict, Tuple, Optional, Any
import matplotlib.colors as mcolors
import math


def format_number(value: float) -> str:
    """Format numbers for display in heatmap cells."""
    if np.isnan(value):
        return ""

    abs_value = abs(value)

    if abs_value == 0:
        return "0"
    elif abs_value < 0.001 or abs_value >= 10000:
        return f"{value:.1e}"
    elif abs_value < 0.01:
        return f"{value:.4f}"
    elif abs_value < 0.1:
        return f"{value:.3f}"
    elif abs_value < 10:
        return f"{value:.2f}"
    else:
        return f"{value:.1f}"


def safe_array_stat(arr: Any, stat_type: str = "mean") -> float:
    """Compute statistics safely handling None values."""
    if arr is None:
        return np.nan
    try:
        if stat_type == "mean":
            return np.mean(arr)
        else:  # variance
            return np.var(arr)
    except:
        return np.nan


def prepare_data(
    df: pd.DataFrame,
    weight_mode: str,
    activation: str,
    intermediate: str,
    volume_stat: str,
    stat_type: str,
) -> pd.Series:
    """Prepare and aggregate data for a specific activation-intermediate pair."""
    # Filter by weight_mode, activation, and intermediate
    df_filtered = df[
        (df["weight_mode"] == weight_mode)
        & (df["activation"] == activation)
        & (df["intermediate"] == intermediate)
    ].copy()

    # Process volume estimates
    df_filtered.loc[:, "vol_stat"] = df_filtered["volume_estimates"].apply(
        lambda x: safe_array_stat(x, volume_stat)
    )

    # Drop rows with NaN values
    df_filtered = df_filtered.dropna(subset=["vol_stat"])

    # If no valid data remains after filtering, return empty Series
    if len(df_filtered) == 0:
        return pd.Series()

    # Group by grid dimensions
    grouped = df_filtered.groupby(["num_additional_layers", "W_amplitude"])["vol_stat"]

    # Calculate the specified statistic
    if stat_type == "mean":
        return grouped.mean()
    elif stat_type == "variance":
        return grouped.var()
    elif stat_type == "variance/mean":
        mean_data = grouped.mean()
        var_data = grouped.var()
        # Avoid division by zero
        valid_indices = mean_data != 0
        agg_data = pd.Series(np.nan, index=mean_data.index)
        agg_data[valid_indices] = np.abs(
            var_data[valid_indices] / mean_data[valid_indices]
        )
        return agg_data

    return pd.Series()


def create_heatmap_data(
    agg_data: pd.Series, all_x_values: List, all_y_values: List
) -> np.ndarray:
    """Convert aggregated data to a 2D array for heatmap visualization."""
    heatmap_data = np.full((len(all_y_values), len(all_x_values)), np.nan)

    # Fill the array with the aggregated values
    for (layer, amp), value in agg_data.items():
        if layer in all_x_values and amp in all_y_values:
            x_idx = all_x_values.index(layer)
            y_idx = all_y_values.index(amp)
            heatmap_data[y_idx, x_idx] = value

    return heatmap_data


def draw_single_heatmap(
    ax: plt.Axes,
    heatmap_data: np.ndarray,
    activation: str,
    intermediate: str,
    all_x_values: List,
    all_y_values: List,
    y_labels: List[str],
    vmin: float,
    vmax: float,
    cmap: str,
) -> plt.Artist:
    """Draw a single heatmap on the provided axes."""
    # Create a masked array to handle NaN values
    masked_data = np.ma.masked_invalid(heatmap_data)

    # Create the heatmap
    im = ax.imshow(
        masked_data, cmap=cmap, aspect="auto", origin="upper", vmin=vmin, vmax=vmax
    )

    # Add value annotations
    for y_idx, y_val in enumerate(all_y_values):
        for x_idx, x_val in enumerate(all_x_values):
            value = heatmap_data[y_idx, x_idx]
            if not np.isnan(value):
                formatted_value = format_number(value)
                text_color = "white" if value > (vmin + vmax) / 2 else "black"
                ax.text(
                    x_idx,
                    y_idx,
                    formatted_value,
                    ha="center",
                    va="center",
                    color=text_color,
                )

    # Set title and labels
    ax.set_title(f"Activation: {activation}, Intermediate: {intermediate}")
    ax.set_xlabel("num_additional_layers")
    ax.set_ylabel("W_amplitude")

    # Set ticks
    ax.set_xticks(range(len(all_x_values)))
    ax.set_yticks(range(len(all_y_values)))
    ax.set_xticklabels(all_x_values)
    ax.set_yticklabels(y_labels)

    return im


def display_no_data_message(
    ax: plt.Axes,
    activation: str,
    intermediate: str,
    all_x_values: List,
    all_y_values: List,
    y_labels: List[str],
    vmin: float,
    vmax: float,
    cmap: str,
) -> plt.Artist:
    """Display a message when no data is available for an activation-intermediate pair."""
    # Create a blank grid with the same structure
    heatmap_data = np.full((len(all_y_values), len(all_x_values)), np.nan)
    masked_data = np.ma.masked_invalid(heatmap_data)

    im = ax.imshow(
        masked_data, cmap=cmap, aspect="auto", origin="upper", vmin=vmin, vmax=vmax
    )

    # Add message
    ax.text(
        0.5,
        0.5,
        f"No valid data for\nactivation: {activation}\nintermediate: {intermediate}",
        ha="center",
        va="center",
        transform=ax.transAxes,
    )

    # Set title, labels, and ticks
    ax.set_title(f"Activation: {activation}, Intermediate: {intermediate}")
    ax.set_xlabel("num_additional_layers")
    ax.set_ylabel("W_amplitude")
    ax.set_xticks(range(len(all_x_values)))
    ax.set_yticks(range(len(all_y_values)))
    ax.set_xticklabels(all_x_values)
    ax.set_yticklabels(y_labels)

    return im


def generate_heatmaps(
    df: pd.DataFrame,
    weight_mode: str,
    stat_type: Literal["mean", "variance", "variance/mean"] = "mean",
    volume_stat: Literal["mean", "variance"] = "mean",
    figsize: Tuple[int, int] = (18, 12),
    cmap: str = "viridis",
    min_data_points: int = 3,  # Minimum number of data points required for a pair to be valid
) -> Figure:
    """
    Generate a single heatmap figure with meaningful activation-intermediate pairs.

    Args:
        df: DataFrame with required columns: activation, intermediate, num_additional_layers,
            W_amplitude, weight_mode, sample_id, and volume_estimates (which may contain None values)
        weight_mode: The weight_mode to filter the data by
        stat_type: The statistic to compute across sample_ids ('mean', 'variance', or 'variance/mean')
        volume_stat: The statistic to compute for volume_estimates ('mean' or 'variance')
        figsize: Figure size as (width, height)
        cmap: Colormap for the heatmaps
        min_data_points: Minimum number of data points required for a pair to be considered valid

    Returns:
        A matplotlib figure containing heatmaps for valid activation-intermediate pairs
    """
    # Filter data by weight_mode
    df_filtered = df[df["weight_mode"] == weight_mode].copy()

    # Get unique values for creating pairs
    preferred_activation_order = ["ReLU", "GELU", "Tanh", "GaussianActivation"]
    unique_activations = df_filtered["activation"].unique()
    activations = [
        act for act in preferred_activation_order if act in unique_activations
    ]
    if not activations:  # In case none of the preferred activations are present
        activations = sorted(unique_activations)

    # Get unique intermediates
    unique_intermediates = sorted(df_filtered["intermediate"].unique())

    # Get all possible x and y values for consistent grid
    all_x_values = sorted(df_filtered["num_additional_layers"].unique())
    all_y_values = sorted(df_filtered["W_amplitude"].unique())

    # Format W_amplitude labels
    y_labels = [format_number(val) for val in all_y_values]

    # Filter to keep only valid pairs with enough data
    valid_pairs = []
    all_data = []

    # Check each pair for valid data
    for activation in activations:
        for intermediate in unique_intermediates:
            # Get aggregated data for this pair
            agg_data = prepare_data(
                df, weight_mode, activation, intermediate, volume_stat, stat_type
            )

            # Only include pairs with sufficient data points
            if len(agg_data) >= min_data_points:
                valid_pairs.append((activation, intermediate))
                all_data.extend(agg_data.values)

    # If no valid pairs found, return a figure with a message
    if not valid_pairs:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(
            0.5,
            0.5,
            f"No valid activation-intermediate pairs found for weight_mode: {weight_mode}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
        )
        ax.axis("off")
        return fig

    # Determine global min/max for consistent color scaling
    if all_data:
        vmin = np.nanmin(all_data)
        vmax = np.nanmax(all_data)
    else:
        vmin, vmax = 0, 1

    # Calculate optimal grid layout
    n_pairs = len(valid_pairs)
    n_cols = min(3, n_pairs)  # Use at most 3 columns
    n_rows = math.ceil(n_pairs / n_cols)

    # Create figure with appropriate grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    # Create heatmaps for each valid pair
    last_im = None
    for i, (activation, intermediate) in enumerate(valid_pairs):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]

        # Get aggregated data for this activation-intermediate pair
        agg_data = prepare_data(
            df, weight_mode, activation, intermediate, volume_stat, stat_type
        )

        # Convert data to format suitable for heatmap
        heatmap_data = create_heatmap_data(agg_data, all_x_values, all_y_values)

        # Draw the heatmap
        last_im = draw_single_heatmap(
            ax,
            heatmap_data,
            activation,
            intermediate,
            all_x_values,
            all_y_values,
            y_labels,
            vmin,
            vmax,
            cmap,
        )

    # Remove any unused subplots
    for i in range(len(valid_pairs), n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        if row < len(axes) and col < len(axes[0]):
            fig.delaxes(axes[row, col])

    # Add colorbar
    if last_im:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        fig.colorbar(last_im, cax=cbar_ax)

    # Add overall title
    volume_stat_name = "Mean" if volume_stat == "mean" else "Variance"
    stat_name = {
        "mean": "Mean",
        "variance": "Variance",
        "variance/mean": "Variance / Mean",
    }[stat_type]

    fig.suptitle(
        f"{volume_stat_name} Volume Estimates - {stat_name} across sample_ids\n"
        f"Weight Mode: {weight_mode} (Valid pairs: {len(valid_pairs)})",
        fontsize=16,
    )

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])  # Make room for colorbar and title

    return fig
