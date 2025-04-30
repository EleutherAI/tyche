import itertools
import os
import einops
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Literal, List, Union, Dict, Tuple, Optional, Any
import matplotlib.colors as mcolors
import math
import matplotlib.pyplot as plt
import numpy as np
import base64
import io
import json
import torch as t
import plotly.graph_objects as go
from tqdm import tqdm


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
    value_column: str,
    value_stat: str,
    stat_type: str,
) -> pd.Series:
    """Prepare and aggregate data for a specific activation-intermediate pair."""
    # Filter by weight_mode, activation, and intermediate
    df_filtered = df[
        (df["weight_mode"] == weight_mode)
        & (df["activation"] == activation)
        & (df["intermediate"] == intermediate)
    ].copy()

    # Check if the column exists
    if value_column not in df_filtered.columns:
        return pd.Series()

    # Special handling for empty dataframe
    if len(df_filtered) == 0:
        return pd.Series()

    # Process the specified column
    # Special handling for volume_estimates column
    if value_column == "volume_estimates":
        # Use the original safe_array_stat directly
        df_filtered.loc[:, "val_stat"] = df_filtered[value_column].apply(
            lambda x: safe_array_stat(x, value_stat)
        )
    else:
        # For other columns, check if they contain arrays or scalar values
        sample_value = df_filtered[value_column].iloc[0]
        if isinstance(sample_value, (list, np.ndarray)):
            # The column contains arrays, apply safe_array_stat
            df_filtered.loc[:, "val_stat"] = df_filtered[value_column].apply(
                lambda x: safe_array_stat(x, value_stat)
            )
        else:
            # The column contains scalar values, no need for array processing
            df_filtered.loc[:, "val_stat"] = df_filtered[value_column]

    # Drop rows with NaN values
    df_filtered = df_filtered.dropna(subset=["val_stat"])

    # If no valid data remains after filtering, return empty Series
    if len(df_filtered) == 0:
        return pd.Series()

    # Group by grid dimensions
    grouped = df_filtered.groupby(["num_additional_layers", "W_amplitude"])["val_stat"]

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
    value_column: str = "volume_estimates",  # New parameter to specify the column
    value_stat: Literal["mean", "variance"] = "mean",  # Renamed from volume_stat
    stat_type: Literal["mean", "variance", "variance/mean"] = "mean",
    figsize: Tuple[int, int] = (18, 12),
    cmap: str = "viridis",
    min_data_points: int = 3,  # Minimum number of data points required for a pair to be valid
    title: Optional[str] = None,
) -> Figure:
    """
    Generate a single heatmap figure with meaningful activation-intermediate pairs.

    Args:
        df: DataFrame with required columns: activation, intermediate, num_additional_layers,
            W_amplitude, weight_mode, sample_id, and the column specified by value_column
        weight_mode: The weight_mode to filter the data by
        value_column: The column to use for plotting (default: 'volume_estimates')
        value_stat: For array-like columns, the statistic to compute ('mean' or 'variance')
        stat_type: The statistic to compute across sample_ids ('mean', 'variance', or 'variance/mean')
        figsize: Figure size as (width, height)
        cmap: Colormap for the heatmaps
        min_data_points: Minimum number of data points required for a pair to be considered valid
        title: Optional custom title for the figure

    Returns:
        A matplotlib figure containing heatmaps for valid activation-intermediate pairs
    """
    # Check if the specified column exists
    if value_column not in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(
            0.5,
            0.5,
            f"Column '{value_column}' not found in the DataFrame",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
        )
        ax.axis("off")
        return fig

    # Filter data by weight_mode
    df_filtered = df[df["weight_mode"] == weight_mode].copy()

    # Get unique values for creating pairs
    preferred_activation_order = [
        "ReLU",
        "GELU",
        "Tanh",
        "GaussianActivation",
        "Sigmoid",
    ]
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
                df,
                weight_mode,
                activation,
                intermediate,
                value_column,
                value_stat,
                stat_type,
            )

            # Only include pairs with sufficient data points
            if len(agg_data) >= min_data_points:
                valid_pairs.append((activation, intermediate))
                # Safe way to collect values for min/max calculation
                all_data.extend(
                    [float(val) for val in agg_data.values if not np.isnan(val)]
                )

    # If no valid pairs found, return a figure with a message
    if not valid_pairs:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(
            0.5,
            0.5,
            f"No valid activation-intermediate pairs found for weight_mode: {weight_mode} using column: {value_column}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
        )
        ax.axis("off")
        return fig

    # Determine global min/max for consistent color scaling
    if all_data:
        vmin = min(all_data)
        vmax = max(all_data)
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
            df,
            weight_mode,
            activation,
            intermediate,
            value_column,
            value_stat,
            stat_type,
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
    if title is None:
        # Get descriptive column name
        column_display_name = value_column.replace("_", " ").title()

        # Describe the type of analysis being performed
        value_stat_name = "Mean" if value_stat == "mean" else "Variance"
        stat_name = {
            "mean": "Mean",
            "variance": "Variance",
            "variance/mean": "Variance / Mean",
        }[stat_type]

        # Create a descriptive title
        if value_column == "volume_estimates" or (
            len(df) > 0 and isinstance(df[value_column].iloc[0], (list, np.ndarray))
        ):
            fig.suptitle(
                f"{value_stat_name} of {column_display_name} - {stat_name} across sample_ids\n"
                f"Weight Mode: {weight_mode} (Valid pairs: {len(valid_pairs)})",
                fontsize=16,
            )
        else:
            fig.suptitle(
                f"{column_display_name} - {stat_name} across sample_ids\n"
                f"Weight Mode: {weight_mode} (Valid pairs: {len(valid_pairs)})",
                fontsize=16,
            )
    else:
        fig.suptitle(
            title,
            fontsize=16,
        )

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])  # Make room for colorbar and title

    return fig


def export_figure_dict_to_html(
    figure_dict, epoch_list, layout="vertical", output_file="multi_heatmaps.html"
):
    """
    Export a dictionary of figure sets to a synchronized HTML viewer

    Parameters:
    -----------
    figure_dict : Dictionary mapping titles to lists of matplotlib Figure objects
        Each key is a title, each value is a list of figures for different epochs
    epoch_list : List of corresponding epoch values
    layout : 'horizontal', 'vertical', or 'grid' - how to arrange the sets
    output_file : Filename for the HTML output
    """
    # Convert dict to ordered lists
    set_titles = list(figure_dict.keys())
    figure_sets = list(figure_dict.values())

    # Ensure all sets have the same number of figures
    num_epochs = len(epoch_list)
    for title, figure_set in figure_dict.items():
        if len(figure_set) != num_epochs:
            raise ValueError(
                f"Figure set '{title}' has {len(figure_set)} figures, but {num_epochs} epochs were provided"
            )

    # Convert all figure sets to base64 images
    encoded_figure_sets = []
    for figure_set in figure_sets:
        encoded_figures = []
        for fig in figure_set:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
            buf.seek(0)
            img_data = base64.b64encode(buf.read()).decode("utf-8")
            encoded_figures.append(img_data)
        encoded_figure_sets.append(encoded_figures)

    # Use JSON to properly format JavaScript arrays
    js_figure_sets = json.dumps(encoded_figure_sets)
    js_epochs = json.dumps(epoch_list)
    js_set_titles = json.dumps(set_titles)

    # Determine CSS for layout
    if layout == "horizontal":
        layout_css = """
            .figures-container {
                display: flex;
                flex-direction: row;
                flex-wrap: wrap;
                justify-content: center;
                gap: 20px;
            }
            .figure-set {
                flex: 1;
                min-width: 300px;
                max-width: 500px;
            }
        """
    elif layout == "vertical":
        layout_css = """
            .figures-container {
                display: flex;
                flex-direction: column;
                gap: 30px;
            }
            .figure-set {
                width: 100%;
            }
        """
    else:  # grid
        layout_css = """
            .figures-container {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
                gap: 20px;
            }
            .figure-set {
                width: 100%;
            }
        """

    # Create the HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Multi-Figure Set Synchronized Viewer</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            .container {{
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .controls {{
                display: flex;
                align-items: center;
                margin-bottom: 20px;
                gap: 15px;
            }}
            .slider-container {{
                flex-grow: 1;
            }}
            .image-container {{
                text-align: center;
                margin-bottom: 15px;
            }}
            .image-container img {{
                max-width: 100%;
                height: auto;
                border: 1px solid #eee;
                border-radius: 4px;
            }}
            button {{
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 14px;
            }}
            button:hover {{
                background-color: #45a049;
            }}
            .button-play {{
                min-width: 80px;
            }}
            .figure-set {{
                border: 1px solid #ddd;
                border-radius: 6px;
                padding: 15px;
                background-color: #f9f9f9;
            }}
            .figure-title {{
                font-weight: bold;
                font-size: 16px;
                margin-bottom: 10px;
                text-align: center;
            }}
            h2 {{
                text-align: center;
                margin-bottom: 20px;
            }}
            .controls-panel {{
                background-color: #f0f0f0;
                padding: 15px;
                border-radius: 6px;
                margin-bottom: 25px;
            }}
            {layout_css}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Multi-Figure Synchronized Viewer</h2>
            
            <div class="controls-panel">
                <div class="controls">
                    <button id="prev-btn" onclick="prevEpoch()">Previous</button>
                    <div class="slider-container">
                        <input type="range" id="epoch-slider" min="0" max="{num_epochs-1}" value="0" style="width: 100%">
                    </div>
                    <button id="next-btn" onclick="nextEpoch()">Next</button>
                    <button id="play-btn" class="button-play" onclick="togglePlay()">Play</button>
                </div>
                
                <div style="margin-top: 10px; display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong>Epoch: <span id="epoch-value">{epoch_list[0]}</span></strong>
                        <span style="margin-left: 10px;">(<span id="current-index">1</span>/{num_epochs})</span>
                    </div>
                    
                    <div>
                        <label for="speed">Animation Speed: </label>
                        <select id="speed">
                            <option value="3000">Slow</option>
                            <option value="1500" selected>Normal</option>
                            <option value="750">Fast</option>
                        </select>
                    </div>
                </div>
            </div>
            
            <div class="figures-container" id="figures-container">
                <!-- Figure sets will be inserted here by JavaScript -->
            </div>
        </div>
        
        <script>
            // Store the figure sets and epochs
            const figureSets = {js_figure_sets};
            const epochs = {js_epochs};
            const setTitles = {js_set_titles};
            let currentIndex = 0;
            let isPlaying = false;
            let playInterval;
            
            // Get DOM elements
            const slider = document.getElementById('epoch-slider');
            const epochValue = document.getElementById('epoch-value');
            const currentIndexDisplay = document.getElementById('current-index');
            const playButton = document.getElementById('play-btn');
            const speedSelect = document.getElementById('speed');
            const figuresContainer = document.getElementById('figures-container');
            
            // Create the figure set elements
            function createFigureSetElements() {{
                figureSets.forEach((figureSet, setIndex) => {{
                    const setDiv = document.createElement('div');
                    setDiv.className = 'figure-set';
                    setDiv.id = `figure-set-${{setIndex}}`;
                    
                    const titleDiv = document.createElement('div');
                    titleDiv.className = 'figure-title';
                    titleDiv.textContent = setTitles[setIndex];
                    setDiv.appendChild(titleDiv);
                    
                    const imageContainer = document.createElement('div');
                    imageContainer.className = 'image-container';
                    
                    const img = document.createElement('img');
                    img.id = `image-${{setIndex}}`;
                    img.alt = setTitles[setIndex];
                    imageContainer.appendChild(img);
                    
                    setDiv.appendChild(imageContainer);
                    figuresContainer.appendChild(setDiv);
                }});
            }}
            
            // Update all figures to show the current epoch
            function updateDisplay() {{
                figureSets.forEach((figureSet, setIndex) => {{
                    const img = document.getElementById(`image-${{setIndex}}`);
                    img.src = 'data:image/png;base64,' + figureSet[currentIndex];
                }});
                
                epochValue.textContent = epochs[currentIndex];
                currentIndexDisplay.textContent = currentIndex + 1;
                slider.value = currentIndex;
            }}
            
            // Event handlers
            slider.addEventListener('input', function() {{
                currentIndex = parseInt(this.value);
                updateDisplay();
            }});
            
            function prevEpoch() {{
                currentIndex = currentIndex > 0 ? currentIndex - 1 : figureSets[0].length - 1;
                updateDisplay();
            }}
            
            function nextEpoch() {{
                currentIndex = currentIndex < figureSets[0].length - 1 ? currentIndex + 1 : 0;
                updateDisplay();
            }}
            
            function togglePlay() {{
                isPlaying = !isPlaying;
                if (isPlaying) {{
                    playButton.textContent = 'Pause';
                    const speed = parseInt(speedSelect.value);
                    playInterval = setInterval(function() {{
                        nextEpoch();
                    }}, speed);
                }} else {{
                    playButton.textContent = 'Play';
                    clearInterval(playInterval);
                }}
            }}
            
            // Update when speed changes
            speedSelect.addEventListener('change', function() {{
                if (isPlaying) {{
                    clearInterval(playInterval);
                    const speed = parseInt(this.value);
                    playInterval = setInterval(function() {{
                        nextEpoch();
                    }}, speed);
                }}
            }});
            
            // Initialize
            createFigureSetElements();
            updateDisplay();
        </script>
    </body>
    </html>
    """

    # Write to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Multi-figure synchronized visualization saved to {output_file}")
    return output_file


def multi_plot(df_path: str, save_dir: str = "multi_heatmaps"):
    # make sure the save directory exists

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    df = pd.read_parquet(df_path)
    epoch_list = df["epoch"].unique()
    epoch_list = [epoch.item() for epoch in df["epoch"].unique()]
    figs_dict = {
        "figs_volume": [],
        "figs_test_loss": [],
        "figs_test_accuracy": [],
        "figs_train_loss": [],
        "figs_train_accuracy": [],
    }
    for epoch in epoch_list:
        mask = df["epoch"] == epoch
        df_epoch = df[mask]
        fig_volume = generate_heatmaps(
            df=df_epoch,
            weight_mode="none",
            stat_type="mean",
            value_stat="mean",
            figsize=(15, 10),
            cmap="coolwarm",
        )
        figs_dict["figs_volume"].append(fig_volume)
        fig_test_loss = generate_heatmaps(
            df=df_epoch,
            weight_mode="none",
            stat_type="mean",
            value_stat="mean",
            figsize=(15, 10),
            cmap="coolwarm",
            value_column="test_loss",
        )
        figs_dict["figs_test_loss"].append(fig_test_loss)
        fig_test_accuracy = generate_heatmaps(
            df=df_epoch,
            weight_mode="none",
            stat_type="mean",
            value_stat="mean",
            figsize=(15, 10),
            cmap="coolwarm",
            value_column="test_accuracy",
        )
        figs_dict["figs_test_accuracy"].append(fig_test_accuracy)
        fig_train_loss = generate_heatmaps(
            df=df_epoch,
            weight_mode="none",
            stat_type="mean",
            value_stat="mean",
            figsize=(15, 10),
            cmap="coolwarm",
            value_column="train_loss",
        )
        figs_dict["figs_train_loss"].append(fig_train_loss)
        fig_train_accuracy = generate_heatmaps(
            df=df_epoch,
            weight_mode="none",
            stat_type="mean",
            value_stat="mean",
            figsize=(15, 10),
            cmap="coolwarm",
            value_column="train_accuracy",
        )
        figs_dict["figs_train_accuracy"].append(fig_train_accuracy)

    export_figure_dict_to_html(
        figure_dict=figs_dict,
        epoch_list=epoch_list,
        output_file=f"{save_dir}/multi_heatmaps.html",
    )


def plot_indicator_table(model, params, save=False):
    device = next(model.parameters()).device
    N = params.N
    group_set = [[i, j, (i + j) % N] for i in range(N) for j in range(N)]
    inputs = t.tensor([g[:2] for g in group_set], dtype=t.long).to(device)

    with t.no_grad():
        model.eval()
        logits = model(inputs)  # shape N^2 x N
        max_prob_entry = t.argmax(logits, dim=-1)  # shape N^2

    output_matrix = einops.rearrange(max_prob_entry, "(n m) -> n m", n=N)  # shape N x N
    hover_labels = [[f"{output_matrix[j][i]}" for i in range(N)] for j in range(N)]
    row_labels = [str(g) for g in range(N)]
    col_labels = row_labels

    # Generate N different colors for the heatmap
    import plotly.colors as pc

    # Using a colorscale that works well for categorical data
    if N <= 10:
        # For small N, use distinct colors from the Plotly qualitative colorscales
        colors = pc.qualitative.Plotly[:N]
    else:
        # For larger N, generate a continuous colorscale with N distinct colors
        colorscale = pc.sequential.Viridis
        colors = [pc.sample_colorscale(colorscale, i / (N - 1))[0] for i in range(N)]

    # Create the colorscale with proper scaling between 0 and 1
    custom_colorscale = []
    for i in range(N):
        # Lower bound for this color
        custom_colorscale.append([i / N, colors[i]])
        # Upper bound for this color (except for the last color)
        if i < N - 1:
            custom_colorscale.append([(i + 1) / N, colors[i]])

    fig = go.Figure(
        data=go.Heatmap(
            z=output_matrix.tolist(),
            showscale=False,
            colorscale=custom_colorscale,
            x=col_labels,
            y=row_labels,
            zmin=0,
            zmax=N - 1,
            customdata=hover_labels,
            hovertemplate="x=%{x}<br>"
            + "y=%{y}<br>"
            + "z=%{customdata}<extra></extra>",
        ),
    )

    fig.update_layout(
        title=f"Final run",
        xaxis={
            "showgrid": True,
            "side": "top",
            "ticks": "outside",
            "tickmode": "array",
            "tickvals": [i for i in range(N)],
            "ticktext": row_labels,
        },
        yaxis={
            "showgrid": True,
            # "autorange": "reversed",
            "side": "left",
            "ticks": "outside",
            "tickmode": "array",
            "tickvals": [i for i in range(N)],
            "ticktext": col_labels,
        },
        height=900,
        width=900,
    )

    if save:
        # Create plots directory if it doesn't exist
        if not os.path.exists("plots"):
            os.mkdir("plots")
        fig.write_html("./plots/plot_final.html")

    return fig


if __name__ == "__main__":

    # Example usage

    bias = [True, False]
    train_set = [1600, 53**2]

    for b, ts in tqdm(itertools.product(bias, train_set)):
        df_path = f"/home/louis/tyche/scripts/shared_database_{b}_{ts}.parquet"
        save_dir = f"multiplot_{b}_{ts}"
        multi_plot(df_path, save_dir=save_dir)
        print("Multi-plot generation complete.")
