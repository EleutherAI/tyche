import json
import torch as t
import matplotlib.pyplot as plt
import os
import torch as t
import numpy as np


def plot_sgld_metric(
    sgld_dicts=[],
    metric="loss",
    title="SGLD Loss",
    average: bool = False,
    log_scale: bool = False,
):
    import matplotlib.pyplot as plt
    import numpy as np

    # Return empty figure if no data
    if not sgld_dicts:
        return plt.figure()

    # Clean data: remove trailing NaNs
    cleaned_dicts = []
    for d in sgld_dicts:
        if metric in d:
            data = np.array(d[metric])
            non_nan = np.where(~np.isnan(data))[0]
            if len(non_nan) > 0:
                clean_dict = d.copy()
                clean_dict[metric] = data[: non_nan[-1] + 1]
                cleaned_dicts.append(clean_dict)

    if not cleaned_dicts:
        return plt.figure()

    # Set up plot
    hyperparams = cleaned_dicts[0]["params"]
    # Format parameters with scientific notation if they're very small or large
    eps_str = (
        f"{hyperparams.eps:.1e}"
        if abs(hyperparams.eps) >= 1000 or abs(hyperparams.eps) <= 0.001
        else str(hyperparams.eps)
    )
    gamma_str = (
        f"{hyperparams.gamma:.1e}"
        if abs(hyperparams.gamma) >= 1000 or abs(hyperparams.gamma) <= 0.001
        else str(hyperparams.gamma)
    )
    nbeta_str = (
        f"{hyperparams.nbeta:.1e}"
        if abs(hyperparams.nbeta) >= 1000 or abs(hyperparams.nbeta) <= 0.001
        else str(hyperparams.nbeta)
    )
    title = f"eps={eps_str}, gamma={gamma_str}, nbeta={nbeta_str}, batch_size={hyperparams.batch_size}, num_steps={hyperparams.num_steps}"
    fig = plt.figure(figsize=(5, 3))

    # Plot data
    if average and len(cleaned_dicts) > 1:
        min_len = min(len(d[metric]) for d in cleaned_dicts)
        truncated = [d[metric][:min_len] for d in cleaned_dicts]
        avg = np.nanmean(truncated, axis=0)
        std = np.nanstd(truncated, axis=0)
        plt.fill_between(np.arange(len(avg)), avg - std, avg + std, alpha=0.4)
        plt.plot(np.arange(len(avg)), avg, label="average")
    else:
        for i, d in enumerate(cleaned_dicts):
            plt.plot(np.arange(len(d[metric])), d[metric], '.-', label=f"Run {i+1}")

    plt.xlabel("Epoch")
    plt.ylabel(metric)
    if log_scale:
        plt.yscale("log")
    plt.title(title)
    plt.legend()

    return plt


def plot_sgld_sweep(
    dir_path: str,
    batch_size: int,
    gamma: float,
    average: bool = False,
    metric: str = "loss",
    steps=None,
):
    # Initialize dictionary to store results
    result_dict = {"figure": None, "plotted_data": []}

    # load sweep_config json as dictionary
    with open(f"{dir_path}/hyperparameters.json", "r") as f:
        sweep_config = json.load(f)

    if steps is None:
        steps = sweep_config["num_steps"][0]

    # Parameter combinations
    params = []
    for n in sweep_config["nbeta"]:
        for e in sweep_config["eps"]:
            params.append((n, e))

    # Calculate number of rows and columns needed
    n_plots = len(params)
    n_cols = min(3, n_plots)  # Maximum 3 columns
    n_rows = (n_plots + n_cols - 1) // n_cols  # Ceiling division

    # Create figure with the exact number of subplots needed
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    if n_plots > 1:
        axs = axs.flatten()  # Flatten to make indexing easier

    # Add a big title to the entire figure
    fig.suptitle(
        f"SGLD Sweep: Batch Size = {batch_size}, Gamma = {gamma}, Metric= {metric}",
        fontsize=20,
        y=0.98,
    )

    # Plot each combination
    for idx, (n, e) in enumerate(params):
        ax = axs[idx] if n_plots > 1 else axs  # Get the current subplot
        plot_data = []  # Store data for current plot

        # Construct folder path
        folder = f"{dir_path}/sgld_eps{e}_nbeta{n}_batch{batch_size}_steps{steps}_gamma{gamma}"

        if abs(e) >= 1000 or abs(e) <= 0.001 or abs(n) >= 1000 or abs(n) <= 0.001:
            ax.set_title(f"eps={e:.1e}, nbeta={n:.1e}")
        else:
            ax.set_title(f"eps={e}, nbeta={n}")

        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)

        # Try to load and plot data
        try:
            if not os.path.exists(folder):
                ax.text(0.5, 0.5, "No data folder", transform=ax.transAxes, ha="center")
                continue

            files = [
                f for f in os.listdir(folder) if f.endswith(".pt") or f.endswith(".pth")
            ]
            if not files:
                ax.text(0.5, 0.5, "No files found", transform=ax.transAxes, ha="center")
                continue

            # Load all dictionaries
            sgld_dicts = []
            for file in files:
                try:
                    dict_data = t.load(f"{folder}/{file}", weights_only=False)
                    if metric in dict_data:
                        # Clean NaN values
                        data = np.array(dict_data[metric])
                        non_nan = np.where(~np.isnan(data))[0]
                        if len(non_nan) > 0:
                            dict_data[metric] = data[: non_nan[-1] + 1]
                            sgld_dicts.append(dict_data)
                except Exception as ex:
                    print(f"Error loading {file}: {ex}")

            if not sgld_dicts:
                ax.text(0.5, 0.5, "No valid data", transform=ax.transAxes, ha="center")
                continue

            # Plot data
            if average and len(sgld_dicts) > 1:
                min_len = min(len(d[metric]) for d in sgld_dicts)
                truncated = [d[metric][:min_len] for d in sgld_dicts]
                avg = np.nanmean(truncated, axis=0)
                std = np.nanstd(truncated, axis=0)
                x = np.arange(len(avg))
                ax.fill_between(x, avg - std, avg + std, alpha=0.4)
                ax.plot(x, avg, label="average")
                plot_data = t.tensor(avg)  # Convert to torch tensor
            else:
                for i, d in enumerate(sgld_dicts):
                    data = d[metric]
                    ax.plot(np.arange(len(data)), data, label=f"Run {i+1}")
                    plot_data.append(t.tensor(data))  # Convert to torch tensor
                if plot_data:  # If there's data, stack it
                    plot_data = t.stack(plot_data)

            ax.legend()

        except Exception as ex:
            print(f"Error for eps={e}, nbeta={n}: {ex}")
            ax.text(0.5, 0.5, "Error plotting", transform=ax.transAxes, ha="center")
            plot_data = t.tensor([])  # Empty tensor for failed plots

        result_dict["plotted_data"].append(plot_data)

    # Remove any unused subplots
    if n_plots > 1:
        for idx in range(n_plots, len(axs)):
            fig.delaxes(axs[idx])

    # Adjust spacing to make room for the suptitle
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Store figure in result dictionary
    result_dict["figure"] = fig
    return result_dict
