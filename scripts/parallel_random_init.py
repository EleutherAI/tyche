import math
import torch as t
from torch.multiprocessing import Pool, Process, set_start_method, Manager
import os
from functools import partial
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from tyche.estimator import VolumeConfig, VolumeEstimator


class GaussianActivation(t.nn.Module):
    def forward(self, x):
        return t.exp(-(x**2))  # Gaussian function


ACTIVATION_FUNCTIONS = [
    t.nn.ReLU(),
    t.nn.GELU(),
    t.nn.Tanh(),
    GaussianActivation(),
]
DEPTH = [1, 2, 3, 4, 5]
WEIGHTSCALE = [math.sqrt(10) ** i for i in np.arange(-1, 2, 0.5)]
WEIGHT_MODES = [
    "uniform",
    "xavier_uniform",
    "normal",
    "xavier_normal",
    "none",
]

database_name = "shared_database_parallel.parquet"


def read_database(filepath=database_name):
    """Read the shared database file if it exists, otherwise return empty DataFrame."""
    if os.path.exists(filepath):
        try:
            return pd.read_parquet(filepath)
        except Exception as e:
            print(f"Error reading database: {e}")
            return pd.DataFrame()
    else:
        print(f"Database file '{filepath}' not found. Creating new DataFrame.")
        return pd.DataFrame()


def write_to_database(new_data, filepath=database_name):
    """Append new data to the existing database."""
    # Read existing data
    existing_df = read_database(filepath)

    # Convert new_data to DataFrame if it's not already
    if not isinstance(new_data, pd.DataFrame):
        new_data = pd.DataFrame(new_data)

    # Combine existing data with new data
    combined_df = pd.concat([existing_df, new_data], ignore_index=True)

    # Create directory if it doesn't exist
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # Write back to file
    try:
        combined_df.to_parquet(filepath)
        print(f"Successfully wrote to {filepath}")
    except Exception as e:
        print(f"Error writing to database: {e}")

    return combined_df


# Try to set the start method to 'spawn' for better CUDA compatibility
try:
    set_start_method("spawn")
except RuntimeError:
    pass


def run_single_config(config_tuple, gpu_id):
    """Run a single configuration on a specific GPU"""
    activ_fn, d, w, weight_mode, i = config_tuple

    t.manual_seed(i)
    model_name = {
        "activation": activ_fn,
        "N": 113,
        "embed_dimension": 24,
        "linear_dimension": 48,
        "intermediate": "pure",
        "embedding_tied": False,
        "unembedding_tied": False,
        "bias_unembed": False,
        "num_additional_layers": d,
        "dimensions": (48),
        "bias_layer": False,
        "W_amplitude": w,
        "weight_mode": weight_mode,
        "device": f"cuda:{gpu_id}",
    }

    cfg = VolumeConfig(
        model_type="mlp",
        model_name=model_name,
        n_samples=100,
        iters=30,
        cutoff=1e-2,
        cache_mode=None,
        chunking=False,
        reduction=None,
        device=f"cuda:{gpu_id}",
        tol=0.035,
        tqdm=False,
    )

    estimator = VolumeEstimator.from_config(cfg)
    try:
        z = estimator.run()
        estimates_tensor = z.estimates.squeeze()
        estimates = estimates_tensor.detach().cpu().numpy()
    except ValueError as e:
        estimates = None

    activ_name = (
        activ_fn.__class__.__name__ if hasattr(activ_fn, "__class__") else str(activ_fn)
    )

    model_name["activation"] = activ_name
    model_name["volume_estimates"] = estimates
    model_name["sample_id"] = i

    return model_name


def process_chunk(configs, gpu_id, results_list):
    """Process a chunk of configurations on a specific GPU and add to shared results list"""
    for config in tqdm(configs, desc=f"GPU {gpu_id}"):
        result = run_single_config(config, gpu_id)
        results_list.append(result)


def run_mlp_basin():
    start_gpu = 3
    num_gpus = 3
    assert num_gpus + start_gpu <= 8, "This script is designed to run on 4 GPUs only."

    num_samples = 3

    # Create all configurations
    all_configs = []
    for activ_fn in ACTIVATION_FUNCTIONS:
        for d in DEPTH:
            for w in WEIGHTSCALE:
                for weight_mode in WEIGHT_MODES:
                    for i in range(num_samples):
                        all_configs.append((activ_fn, d, w, weight_mode, i))

    # Print total configurations to be run
    total_configs = len(all_configs)
    print(f"Total configurations to run: {total_configs}")

    # Use Manager to create a shared list for results
    with Manager() as manager:
        results = manager.list()

        # Split configs into chunks for each GPU
        chunks = np.array_split(all_configs, num_gpus)
        processes = []

        # Create and start a process for each GPU
        for i, gpu_id in enumerate(range(start_gpu, start_gpu + num_gpus)):
            # Use i to index into chunks, but use gpu_id for the actual GPU device
            process_configs = chunks[i]  # Access chunks using index 0 to num_gpus-1
            p = Process(target=process_chunk, args=(process_configs, gpu_id, results))
            processes.append(p)
            p.start()
        # Wait for all processes to complete
        for p in processes:
            p.join()

        # Convert shared list to normal list
        final_results = list(results)

    return final_results


# Main execution
if __name__ == "__main__":
    # Run the MLP basin function with parallelization
    results = run_mlp_basin()

    # Convert results to DataFrame and append to the database
    df = pd.DataFrame(results)
    write_to_database(df)
