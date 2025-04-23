import math
import torch as t
from torch.multiprocessing import Pool, Process, set_start_method, Manager
import os
from functools import partial
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from tyche.estimator import VolumeConfig, VolumeEstimator
from tyche.MLP_metrics import run_single_estimator_config
from tyche.inductive_bias import MLPConfig


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
    "constant",
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


def process_chunk(configs, gpu_id, results_list):
    """Process a chunk of configurations on a specific GPU and add to shared results list"""
    for config in tqdm(configs, desc=f"GPU {gpu_id}"):
        result = run_single_estimator_config(config, gpu_id)
        results_list.append(result)


def run_mlp_basin(mlp_config: MLPConfig = MLPConfig()):
    start_gpu = 4
    num_gpus = 3
    assert num_gpus + start_gpu <= 8, "This script is designed to run on 8 GPUs max."

    num_samples = 1

    # Create all configurations
    all_configs = []
    for activ_fn in ACTIVATION_FUNCTIONS:
        for d in DEPTH:
            for w in WEIGHTSCALE:
                for weight_mode in WEIGHT_MODES:
                    for i in range(num_samples):
                        mlp_config = mlp_config
                        mlp_config.activation = activ_fn
                        mlp_config.num_additional_layers = d
                        mlp_config.W_amplitude = w
                        mlp_config.weight_mode = weight_mode
                        mlp_config.seed = i

                        all_configs.append(mlp_config)

    # Print total configurations to be run
    total_configs = len(all_configs)
    print(f"Total configurations to run: {total_configs}")

    # Use Manager to create a shared list for results
    with Manager() as manager:
        results = manager.list()

        # randomly shuffle the configurations because some configs take longer
        np.random.shuffle(all_configs)

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
    database_name = "shared_database.parquet"
    # Convert results to DataFrame and append to the database
    df = pd.DataFrame(results)
    write_to_database(df, database_name)
