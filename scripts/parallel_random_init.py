import copy
import itertools
import math
import torch as t
from torch.multiprocessing import Pool, Process, set_start_method, Manager
import os
from functools import partial
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from tyche.estimator import VolumeConfig, VolumeEstimator
from tyche.MLP_metrics import run_train_and_estimator
from tyche.inductive_bias import MLPConfig


class GaussianActivation(t.nn.Module):
    def forward(self, x):
        return t.exp(-(x**2))  # Gaussian function


ACTIVATION_FUNCTIONS = [
    t.nn.ReLU(),
    t.nn.GELU(),
    t.nn.Tanh(),
    t.nn.Sigmoid(),
    GaussianActivation(),
]

ACTIVATION_PAIRS = [(activ, "pure") for activ in ACTIVATION_FUNCTIONS] + [
    (t.nn.ReLU(), "complex_multiplication")
]

DEPTH = [1, 3, 5]
WEIGHTSCALE = [math.sqrt(10) ** i for i in np.arange(-1, 2, 1)]
WEIGHT_MODES = ["none"]


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


def write_to_database(new_data, filepath=database_name, overwrite=True):
    """
    Write new data to the database.

    Args:
        new_data: Data to write to the database
        filepath: Path to the database file
        overwrite: If True, overwrite existing data; if False, append to existing data
    """
    # Convert new_data to DataFrame if it's not already
    if not isinstance(new_data, pd.DataFrame):
        new_data = pd.DataFrame(new_data)

    # If overwrite is False, read and combine with existing data
    if not overwrite:
        # Read existing data
        existing_df = read_database(filepath)
        # Combine existing data with new data
        combined_df = pd.concat([existing_df, new_data], ignore_index=True)
    else:
        # Just use the new data
        combined_df = new_data

    # Create directory if it doesn't exist
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # Write back to file
    try:
        combined_df.to_parquet(filepath)
        print(f"Successfully {'overwrote' if overwrite else 'wrote to'} {filepath}")
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
        results = run_train_and_estimator(config, gpu_id)
        results_list += results


def run_mlp_basin(mlp_config: MLPConfig = MLPConfig()):
    start_gpu = 1
    num_gpus = 7
    assert num_gpus + start_gpu <= 8, "This script is designed to run on 8 GPUs max."

    num_samples = 10
    all_configs = []
    for (activ_fn, intermediate_fn), d, w, weight_mode, i in itertools.product(
        ACTIVATION_PAIRS, DEPTH, WEIGHTSCALE, WEIGHT_MODES, range(num_samples)
    ):
        config_dict = {
            "activation": activ_fn,
            "intermediate": intermediate_fn,  # Added intermediate function
            "num_additional_layers": d,
            "W_amplitude": w,
            "weight_mode": weight_mode,
            "seed": i,
        }
        custom_config = copy.deepcopy(mlp_config)
        custom_config.update(**config_dict)
        all_configs.append(custom_config)

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
    # mlp_config = MLPConfig(
    #     N=53,
    #     embed_dimension=36,
    #     linear_dimension=48,
    #     dimensions=48,
    #     train_data_size=53**2,
    #     training_epochs=0,
    #     eval_interval=1000,
    #     weight_decay=2e-4,
    #     bias_layer=True,
    #     bias_unembed=True,
    # )

    bias = [True, False]
    train_set = [1600, 53**2]

    for b, ts in tqdm(itertools.product(bias, train_set)):
        mlp_config = MLPConfig(
            N=53,
            embed_dimension=36,
            linear_dimension=48,
            dimensions=48,
            train_data_size=ts,
            training_epochs=10001,
            eval_interval=500,
            weight_decay=2e-4,
            bias_layer=b,
            bias_unembed=b,
        )

        print(f"Running with bias={b}, train_data_size={ts}")

        # Run the MLP basin with the current configuration
        results = run_mlp_basin(mlp_config=mlp_config)

        database_name = f"shared_database_{b}_{ts}.parquet"
        # Convert results to DataFrame and append to the database
        df = pd.DataFrame(results)
        write_to_database(df, database_name)
