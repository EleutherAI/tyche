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
from tyche.local_cache import CacheContext
from tyche.math import gaussint_ln_riemann


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

DEPTH = [1, 2, 3, 4, 5]
WEIGHTSCALE = [math.sqrt(10) ** i for i in np.arange(-1, 2, 0.5)]
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


def process_chunk(configs, gpu_id, results_list, volume_config):
    """Process a chunk of configurations on a specific GPU and add to shared results list"""
    for config in tqdm(configs, desc=f"GPU {gpu_id}"):
        results = run_train_and_estimator(config, gpu_id, volume_config=volume_config)
        results_list += results


def run_mlp_basin(
    mlp_config: MLPConfig = MLPConfig(), volume_config: VolumeConfig = VolumeConfig()
):
    start_gpu = 0
    num_gpus = 8
    assert num_gpus + start_gpu <= 8, "This script is designed to run on 8 GPUs max."

    num_samples = 100
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
            p = Process(
                target=process_chunk,
                args=(process_configs, gpu_id, results, volume_config),
            )
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
    mlp_config = MLPConfig(
        N=53,
        embed_dimension=36,
        linear_dimension=48,
        dimensions=48,
        train_data_size=53**2,
        training_epochs=0,
        eval_interval=1,
        weight_decay=2e-4,
        bias_layer=True,
        bias_unembed=True,
    )

    # bias = [True, False]
    # train_set = [1600, 53**2]

    # for b, ts in tqdm(itertools.product(bias, train_set)):
    #     mlp_config = MLPConfig(
    #         N=53,
    #         embed_dimension=36,
    #         linear_dimension=48,
    #         dimensions=48,
    #         train_data_size=ts,
    #         training_epochs=10001,
    #         eval_interval=500,
    #         weight_decay=2e-4,
    #         bias_layer=b,
    #         bias_unembed=b,
    #     )

    #     print(f"Running with bias={b}, train_data_size={ts}")

    #     # Run the MLP basin with the current configuration

    cfg = VolumeConfig(
        model_type="mlp",
        n_samples=100,
        iters=15,
        cutoff=1e-2,
        cache_mode=None,
        chunking=False,
        reduction=None,
        tol=0.0351,
        tqdm=False,
        gaussint_fn=gaussint_ln_riemann,
        # gaussint_fn=None,
    )

    for sigma_factor in [3, 10]:
        cfg.sigma_factor = sigma_factor
        results = run_mlp_basin(mlp_config=mlp_config, volume_config=cfg)

        database_name = f"shared_database_{sigma_factor}.parquet"
        # Convert results to DataFrame and append to the database
        df = pd.DataFrame(results)

        write_to_database(df, database_name)
