# make a Gaussian activation function
import math
import numpy as np
import torch as t
from tqdm import tqdm
from tyche.estimator import VolumeConfig, VolumeEstimator
from tyche.inductive_bias import MLPConfig
import pandas as pd
import os

database_name = "shared_database_longer.parquet"


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


def run_mlp_basin():
    results = []
    num_samples = 3

    for activ_fn in tqdm(ACTIVATION_FUNCTIONS):
        for d in DEPTH:
            for w in WEIGHTSCALE:
                for weight_mode in WEIGHT_MODES:
                    for i in range(num_samples):
                        t.manual_seed(i)
                        model_name = {
                            "activation": activ_fn,
                            "N": 113,
                            "embed_dimension": 24,
                            "linear_dimension": 48,
                            "intermediate": "pure",  # pure, real_multiplication, complex_multiplication, quaterionic_multiplication
                            "embedding_tied": False,
                            "unembedding_tied": False,
                            "bias_unembed": False,
                            "num_additional_layers": d,
                            "dimensions": (48),
                            "bias_layer": False,
                            "W_amplitude": w,
                            "weight_mode": weight_mode,  # uniform, xavier_uniform, normal, xavier_normal
                            "device": "cuda:7",
                        }

                        cfg = VolumeConfig(
                            model_type="mlp",
                            model_name=model_name,
                            n_samples=100,  # number of MC samples
                            iters=30,
                            cutoff=1e-2,  # KL-divergence cutoff (nats)
                            cache_mode=None,  # see below
                            chunking=False,  # whether to use chunk_and_tokenize
                            reduction=None,
                            device="cuda:7",
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
                            activ_fn.__class__.__name__
                            if hasattr(activ_fn, "__class__")
                            else str(activ_fn)
                        )

                        model_name["activation"] = activ_name  # Store as string
                        model_name["volume_estimates"] = estimates
                        model_name["sample_id"] = i  # Ensure it's a simple int

                        results.append(model_name)

    return results


if __name__ == "__main__":

    # Run the MLP basin function
    results = run_mlp_basin()

    # Convert results to DataFrame and append to the database
    df = pd.DataFrame(results)

    write_to_database(
        df,
    )
