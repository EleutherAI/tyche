# Tyche: Precisely estimating the sizes of basins in neural net parameter space corresponding to interpretable behaviors
*She was often shown winged, wearing a crown, and bearing a sceptre and cornucopia; but she also appeared blindfolded and with various devices signifying uncertainty and risk. Among her monuments was a temple at Argos, where the legendary Palamedes is said to have dedicated to her the first set of dice.*

## Purpose

This repo contains tool to measure the sizes of behavioral basins via Monte Carlo methods.

A behavioral basin is a region in parameter space defined by functional behavior, such as loss on a dataset, rather than e.g. similarity of weight vectors.

"Size" here means total prior measure, where the prior is uniform (so that size = volume) or Gaussian (e.g. init distribution density).

The sizes of behavioral basins can be interpreted as a measure of the complexity of learned behaviors (by taking the negative logarithm). When the prior is normalized, they can also be interpreted as the prior probability of the basin-defining behavior.

## Installation

* Python 3.11 is recommended
* `pip install -e .` or `pip install .`

This installs two packages: `tyche` and `palamedes`.

`tyche` uses binary search to find the edge of a basin, the Huang et al volume estimator to estimate volume, custom Gaussian integral code to maintain precision for very high-dimensional integrals, and a preconditioner matrix to improve sample efficiency by alleviating "Jensen bias". It is fast but can lead to dramatic underestimates of basin size. As seen in "[Estimating the Probability of Sampling a Trained Neural Network at Random](https://arxiv.org/abs/2501.18812)".

`palamedes` (WIP) is a new project that uses SGLD and thermodynamic integration to achieve a measurement of basin volume. It is slower than `tyche` but (once it's done) will give much more precise basin size measurements.

The remaining documentation is `tyche`-specific.

## Usage (HuggingFace models)

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from tyche import VolumeConfig, VolumeEstimator

# Load any CausalLM model, tokenizer, and dataset
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-14m")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-14m")
tokenizer.pad_token_id = 1  # pythia-specific
tokenizer.eos_token_id = 0  # pythia-specific
dataset = load_dataset("EleutherAI/lambada_openai", name="en", split="test", trust_remote_code=True)

# Configure the estimator
cfg = VolumeConfig(model=model, 
                   tokenizer=tokenizer, 
                   dataset=dataset, 
                   text_key="text",  # must match dataset field
                   n_samples=10,  # number of MC samples
                   cutoff=1e-2,  # KL-divergence cutoff (nats)
                   max_seq_len=2048,  # max sequence length for tokenizer or chunk_and_tokenize
                   val_size=10,  # number of dataset sequences to use. default (None) uses all.
                   cache_mode=None,  # see below
                   chunking=False,  # whether to use chunk_and_tokenize
                   )
estimator = VolumeEstimator.from_config(cfg)

# Run the estimator
result = estimator.run()
```

The result object is a `VolumeResult` with the following fields:

* `estimates`: estimated log-probability of basin (natural log!)
* `deltas`: actual KL differences (should be within ±10% of cutoff)
* `props`, `mults`, `logabsint`: pieces of estimation calculation (for debugging)

Preconditioners are not yet supported for this interface.


### Cache mode

This setting controls how the original (unperturbed, center-of-basin) model outputs are cached:
* `None`: (default) compute original-model outputs from scratch each iteration
* `"cpu"`: keep original-model outputs on the CPU, moving to GPU in batches (slow...)
* `"gpu"`: keep original-model outputs on the GPU (OOMs if val_size is too large)


## Usage (models from the paper)

An interface for ConvNeXt and Pythia is available through `src/basin_volume/estimator.py`, with example usage (similar to the HuggingFace interface above) in `scripts/expt_paper.py`.

The MLP on `digits` is implemented in JAX on the branch `jax-hybrid`, which has additional dependencies:

* `pip install -U "jax[cuda12]"`
* `pip install -e .` or `pip install .`

See `notebooks/bigmlp_basins.ipynb` for usage.


## Structure

`notebooks/`: Jupyter notebooks

`src/tyche/`: package source

`.../convnext.py`: ConvNeXt on `cifar10`

`.../data.py`: data preprocessing (from `sparsify`)

`.../estimator.py`: classes for managing experiments and models

`.../math.py`: integrals and such for high-dim geometry

`.../precondition.py`: preconditioners

`.../pythia.py`: Pythia on the Pile

`.../utils.py`: misc helpful tools

`.../volume.py`: core volume-estimation code

`src/palamedes/`: Palamedes package source

`scripts/`: command-line scripts (Python and shell)

`.../expt_paper.py`: actual script used for results in paper

`.../train_vision.py`: training script for ConvNeXt models (adapted from [https://github.com/EleutherAI/features-across-time])

`old/`: large collection of old experiments and code (messy)

`.../basin_precondition.ipynb`: early version of this project as a giant Jupyter notebook
