from dataclasses import dataclass
import datetime
import itertools
import os
import einops
from torch import nn
import torch as t
from typing import Callable, List, Optional, Tuple, Union
from jaxtyping import Float, Int
from tyche.estimator import VolumeConfig, VolumeEstimator
from tyche.inductive_bias import MLP_VARIANTS, MLPConfig


def run_single_estimator_config(config_tuple, gpu_id: Optional[int]) -> dict:
    """Run a single configuration on a specific GPU"""
    activ_fn, d, w, weight_mode, i = config_tuple
    if gpu_id is None:
        device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
    else:
        device = f"cuda:{gpu_id}"

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
        "device": device,
        "b_amplitude": 0.0,
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
        device=device,
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


@t.no_grad()
def measure_metrics(
    model: MLP_VARIANTS,
    training_data: t.Tensor,
) -> dict:
    """Measure metrics for a given MLP configuration and model."""
    # TODO: Fix model_name mess
    metrics = {}
    loss = t.nn.CrossEntropyLoss()

    save_dir = 
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Assuming model has a method to compute loss and accuracy
    with t.no_grad():
        model_dir = os.path.join(save_dir, f"model_{i}.pt")
        t.save(model.state_dict(), model_dir)
        test_loss, test_acc = model.test_loss_and_acc()
        metrics["test_loss"] = test_loss.item()
        metrics["test_accuracy"] = test_acc.item()

        inputs, labels = training_data
        logits = model(inputs)
        loss_value = loss(logits, labels)
        metrics["train_loss"] = loss_value.item()
        metrics["train_accuracy"] = (
            (logits.argmax(dim=-1) == labels).float().mean().item()
        )

        cfg = VolumeConfig(
            model_type="mlp",
            model_name=model_dir,
            n_samples=100,
            iters=30,
            cutoff=1e-2,
            cache_mode=None,
            chunking=False,
            reduction=None,
            device=model.mlp_config.device,
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

        metrics["volume_estimates"] = estimates

    return metrics


def run_single_train_and_run(custom_config: MLPConfig, gpu_id: Optional[int]):

    t.manual_seed(custom_config.seed)

    results = []
    if gpu_id is None:
        device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
    else:
        device = f"cuda:{gpu_id}"
    t.manual_seed(custom_config.seed)

    model = MLP_VARIANTS(custom_config)

    model.initialize_weights()
    volume_cfg = VolumeConfig(
        model_type="mlp",
        model=model,
        n_samples=100,
        iters=30,
        cutoff=1e-2,
        cache_mode=None,
        chunking=False,
        reduction=None,
        device=device,
        tol=0.035,
        tqdm=False,
    )

    optimizer = t.optim.Adam(
        model.parameters(), lr=custom_config.lr, weight_decay=custom_config.weight_decay
    )

    loss = t.nn.CrossEntropyLoss()

    metrics = measure_metrics(model, training_data=model.data)

    results.append(metrics)

    for i in range(custom_config.epochs):

        optimizer.zero_grad()

        if i % model.mlp_config.eval_interval == 0 and i > 0:
            metrics = measure_metrics(model, model.data)
            results.append(metrics)

        inputs, labels = model.data

        logits = model.forward(inputs)
        loss_value = loss(logits, labels)

        loss_value.backward()

        optimizer.step()
