from dataclasses import asdict, dataclass
import datetime
import itertools
import os
import einops
from torch import nn
import torch as t
from typing import Callable, List, Optional, Tuple, Union
from jaxtyping import Float, Int
from tqdm import tqdm
from tyche.estimator import VolumeConfig, VolumeEstimator
from tyche.inductive_bias import MLP_VARIANTS, MLPConfig


def measure_metrics(
    model: MLP_VARIANTS,
    volume_config: VolumeConfig = VolumeConfig(),
    epoch: int = 0,
) -> dict:
    """Measure metrics for a given MLP configuration and model."""

    metrics = {}
    loss = t.nn.CrossEntropyLoss()

    save_dir = f"/mnt/ssd-1/louis/inductive_bias/{model.path_name}"

    if model.mlp_config.save_model:
        os.makedirs(save_dir, exist_ok=True)
        model_dir = os.path.join(save_dir, f"model_{epoch}.pt")
        t.save(model.state_dict(), model_dir)

    # Assuming model has a method to compute loss and accuracy
    with t.no_grad():

        test_dict = model.test_loss_and_acc()
        metrics["test_loss"] = test_dict["loss"].item()
        metrics["test_accuracy"] = test_dict["accuracy"].item()

        inputs, labels = model.data
        logits = model(inputs)
        loss_value = loss(logits, labels)
        metrics["train_loss"] = loss_value.item()
        metrics["train_accuracy"] = (
            (logits.argmax(dim=-1) == labels).float().mean().item()
        )

        estimator = VolumeEstimator.from_config(volume_config)
        try:
            z = estimator.run()
            estimates_tensor = z.estimates.squeeze()
            estimates = estimates_tensor.detach().cpu().numpy()
        except ValueError as e:
            estimates = None

        metrics["volume_estimates"] = estimates

    return metrics


def run_train_and_estimator(
    custom_config: MLPConfig,
    gpu_id: Optional[int],
    volume_config: VolumeConfig = VolumeConfig(),
):

    t.manual_seed(custom_config.seed)

    results = []
    if gpu_id is None:
        device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
    else:
        device = f"cuda:{gpu_id}"
    custom_config.device = device

    t.manual_seed(custom_config.seed)

    model = MLP_VARIANTS(custom_config)
    model.initialize_weights()

    optimizer = t.optim.Adam(
        model.parameters(), lr=custom_config.lr, weight_decay=custom_config.weight_decay
    )

    loss = t.nn.CrossEntropyLoss()

    metrics = measure_metrics(model, volume_config=volume_config)
    metrics["epoch"] = 0
    results.append(metrics)

    for i in tqdm(range(custom_config.training_epochs)):

        optimizer.zero_grad()

        if i % model.mlp_config.eval_interval == 0 and i > 0:
            metrics = measure_metrics(model, volume_config=volume_config, epoch=i)
            metrics["epoch"] = i
            results.append(metrics)

        inputs, labels = model.data

        logits = model(inputs)
        loss_value = loss(logits, labels)

        loss_value.backward()

        optimizer.step()

    config_dict = asdict(custom_config)

    activ_fn = custom_config.activation
    activ_name = (
        activ_fn.__class__.__name__ if hasattr(activ_fn, "__class__") else str(activ_fn)
    )

    config_dict["activation"] = activ_name
    merged_results = [{**config_dict, **result} for result in results]

    return merged_results


if __name__ == "__main__":
    # Example usage
    custom_config = MLPConfig(
        activation=t.nn.ReLU(),
        num_additional_layers=2,
        dimensions=(48,),
        W_amplitude=0.1,
        weight_mode="uniform",
        seed=42,
        training_epochs=10,
        lr=0.001,
        weight_decay=0.0001,
        eval_interval=1,
    )

    gpu_id = 7  # Specify GPU ID or None for CPU
    results = run_train_and_estimator(custom_config, gpu_id)
    print(results)
