from dataclasses import dataclass
import datetime
import json
import os
from typing import List, Optional
from jaxtyping import Float
from torch import Tensor
import torch as t
import copy
from tqdm import tqdm
from tyche.convnext import load_cifar10_splits_dict, load_convnext_checkpoint


@dataclass
class SGLDParams:
    """See https://arxiv.org/pdf/2308.12108#page=20 for details."""

    eps: float = 1  # total step size of SGLD step
    gamma: float = (
        1  # radius of the ball, controls how far we stray away from w* (higher is more exploration)
    )
    nbeta: float = (
        1  # temperature (up to multiplication by data set size), scales the SGD part (as opposed to the noise part)
    )
    num_steps: int = 1
    batch_size: int = 128  # batch size
    checkpoint_every: Optional[int] = None  # checkpoint every n steps
    save_dir: Optional[str] = None  # directory to save checkpoints
    preconditioner: str = "none"  # preconditioner to use, "none" or "rmsprop"
    alpha_rmsprop: float = 0.99  # rmsprop alpha
    lambda_rmsprop: float = 1e-5  # rmsprop eps


def logit_loss(
    p: Float[t.Tensor, "batch_size d_output"], q: Float[t.Tensor, "batch_size d_output"]
) -> Float[t.Tensor, ""]:
    log_p = p - t.logsumexp(p, dim=-1, keepdim=True)  # log softmax of p
    log_q = q - t.logsumexp(q, dim=-1, keepdim=True)  # log softmax of q

    return (log_p.exp() * (log_p - log_q)).sum(dim=-1).mean()


def rms_preconditioner(
    gradient: Float[t.Tensor, " d_params"],
    v_rmsprop: Float[t.Tensor, " d_params"],
    time_step: int,
    sgld_params: SGLDParams,
) -> Float[t.Tensor, " d_params"]:

    v_rmsprop = (
        sgld_params.alpha_rmsprop * v_rmsprop
        + (1 - sgld_params.alpha_rmsprop) * gradient**2
    )

    # bias correction
    v_rmsprop_corrected = v_rmsprop / (1 - sgld_params.alpha_rmsprop ** (time_step + 1))

    preconditioner = 1 / (t.sqrt(v_rmsprop_corrected) + sgld_params.lambda_rmsprop)

    # print((preconditioner - preconditioner.mean()).abs().max().item())

    return preconditioner, v_rmsprop

def cycle(iterable, limit=None):
    """
    Use this function to cycle through a dataloader. Unlike itertools.cycle, this function doesn't cache
    values in memory.

    Note: Be careful with cycling a shuffled interable. The shuffling will be different for each loop dependent on the seed
    state, unlike with itertools.cycle.

    :param iterable: Iterable to cycle through
    :param limit: Number of cycles to go through. If None, cycles indefinitely.
    """
    index = 0
    if limit is None:
        limit = float("inf")
    while True:
        for x in iterable:
            if index >= limit:
                return
            else:
                yield x
            index += 1

def sgld(
    model,
    sgld_params: SGLDParams,
    device,
    dataset: List | t.utils.data.DataLoader,
    cost_fn: str = "cross_entropy",  # "KL" or "cross_entropy" or "zero"
    fp16: bool = False,
):
    """Run SGLD on the model using the given dataset and cost function."""

    # Record metrics
    sgld_dict = {}
    sgld_dict["params"] = sgld_params
    sgld_dict["loss"] = []
    sgld_dict["L2_norm"] = []
    sgld_dict["L2_distance_init"] = []
    sgld_dict["v_rmsprop"] = []
    sgld_dict["preconditioner"] = []
    sgld_dict["gradient_norm"] = []
    save_dir = sgld_params.save_dir

    model = copy.deepcopy(model)

    if isinstance(dataset, t.utils.data.DataLoader):
        # Move entire dataset to GPU before cycling through it
        dataset_list = [
            (inputs.to(device), labels.to(device)) for inputs, labels in dataset
        ]
        dataset_iter = cycle(dataset_list)

    with t.no_grad():

        w_init = (
            t.nn.utils.parameters_to_vector(model.parameters())
            .detach()
            .clone()
            .to(device)
        )

        all_inputs, all_labels = (
            (None, None)  # not implementing right now bc we don't need it for crossentropy
            if isinstance(dataset, t.utils.data.DataLoader)
            else dataset
        )

        if all_inputs is not None:

            logits_init = (
                model(all_inputs)["logits"]
                if "logits" in model(all_inputs)
                else model(all_inputs).logits
            )

        preconditioner = t.ones_like(w_init, device=device)
        v_rmsprop = t.zeros_like(w_init, device=device)

    if save_dir is None and sgld_params.checkpoint_every:
        model_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_dir = "/mnt/ssd-1/louis/palamedes/sgld_samples/" + model_time
        os.makedirs(save_dir, exist_ok=True)
        t.save(w_init, f"{save_dir}/model_0.pt")
        print(f"Saving to {save_dir}")

    for i in range(sgld_params.num_steps):

        if (
            sgld_params.checkpoint_every
            and i % sgld_params.checkpoint_every == 0
            and i > 0
        ):
            t.save(w, f"{save_dir}/model_{_}.pt")

        model.zero_grad()

        if cost_fn != "zero":
            if isinstance(dataset, t.utils.data.DataLoader):
                inputs, labels = next(dataset_iter)
            else:
                indices = t.randint(0, dataset[0].shape[0], (sgld_params.batch_size,))
                inputs, labels = dataset[0][indices], dataset[1][indices]

            logits = (
                model(inputs)["logits"]
                if "logits" in model(inputs)
                else model(inputs).logits
            )

            if cost_fn == "KL":
                cost = logit_loss(logits_init[indices], logits)
            elif cost_fn == "cross_entropy":
                cost = t.nn.functional.cross_entropy(logits, labels)
                print(cost.item())
            sgld_dict["loss"].append(cost.item())

            cost.backward()

        with t.no_grad():

            w = t.nn.utils.parameters_to_vector(model.parameters()).to(dtype=t.float32)

            loss_grad = -t.nn.utils.parameters_to_vector(
                [
                    p.grad if p.grad is not None else t.zeros_like(p)
                    for p in model.parameters()
                ]
            ).to(dtype=t.float32)

            if sgld_params.preconditioner != "none":
                preconditioner, v_rmsprop = rms_preconditioner(
                    gradient=loss_grad,
                    v_rmsprop=v_rmsprop,
                    sgld_params=sgld_params,
                    time_step=i,
                )

            localization_grad = (w_init - w) * sgld_params.gamma

            total_grad = (
                sgld_params.eps
                / 2
                * (sgld_params.nbeta * loss_grad + localization_grad)
            )  # we don't need to divide by batch size because cross entropy already averages over that

            total_grad = preconditioner * total_grad

            gaussian_noise = (
                t.randn_like(w, device=device)
                * t.sqrt(t.tensor(sgld_params.eps))
                * t.sqrt(preconditioner)
            )

            w = w + total_grad + gaussian_noise
            if fp16:
                w = w.to(dtype=t.float16)
            t.nn.utils.vector_to_parameters(w, model.parameters())

            sgld_dict["L2_norm"].append(t.norm(w).item())

            sgld_dict["L2_distance_init"].append(t.norm(w - w_init).item())

            sgld_dict["v_rmsprop"].append(v_rmsprop.median().item())

            sgld_dict["preconditioner"].append(preconditioner.median().item())

            sgld_dict["gradient_norm"].append(t.norm(loss_grad).item())

            if cost_fn != "zero" and t.isnan(cost).any():
                print("Cost is NaN, returning")
                return sgld_dict
    return sgld_dict


if __name__ == "__main__":

    device = t.device("cuda:7" if t.cuda.is_available() else "cpu")

    cifar10_ds = load_cifar10_splits_dict(size=10000, device=device)

    val_ds = cifar10_ds["val"].to(device)
    clean_ds = cifar10_ds["clean"].to(device)
    poisoned_ds = cifar10_ds["poison"].to(device)
    val_ds_labels = cifar10_ds["val_labels"].to(device)
    clean_ds_labels = cifar10_ds["clean_labels"].to(device)
    poisoned_ds_labels = cifar10_ds["poison_labels"].to(device)

    params = SGLDParams(
        eps=1e-4,
        gamma=0,
        nbeta=1000,
        batch_size=512,
        num_steps=10000,
        checkpoint_every=100,
        preconditioner="rmsprop",
    )

    RUNS_DIR = "/mnt/ssd-1/adam/basin-volume/runs"

    models_clean = [
        load_convnext_checkpoint(
            RUNS_DIR + f"/b16pai_p001/checkpoint-{2**(step+1)}", device=device
        )
        for step in range(16)
    ]

    model = models_clean[-1]

    sgld_dict = sgld(
        model,
        params,
        dataset=[val_ds, val_ds_labels],
        cost_fn="KL",
        device=device,
    )

    # save sgld_dict as json

    with open("sgld_dict.json", "w") as f:
        json.dump(sgld_dict, f)
        f.close()
