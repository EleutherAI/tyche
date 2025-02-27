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


def logit_loss(
    p: Float[t.Tensor, "batch_size d_output"], q: Float[t.Tensor, "batch_size d_output"]
) -> Float[t.Tensor, ""]:
    log_p = p - t.logsumexp(p, dim=-1, keepdim=True)  # log softmax of p
    log_q = q - t.logsumexp(q, dim=-1, keepdim=True)  # log softmax of q

    # KL = sum(exp(log_p) * (log_p - log_q))
    return (log_p.exp() * (log_p - log_q)).sum(dim=-1).mean()


def sgld(
    model,
    sgld_params: SGLDParams,
    device,
    dataset: List,
    cost_fn: str = "cross_entropy",  # "KL" or "cross_entropy" or "zero"
):
    """Run SGLD on the model using the given dataset and cost function."""

    sgld_dict = {}
    sgld_dict["params"] = sgld_params
    sgld_dict["loss"] = []
    sgld_dict["L2_norm"] = []
    sgld_dict["L2_distance_init"] = []
    save_dir = sgld_params.save_dir

    model = copy.deepcopy(model)

    with t.no_grad():

        w_init = (
            t.nn.utils.parameters_to_vector(model.parameters())
            .detach()
            .clone()
            .to(device)
        )

        logits_init = model(dataset[0])["logits"]

    if save_dir is None and sgld_params.checkpoint_every:
        model_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_dir = "/mnt/ssd-1/louis/palamedes/sgld_samples/" + model_time
        os.makedirs(save_dir, exist_ok=True)
        t.save(w_init, f"{save_dir}/model_0.pt")
        print(f"Saving to {save_dir}")

    for _ in range(sgld_params.num_steps):

        if (
            sgld_params.checkpoint_every
            and _ % sgld_params.checkpoint_every == 0
            and _ > 0
        ):
            t.save(w, f"{save_dir}/model_{_}.pt")

        model.zero_grad()

        if cost_fn != "zero":
            indices = t.randint(0, dataset[0].shape[0], (sgld_params.batch_size,))

            inputs, labels = dataset[0][indices], dataset[1][indices]

            logits = model(inputs)["logits"]

            if cost_fn == "KL":
                cost = logit_loss(logits_init[indices], logits)
            elif cost_fn == "cross_entropy":
                cost = t.nn.functional.cross_entropy(logits, labels)
            sgld_dict["loss"].append(cost.item())

            cost.backward()

        with t.no_grad():

            w = t.nn.utils.parameters_to_vector(model.parameters())

            loss_grad = -t.nn.utils.parameters_to_vector(
                [
                    p.grad if p.grad is not None else t.zeros_like(p)
                    for p in model.parameters()
                ]
            )
            localization_grad = (w_init - w) * sgld_params.gamma

            total_grad = (
                sgld_params.eps
                / 2
                * (sgld_params.nbeta * loss_grad + localization_grad)
            )  # we don't need to divide by batch size because cross entropy already averages over that

            gaussian_noise = t.randn_like(w, device=device) * t.sqrt(
                t.tensor(sgld_params.eps)
            )

            w = w + total_grad + gaussian_noise
            t.nn.utils.vector_to_parameters(w, model.parameters())

            sgld_dict["L2_norm"].append(t.norm(w).item())

            sgld_dict["L2_distance_init"].append(t.norm(w - w_init).item())

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
