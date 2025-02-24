from dataclasses import dataclass
from typing import List, Optional
from jaxtyping import Float
from torch import Tensor
import torch as t
import copy
from tqdm import tqdm


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
    cost_fn: str = "cross_entropy",  # "KL" or "cross_entropy"
):
    """Run SGLD on the model using the given dataset and cost function."""

    sgld_dict = {}
    sgld_dict["params"] = sgld_params
    sgld_dict["loss"] = []
    sgld_dict["L2_norm"] = []
    sgld_dict["L2_distance_init"] = []

    model = copy.deepcopy(model)

    with t.no_grad():

        w_init = (
            t.nn.utils.parameters_to_vector(model.parameters())
            .detach()
            .clone()
            .to(device)
        )

        logits_init = model(dataset[0])["logits"]

    for _ in range(sgld_params.num_steps):

        model.zero_grad()

        indices = t.randint(0, dataset[0].shape[0], (sgld_params.batch_size,))

        inputs, labels = dataset[0][indices], dataset[1][indices]

        logits = model(inputs)["logits"]

        if cost_fn == "KL":
            cost = logit_loss(logits_init, logits)
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

            if t.isnan(cost).any():
                print("Cost is NaN, returning")
                return sgld_dict
    return sgld_dict
