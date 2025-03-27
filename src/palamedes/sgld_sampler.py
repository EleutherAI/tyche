from dataclasses import dataclass
import datetime
import json
import os
import pickle
from typing import List, Optional
from itertools import cycle as iter_cycle
from jaxtyping import Float
from torch import Tensor
import torch as t
import copy
from tqdm import tqdm
from tyche.convnext import load_cifar10_splits_dict, load_convnext_checkpoint
import argparse


@dataclass
class SGLDParams:
    """Parameters for Stochastic Gradient Langevin Dynamics (SGLD) sampling.
    
    Attributes:
        eps: Total step size of SGLD step
        gamma: Coefficient of localization term. Controls exploration distance from w*
               (higher values = less exploration)
        gamma_prior: Coefficient of prior term. Controls exploration distance from zero
                    (higher values = less exploration)
        nbeta: Temperature parameter (multiplied by dataset size).
               Scales the SGD component relative to noise
        num_steps: Number of SGLD iterations to perform
        batch_size: Mini-batch size for gradient computation
        checkpoint_every: Optional; Save model checkpoint every n steps
        save_dir: Optional; Directory path for saving checkpoints
        preconditioner: Gradient preconditioner type ("none" or "rmsprop")
        alpha_rmsprop: RMSprop momentum parameter (used if preconditioner="rmsprop")
        lambda_rmsprop: RMSprop stability parameter (used if preconditioner="rmsprop")
    
    Reference: https://arxiv.org/pdf/2308.12108#page=20
    """

    eps: float = 1
    gamma: float = 1
    gamma_prior: float = 0
    nbeta: float = 1
    num_steps: int = 1
    batch_size: int = 128
    checkpoint_every: Optional[int] = None
    save_dir: Optional[str] = None
    preconditioner: str = "none"
    alpha_rmsprop: float = 0.99
    lambda_rmsprop: float = 1e-5

    def __post_init__(self):
        """Validate parameters after initialization."""
        if self.preconditioner not in ["none", "rmsprop"]:
            raise ValueError("preconditioner must be either 'none' or 'rmsprop'")
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


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

def sgld(
    model,
    sgld_params: SGLDParams,
    device,
    dataset: List | t.utils.data.DataLoader,
    cost_type: str = "cross_entropy",  # "KL" or "cross_entropy" or "zero"
    fp16: bool = False,
    mala: bool = False,  # Metropolis step, not yet implemented
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
        dtype = t.float16 if fp16 else t.float32
        dataset_list = [
            (inputs.to(device, dtype=dtype), labels.to(device, dtype=dtype)) for inputs, labels in dataset
        ]
        dataset_iter = iter_cycle(dataset_list)

    with t.no_grad():

        w_init = (
            t.nn.utils.parameters_to_vector(model.parameters())
            .detach()
            .clone()
            .to(device)
        )

        if isinstance(dataset, t.utils.data.DataLoader):
            # get logits for each batch
            logits_init = {}
            for inputs, labels in dataset_list:
                model_outputs = model(inputs)
                logits_init[inputs] = (model_outputs["logits"] if "logits" in model_outputs else model_outputs.logits)

        else:
            all_inputs, all_labels = dataset

            model_outputs = model(all_inputs)

            logits_init = (
                model_outputs["logits"]
                if "logits" in model_outputs
                else model_outputs.logits
            )

        preconditioner = t.ones_like(w_init, device=device)
        v_rmsprop = t.zeros_like(w_init, device=device)

    if save_dir is None and sgld_params.checkpoint_every:
        model_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_dir = "/mnt/ssd-1/louis/palamedes/sgld_samples/" + model_time
        os.makedirs(save_dir, exist_ok=True)
        t.save(w_init, f"{save_dir}/model_0.pt")
        print(f"Saving to {save_dir}")

    for i in tqdm(range(sgld_params.num_steps)):

        if (
            sgld_params.checkpoint_every
            and i % sgld_params.checkpoint_every == 0
            and i > 0
        ):
            t.save(w, f"{save_dir}/model_{_}.pt")

        model.zero_grad()

        # TODO define a local fn to compute cost --> exp(delta cost)

        if cost_type != "zero":
            if isinstance(dataset, t.utils.data.DataLoader):
                inputs, labels = next(dataset_iter)
            else:
                indices = t.randint(0, dataset[0].shape[0], (sgld_params.batch_size,))
                inputs, labels = dataset[0][indices], dataset[1][indices]

            model_outputs = model(inputs)
    
            logits = (
                model_outputs["logits"]
                if "logits" in model_outputs
                else model_outputs.logits
            )

            if cost_type == "KL":
                if isinstance(dataset, t.utils.data.DataLoader):
                    cost = logit_loss(logits_init[inputs], logits)
                else:
                    cost = logit_loss(logits_init[indices], logits)
            elif cost_type == "cross_entropy":
                cost = t.nn.functional.cross_entropy(logits, labels)
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
            prior_grad = -w * sgld_params.gamma_prior
            
            total_grad = (
                sgld_params.eps / 2
                * (sgld_params.nbeta * loss_grad + localization_grad + prior_grad)
            )  # we don't need to divide by batch size because cross entropy already averages over that

            total_grad = preconditioner * total_grad

            gaussian_noise = (
                t.randn_like(w, device=device)
                * t.sqrt(t.tensor(sgld_params.eps))
                * t.sqrt(preconditioner)
            )

            # okay so eps/2 * preconditioner is our equivalent of tau in the Wikipedia article.
            # which... is fine, we reject or accept according to pi and q
            # pi really requires a cost function which we don't have yet
            # q is weird and I don't understand it

            w = w + total_grad + gaussian_noise
            if fp16:
                w = w.to(dtype=t.float16)
            t.nn.utils.vector_to_parameters(w, model.parameters())

            sgld_dict["L2_norm"].append(t.norm(w).item())

            sgld_dict["L2_distance_init"].append(t.norm(w - w_init).item())

            sgld_dict["v_rmsprop"].append(v_rmsprop.median().item())

            sgld_dict["preconditioner"].append(preconditioner.median().item())

            sgld_dict["gradient_norm"].append(t.norm(loss_grad).item())

            if cost_type != "zero" and t.isnan(cost).any():
                print("Cost is NaN, returning")
                return sgld_dict
    return sgld_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SGLD sampling")
    parser.add_argument("--eps", type=float, default=1e-10, help="Total step size")
    parser.add_argument("--gprior", type=float, default=None, help="Coefficient of prior term")
    parser.add_argument("--nbeta", type=float, default=1_000_000, help="Temperature scaling (beta * dataset size)")
    parser.add_argument("--cuda", type=int, default=0, help="CUDA device index")
    parser.add_argument("--ralpha", type=float, default=0.99, help="RMSProp alpha")
    parser.add_argument("--rlambda", type=float, default=1e-4, help="RMSProp lambda")
    parser.add_argument("--num_steps", type=int, default=10_000, help="Number of steps")
    args = parser.parse_args()
    
    cuda_device = args.cuda
    data_dir = "/mnt/ssd-1/adam/basin-volume/data"

    device = t.device(f"cuda:{cuda_device}" if t.cuda.is_available() else "cpu")

    if device.type == "cpu":
        raise ValueError("CUDA device not found")

    with open(data_dir + "/cifar10_ds.pkl", 'rb') as f:
        cifar10_ds = pickle.load(f)

    clean_ds = cifar10_ds["clean"].to(device)
    clean_ds_labels = cifar10_ds["clean_labels"].to(device)

    sigma = 0.03358687
    
    # Use command line args.gprior if provided, otherwise calculate from sigma
    gamma_prior = args.gprior if args.gprior is not None else 1/sigma**2

    params = SGLDParams(
        eps=args.eps,  # total step size
        gamma=0,  # localization term
        gamma_prior=gamma_prior,  # prior term
        batch_size=4096,  # batch size
        nbeta=args.nbeta,  # this is not beta, but beta * n, where n= |whole dataset|
        num_steps=args.num_steps,  # number of steps
        checkpoint_every=None,  # If integer, saves weights every checkpoint_every steps
        save_dir=None,  # if not specified, it defaults to /mnt/ssd-1/louis/palamedes/sgld_samples/{current_time}
        # device=device,
        preconditioner="rmsprop",  # "none" or "rmsprop
        alpha_rmsprop=args.ralpha,  # rmsprop parameter
        lambda_rmsprop=args.rlambda,  # rmsprop parameter
    )

    hyparams = ['eps', 'nbeta', 'batch_size', 'gamma_prior', 'num_steps', 'alpha_rmsprop', 'lambda_rmsprop']
    hyparam_str = '_'.join([f'{k}={params.__dict__[k]:.2g}' for k in hyparams])
    print(hyparam_str)

    RUNS_DIR = "/mnt/ssd-1/adam/basin-volume/runs"

    model = load_convnext_checkpoint(
        RUNS_DIR + f"/b16pai_p001/checkpoint-{2**(16)}", device=device
    )

    sgld_dict = sgld(
        model,
        params,
        dataset=[clean_ds, clean_ds_labels],
        cost_type="KL",
        device=device,
        fp16=True,
    )

    with open(data_dir + f'/sgld_{hyparam_str}.pkl', 'wb') as f:
        pickle.dump(sgld_dict, f)