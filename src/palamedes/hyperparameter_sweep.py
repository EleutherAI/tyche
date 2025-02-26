import datetime
import json
import os
from palamedes.sgld_sampler import SGLDParams, sgld
import torch as t
from tyche.convnext import load_convnext_checkpoint
from tyche.convnext import load_cifar10_splits_dict


RUNS_DIR = "/mnt/ssd-1/adam/basin-volume/runs"

RUNS_DIR_sgld = "/mnt/ssd-1/louis/palamedes/sgld_samples"


device = t.device("cuda:5" if t.cuda.is_available() else "cpu")


# load model
models_clean = [
    load_convnext_checkpoint(
        RUNS_DIR + f"/b16pai_p001/checkpoint-{2**(step+1)}", device=device
    )
    for step in range(16)
]

# load data
cifar10_ds = load_cifar10_splits_dict(size=10000, device=device, seed=42)
val_ds = cifar10_ds["val"].to(device)
clean_ds = cifar10_ds["clean"].to(device)
poisoned_ds = cifar10_ds["poison"].to(device)
val_ds_labels = cifar10_ds["val_labels"].to(device)
clean_ds_labels = cifar10_ds["clean_labels"].to(device)
poisoned_ds_labels = cifar10_ds["poison_labels"].to(device)
print("Data and model loaded")

model = models_clean[-1]


def sweep(model, dataset, device, hyperparameters_sweep, chains=1):
    model_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(model_time)
    # save hyperparameters dict to  f"{RUNS_DIR_sgld}/{model_time}/hyperparameters.json"
    # Create directory if it doesn't exist
    os.makedirs(f"{RUNS_DIR_sgld}/{model_time}", exist_ok=True)

    # Save hyperparameters
    with open(f"{RUNS_DIR_sgld}/{model_time}/hyperparameters.json", "w") as f:
        json.dump(hyperparameters_sweep, f)
        print("Hyperparameters saved")

    for b in hyperparameters_sweep["batch_size"]:
        for n in hyperparameters_sweep["nbeta"]:
            for e in hyperparameters_sweep["eps"]:
                for s in hyperparameters_sweep["num_steps"]:
                    for g in hyperparameters_sweep["gamma"]:
                        for chain in range(chains):
                            sgld_params = SGLDParams(
                                eps=e,
                                nbeta=n,
                                batch_size=b,
                                num_steps=s,
                                gamma=g,
                            )
                            sgld_dict = sgld(
                                model=model,
                                sgld_params=sgld_params,
                                device=device,
                                dataset=dataset,
                                cost_fn="cross_entropy",  # "KL" or "cross_entropy"
                            )

                            # Create the directory path
                            save_dir = f"{RUNS_DIR_sgld}/{model_time}/sgld_eps{e}_nbeta{n}_batch{b}_steps{s}_gamma{g}"

                            # Check if directory exists, if not create it
                            if not os.path.exists(save_dir):
                                os.makedirs(save_dir, exist_ok=True)

                            # Save the file
                            t.save(
                                sgld_dict,
                                f"{save_dir}/{chain}_sgld_dict.pth",
                            )

    print(f"Saved SGLD samples to {RUNS_DIR_sgld}/{model_time}")


if __name__ == "__main__":
    hyperparameters_sweep = {
        "eps": [1e-6, 3e-6, 6e-6],
        "nbeta": [1e4],
        "batch_size": [512],
        "gamma": [0, 1],
        "num_steps": [1000],
    }
    sweep(
        model=model,
        dataset=[clean_ds, clean_ds_labels],
        hyperparameters_sweep=hyperparameters_sweep,
        chains=2,
    )
