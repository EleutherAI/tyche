import datetime
import os
from palamedes.sgld_sampler import SGLDParams, sgld
import torch as t
from tyche.convnext import load_convnext_checkpoint
from tyche.convnext import load_cifar10_splits_dict


RUNS_DIR = "/mnt/ssd-1/adam/basin-volume/runs"

RUNS_DIR_sgld = "/mnt/ssd-1/louis/palamedes/sgld_samples"


device = t.device("cuda:7" if t.cuda.is_available() else "cpu")


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


def sweep(eps, nbeta, batch_size, num_steps, gamma, chains=1):
    model_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(model_time)

    for b in batch_size:
        for n in nbeta:
            for e in eps:
                for s in num_steps:
                    for g in gamma:
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
                                dataset=[val_ds, val_ds_labels],
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

    print(f"Saved SGLD samples to {save_dir}")


if __name__ == "__main__":

    eps = [1e-6, 1e-4, 1e-2]

    nbeta = [
        1e-3,
        1,
        1e3,
    ]
    batch_size = [10, 100, 512]
    gamma = [0, 1e-1, 1]
    num_steps = [1500]

    sweep(eps, nbeta, batch_size, num_steps, gamma=gamma, chains=3)
