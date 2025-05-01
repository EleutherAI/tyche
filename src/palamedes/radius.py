import copy
from matplotlib import pyplot as plt
import torch as t
from jaxtyping import Float
from tqdm import tqdm
from tyche.convnext import load_convnext_checkpoint


@t.no_grad()
def plot_direction_cost(
    model,
    center,
    direction,
    mult,
    cost_fn,
    device,
    cutoff: Float,
    dataset: Float[t.Tensor, "batch_data 3 d_image d_image"],
    steps: int = 100,
    plot=True,
):
    sample_mistake = False

    RUNS_DIR = "/mnt/ssd-1/adam/basin-volume/runs"

    # copy model
    model_init = copy.deepcopy(model).to(device)

    model_interpolate = copy.deepcopy(model).to(device)

    t.nn.utils.vector_to_parameters(center, model_init.parameters())

    logits_center = model_init(dataset)["logits"]

    cost_list = []
    for i in range(steps):

        t.nn.utils.vector_to_parameters(
            center + direction * mult * i / steps, model_interpolate.parameters()
        )
        logits_directions = model_interpolate(dataset)["logits"]

        cost = cost_fn(logits_center, logits_directions)

        if cost > cutoff and i < steps - 10:
            sample_mistake = True

        cost_list.append(cost.cpu().numpy())
    if plot or sample_mistake:
        plt.plot(cost_list)
        plt.xlabel("Steps")
        plt.ylabel("Cost")
        plt.title("Cost vs Steps")
        plt.show()
    if sample_mistake:
        return direction, center, mult
    else:
        return 1
