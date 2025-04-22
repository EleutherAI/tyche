from dataclasses import dataclass
import datetime
import itertools
import einops
from torch import nn
import torch as t
from typing import Callable, List, Tuple, Union
from jaxtyping import Float, Int


@dataclass
class MLPConfig:
    activation: nn.Module = t.nn.ReLU()
    N: Int = 113
    embed_dimension: Int = 24
    linear_dimension: Int = 48

    intermediate: str = (
        "pure"  # pure, real_multiplication, complex_multiplication, quaterionic_multiplication
    )
    embedding_tied: bool = False
    unembedding_tied: bool = False
    bias_unembed: bool = False
    num_additional_layers: Int = 0
    dimensions: Union[Tuple[Int], Int] = ()
    bias_layer: bool = False
    W_amplitude: float = 1.0
    weight_mode: str = "uniform"  # uniform, xavier_uniform, normal, xavier_normal
    device: Union[str, t.device] = "cpu"

    def __post_init__(self):
        if isinstance(self.dimensions, int):
            self.dimensions = (self.dimensions,) * (self.num_additional_layers)
        if len(self.dimensions) == 1:
            self.dimensions = self.dimensions * (self.num_additional_layers)
        assert (
            len(self.dimensions) == self.num_additional_layers
        ), "Dimensions must be of length num_additional_layers"


class MLP_VARIANTS(t.nn.Module):
    def __init__(self, params: MLPConfig):
        super().__init__()

        self.path_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.params = params
        self.N = params.N
        self.embed_dimension = params.embed_dimension
        self.linear_dimension = params.linear_dimension
        self.epoch = 0
        self.device = params.device
        t.set_default_device(self.device)

        layers = []
        # Init all weights
        self.embedding_left = t.nn.Linear(self.N, self.embed_dimension, bias=False)

        self.embedding_right = t.nn.Linear(self.N, self.embed_dimension, bias=False)

        self.linear_left = t.nn.Linear(
            self.embed_dimension, self.linear_dimension, bias=False
        )
        self.linear_right = t.nn.Linear(
            self.embed_dimension, self.linear_dimension, bias=False
        )

        d_unembed = (
            self.linear_dimension
            if params.intermediate == "pure"
            else self.embed_dimension
        )

        self.unembedding = t.nn.Linear(d_unembed, self.N, bias=params.bias_unembed)

        for i in range(params.num_additional_layers):
            if i == 0:
                layers.append(
                    t.nn.Linear(
                        self.linear_dimension,
                        params.dimensions[i],
                        bias=self.params.bias_layer,
                    )
                )
                layers.append(self.params.activation)
            elif i == params.num_additional_layers - 1:
                layers.append(
                    t.nn.Linear(
                        params.dimensions[i - 1],
                        d_unembed,
                        bias=self.params.bias_layer,
                    )
                )
                layers.append(self.params.activation)
            else:
                layers.append(
                    t.nn.Linear(
                        self.params.dimensions[i - 1],
                        self.params.dimensions[i],
                        bias=self.params.bias_layer,
                    )
                )
                layers.append(self.params.activation)

        self.linear_stack = nn.Sequential(*layers)

        if params.embedding_tied:
            self.embedding_right.weight = self.embedding_left.weight

        if params.unembedding_tied:

            assert (
                d_unembed == self.embed_dimension
            ), "Unembedding can only be tied if dimensions match"

            # warn if params.embedding_tied is False
            if not params.embedding_tied:
                print(
                    "Warning: unembedding tied (to left embedding) but embeddins are not tied"
                )

    def forward(
        self, a: Union[Int[t.Tensor, "batch_size 2"], Float[t.Tensor, "batch_size 2N"]]
    ) -> Float[t.Tensor, "batch_size d_vocab"]:

        if a.shape[-1] == 2:
            a_1, a_2 = a[..., 0], a[..., 1]
            input_1 = (
                t.nn.functional.one_hot(a_1, num_classes=self.N)
                .to(device=a_1.device)
                .to(dtype=t.float32)
            )
            input_2 = (
                t.nn.functional.one_hot(a_2, num_classes=self.N)
                .to(device=a_2.device)
                .to(dtype=t.float32)
            )
        elif a.shape[-1] == 2 * self.N:
            input_1 = a[..., : self.N]
            input_2 = a[..., self.N :]
        else:
            raise ValueError(
                f"Input shape {a.shape} is not supported. Expected shape (batch_size, 2) or (batch_size, 2 * N)."
            )

        x_1 = self.embedding_left(input_1)  # shape (batch_size, embedding)
        x_2 = self.embedding_right(input_2)

        intermediate = self.intermediate_layer(
            x_1, x_2
        )  # shape (batch_size, embedding)

        if self.params.unembedding_tied:
            out = intermediate @ self.embedding_left.weight.data.T
        else:
            out = self.unembedding(intermediate)

        return out

    def intermediate_layer(
        self,
        x: Float[t.Tensor, "batch_size embed_dim"],
        y: Float[t.Tensor, "batch_size embed_dim"],
    ) -> Float[t.Tensor, "batch_size embedding"]:

        d = self.embed_dimension

        if self.params.intermediate == "pure":

            hidden = self.linear_left(x) + self.linear_right(y)

            hidden_post_act = self.params.activation(hidden)
            hidden_post_act = self.linear_stack(hidden_post_act)

            out = hidden_post_act

        if self.params.intermediate == "real_multiplication":
            real_prod = x * y  # shape (batch_size, embedding)
            return real_prod

        if self.params.intermediate == "complex_multiplication":
            assert (
                d % 2 == 0
            ), "Embedding dimension must be even for complex multiplication"

            x_complex = einops.rearrange(
                x, "batch (embed_dim c) -> batch embed_dim c", c=2
            )
            y_complex = einops.rearrange(
                y, "batch (embed_dim c) -> batch embed_dim c", c=2
            )
            complex_prod = t.zeros_like(x_complex)

            complex_prod[..., 0] = (
                x_complex[..., 0] * y_complex[..., 0]
                - x_complex[..., 1] * y_complex[..., 1]
            )

            complex_prod[..., 1] = (
                x_complex[..., 0] * y_complex[..., 1]
                + x_complex[..., 1] * y_complex[..., 0]
            )

            # reshape back to original shape

            complex_prod_flat = einops.rearrange(
                complex_prod, "batch embed_dim c -> batch (embed_dim c)"
            )

            return complex_prod_flat

        if self.params.intermediate == "quaterionic_multiplication":
            assert (
                d % 4 == 0
            ), "Embedding dimension must be divisible by 4 for quaterionic multiplication"
            x_quat = einops.rearrange(
                x, "batch (embed_dim c) -> batch embed_dim c", c=4
            )
            y_quat = einops.rearrange(
                y, "batch (embed_dim c) -> batch embed_dim c", c=4
            )
            quat_prod = t.zeros_like(x_quat)
            # TODO: check this again
            quat_prod[..., 0] = (
                x_quat[..., 0] * y_quat[..., 0]
                - x_quat[..., 1] * y_quat[..., 1]
                - x_quat[..., 2] * y_quat[..., 2]
                - x_quat[..., 3] * y_quat[..., 3]
            )
            quat_prod[..., 1] = (
                x_quat[..., 0] * y_quat[..., 1]
                + x_quat[..., 1] * y_quat[..., 0]
                + x_quat[..., 2] * y_quat[..., 3]
                - x_quat[..., 3] * y_quat[..., 2]
            )
            quat_prod[..., 2] = (
                x_quat[..., 0] * y_quat[..., 2]
                - x_quat[..., 1] * y_quat[..., 3]
                + x_quat[..., 2] * y_quat[..., 0]
                + x_quat[..., 3] * y_quat[..., 1]
            )
            quat_prod[..., 3] = (
                x_quat[..., 0] * y_quat[..., 3]
                + x_quat[..., 1] * y_quat[..., 2]
                - x_quat[..., 2] * y_quat[..., 1]
                + x_quat[..., 3] * y_quat[..., 0]
            )

            # reshape back to original shape

            quat_prod_flat = einops.rearrange(
                quat_prod, "batch embed_dim c -> batch (embed_dim c)"
            )

            return quat_prod_flat

        return out

    def initialize_weights(self):

        if self.params.weight_mode == "uniform":
            init_fn = lambda tensor: t.empty_like(tensor, dtype=t.float32).uniform_(
                -self.params.W_amplitude, self.params.W_amplitude
            )
        elif self.params.weight_mode == "xavier_uniform":
            init_fn = lambda tensor: nn.init.xavier_uniform_(
                t.empty_like(tensor, dtype=t.float32), gain=self.params.W_amplitude
            )
        elif self.params.weight_mode == "normal":
            init_fn = lambda tensor: t.empty_like(tensor, dtype=t.float32).normal_(
                mean=0, std=self.params.W_amplitude
            )
        elif self.params.weight_mode == "xavier_normal":
            init_fn = lambda tensor: nn.init.xavier_normal_(
                t.empty_like(tensor, dtype=t.float32), gain=self.params.W_amplitude
            )
        elif self.params.weight_mode == "none":
            return

        # Initialize weights and biases for all linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module.weight is not None:
                    module.weight.data = init_fn(module.weight.data)
                if module.bias is not None:
                    module.bias.data = init_fn(module.bias.data)

    @t.no_grad()
    def test_loss_and_acc(self) -> dict[str, Float[t.Tensor, "instance"]]:
        """Create all possible pairs (x,y) and return loss and accuracy for all groups in group_dataset."""
        N = self.params.N
        device = self.embedding_left.weight.device
        test_inputs = t.tensor(
            list(itertools.product(range(N), repeat=2)), device=device
        )

        test_labels = test_inputs.sum(dim=1) % N
        model_pred = self.forward(test_inputs)
        loss = t.nn.functional.cross_entropy(model_pred, test_labels)
        accuracy = (model_pred.argmax(dim=-1) == test_labels).float().mean()

        return {"loss": loss, "accuracy": accuracy}
