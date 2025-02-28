import gc
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union, Literal
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
import itertools

from .data import chunk_and_tokenize
from .volume import get_estimates_vectorized_gauss, VolumeResult
from .precondition import matrix_preconditioner, diag_preconditioner
from .utils import BASIN_VOLUME_DIR, list_largest_tensors, get_strings_and_tokenize
from .pythia import *
from .convnext import (
    load_convnext_checkpoint,
    load_cifar10_splits,
    get_convnext_logits,
    load_convnext_adam_vectors
)
from .vectors import ImplicitVector, ImplicitParamVector, ImplicitRandomVector

@dataclass
class VolumeConfig:
    # Common parameters
    n_samples: int = 100
    model_batch_size: Optional[int] = None
    sigma: Optional[float] = None  # If None, compute from params
    l2_reg: float = 0.0
    cutoff: float = 1e-2
    tol: float = 1e-2
    y_tol: float = 5
    seed: int = 42
    tqdm: bool = True
    debug: bool = False
    reduction: Optional[Literal["mean"]] = "mean" # Reduction over the batch dimension
    iters: int = 10

    # Model-specific parameters
    model_type: Literal["causal", "pythia", "convnext", "mlp"] = "causal"
    model_name: Optional[str] = None  # pythia size ("31m"), convnext run name, or mlp config name
    checkpoint_step: Optional[int] = None  # For pythia/convnext
    val_size: Optional[int] = None  # Number of validation datapoints
    split: Literal[None, "clean", "poison", "val"] = None  # For convnext

    # For HF models
    # Model and dataset params
    model: Optional[AutoModelForCausalLM] = None
    tokenizer: Optional[AutoTokenizer] = None
    dataset: Optional[Dataset] = None
    text_key: Optional[str] = None
    max_seq_len: Optional[int] = None
    # Cache params
    cache_mode: Literal[None, "cpu", "gpu"] = None
    chunking: bool = False
    data_batch_size: Optional[int] = None
    # Implicit vectors params
    implicit_vectors: bool = False
    block_size: int = 1024**2

    # Preconditioner params
    preconditioner_type: Literal[None, "adam"] = None
    preconditioner_eps: float = 1e-5
    preconditioner_exponent: float = 0.5
    adam_order: int = 2  # 1 for exp_avg, 2 for exp_avg_sq

    last_token_only: bool = False

    # New optional fields for multi-dataset dependence estimation
    dataset2: Optional[Dataset] = None
    text_key2: Optional[str] = None
    val_size2: Optional[int] = None
    dataset_ref: Optional[Dataset] = None
    text_key_ref: Optional[str] = None
    val_size_ref: Optional[int] = None
    scale_ref: float = 0.8

class VolumeEstimator(ABC):
    def __init__(self, config: VolumeConfig):
        self.config = config
        self.set_defaults()
        self.setup_model()
        if self.config.preconditioner_type == "adam":
            self.load_adam_vector()
        self.set_preconditioner()
        
    @abstractmethod
    def set_defaults(self):
        """Set default values for config"""
        pass
    
    @abstractmethod
    def setup_model(self):
        """Load model checkpoint and set up apply_fn"""
        pass
    
    @abstractmethod
    def load_adam_vector(self):
        """Load ADAM vector from checkpoint"""
        pass
    
    def set_preconditioner(self):
        match self.config.preconditioner_type:
            case "adam":
                match self.config.adam_order:
                    case 1:
                        adam_vector = self.adam1
                    case 2:
                        adam_vector = self.adam2
                    case _:
                        raise ValueError(f"Invalid ADAM order: {self.config.adam_order}")

                self.preconditioner = diag_preconditioner(
                    adam_vector,
                    eps=self.config.preconditioner_eps,
                    exponent=self.config.preconditioner_exponent
                )
            case None:
                self.preconditioner = None
            case _:
                raise ValueError(f"Invalid preconditioner type: {self.config.preconditioner_type}")

    @torch.inference_mode()
    def run(self) -> VolumeResult:
        if self.config.sigma is None:
            self.config.sigma = torch.sqrt((self.params @ self.params) / self.params.shape[0])
        if self.config.debug:
            print(f"sigma = {self.config.sigma}")
        multi_results = {}
        for key, kl_fn in self.kl_fns.items():
            print(f"Estimating {key} volume")
            multi_results[key] = get_estimates_vectorized_gauss(
                n=self.config.n_samples,
                batch_size=self.config.model_batch_size,
                sigma=self.config.sigma,
                preconditioner=self.preconditioner,
                fn=kl_fn,
                params=self.params,
                tol=self.config.tol,
                y_tol=self.config.y_tol,
                seed=self.config.seed,
                cutoff=self.config.cutoff,
                with_tqdm=self.config.tqdm,
                debug=self.config.debug,
                iters=self.config.iters,
            )
        return VolumeResult(
            estimates={k: multi_results[k].estimates for k in multi_results},
            props={k: multi_results[k].props for k in multi_results},
            mults={k: multi_results[k].mults for k in multi_results},
            deltas={k: multi_results[k].deltas for k in multi_results},
            gaussint={k: multi_results[k].gaussint for k in multi_results},
        )
    
    @classmethod
    def from_config(cls, config: VolumeConfig):
        if config.model_type == "pythia":
            return PythiaEstimator(config)
        elif config.model_type == "convnext":
            return ConvNextEstimator(config)
        elif config.model_type == "mlp":
            raise NotImplementedError("MLP requires JAX, see branch `jax-hybrid`")
        elif config.model_type == "causal":
            assert config.model is not None, "model must be provided for causal models"
            assert config.tokenizer is not None, "tokenizer must be provided for causal models"
            assert config.dataset is not None, "dataset must be provided for causal models"
            return CausalLMEstimator(config)
        else:
            raise ValueError(f"Invalid model type: {config.model_type}")


class CausalLMEstimator(VolumeEstimator):
    def set_defaults(self):
        if self.config.model_batch_size is None:
            self.config.model_batch_size = 1
        if self.config.data_batch_size is None:
            self.config.data_batch_size = 1
        if self.config.max_seq_len is None:
            self.config.max_seq_len = 2048
        if self.config.val_size is None:
            self.config.val_size = 10
        if self.config.text_key is None:
            self.config.text_key = "text"
        if self.config.dataset2 is not None:
            if self.config.text_key2 is None:
                self.config.text_key2 = self.config.text_key
            if self.config.val_size2 is None:
                self.config.val_size2 = self.config.val_size
        if self.config.dataset_ref is not None:
            if self.config.text_key_ref is None:
                self.config.text_key_ref = self.config.text_key
            if self.config.val_size_ref is None:
                self.config.val_size_ref = self.config.val_size

    def _prepare_dataset(self, dataset, text_key: str, val_size: int):
        # First, extract the validation subset
        if hasattr(dataset, "__len__"):
            val_dataset = dataset.select(range(min(len(dataset), val_size)))
        else:
            # For IterableDataset, take the first val_size examples
            from datasets import Dataset
            val_examples = list(itertools.islice(dataset, val_size))
            val_dataset = Dataset.from_dict({k: [ex[k] for ex in val_examples] 
                                           for k in val_examples[0].keys()})
        
        if self.config.chunking:
            tokens = chunk_and_tokenize(
                val_dataset, self.tokenizer, max_seq_len=self.config.max_seq_len, text_key=text_key
            )
            if hasattr(tokens, "__len__"):
                tokens = tokens["input_ids"]
        else:
            tokens = get_strings_and_tokenize(
                val_dataset, text_key, self.tokenizer,
                max_seq_len=self.config.max_seq_len,
                padding='max_length',
                truncation=True,
                format="torch"
            )
        
        val_data = tokens.to("cuda")
        probs_p = None
        if self.config.cache_mode:
            probs_list = []
            for i in range(0, val_data.shape[0], self.config.data_batch_size):
                seqs = val_data[i:i+self.config.data_batch_size]
                logits = self.apply_fn(self.params, seqs)
                probs = torch.nn.functional.softmax(logits, dim=-1)
                if self.config.cache_mode == "cpu":
                    probs_list.append(probs.to("cpu"))
                elif self.config.cache_mode == "gpu":
                    probs_list.append(probs)
                else:
                    raise ValueError(f"Invalid cache mode: {self.config.cache_mode}")
            probs_p = torch.cat(probs_list, dim=0)
        return val_data, probs_p

    def setup_model(self):
        self.model = self.config.model
        self.tokenizer = self.config.tokenizer
        self.dataset = self.config.dataset
        self.model.eval()
        self.model.to("cuda")

        if self.config.implicit_vectors:
            self.params = ImplicitParamVector(self.model, self.config.block_size)
        else:
            self.params = torch.nn.utils.parameters_to_vector(self.model.parameters()).detach()

        self.config.tol = self.params.shape[0] * 10 / 2**24
        self.config.y_tol = self.config.tol * 10

        def apply_fn(params, x):
            if self.config.implicit_vectors:
                assert params.module == self.model, "module must match"
            else:
                torch.nn.utils.vector_to_parameters(params, self.model.parameters())
            if hasattr(self.model, "hf_model"):
                return self.model.hf_model(x).logits.detach()
            else:
                return self.model(x).logits.detach()
        self.apply_fn = apply_fn

        # Prepare datasets and their probabilities
        self.val_data, self.probs_p = self._prepare_dataset(self.dataset, self.config.text_key, self.config.val_size)
        if self.config.dataset2 is not None:
            self.val_data2, self.probs_p2 = self._prepare_dataset(self.config.dataset2, self.config.text_key2, self.config.val_size2)
        if self.config.dataset_ref is not None:
            self.val_data_ref, self.probs_p_ref = self._prepare_dataset(self.config.dataset_ref, self.config.text_key_ref, self.config.val_size_ref)

        def kl_fn_factory(val_data, probs_p, multiplier=1):
            def kl_fn(a, b, mults=None):
                def compute_multiplier(b, mults, i):
                    if mults is None:
                        return b
                    elif mults.shape[0] == self.val_data.shape[0]:
                        return mults[i] * b
                    elif mults.shape[0] == 1:
                        return mults[0] * b
                    else:
                        raise ValueError(f"Invalid mults: {mults}")

                if self.config.implicit_vectors:
                    assert a == self.params, "a must be the same as the model parameters"
                else:
                    b = compute_multiplier(b, mults, 0)
                    params_q = a + b
                if self.config.reduction == "mean":
                    kl_sum = 0.0
                    count = 0
                elif self.config.reduction is None:
                    kl_sum = torch.zeros(self.val_data.shape[0], device=self.val_data.device)
                    count = torch.zeros(self.val_data.shape[0], device=self.val_data.device)
                else:
                    raise ValueError(f"Invalid reduction: {self.config.reduction}")
                # Process one batch at a time

                selection = slice(None) if not self.config.last_token_only else slice(-1)
                for i in range(0, self.val_data.shape[0], self.config.data_batch_size):
                    seqs = self.val_data[i:i+self.config.data_batch_size]
                    if self.config.implicit_vectors:
                        if b:
                            a.add_(compute_multiplier(b, mults, i))
                        logits_q = self.apply_fn(a, seqs)[..., selection]
                        if b:
                            a.sub_(compute_multiplier(b, mults, i))
                    else:
                        if mults and mults.shape[0] == self.val_data.shape[0]:
                            b = compute_multiplier(b, mults, i)
                            params_q = a + b
                        logits_q = self.apply_fn(params_q, seqs)[..., selection]
                    logprobs_q = torch.nn.functional.log_softmax(logits_q, dim=-1)
                    
                    if self.config.cache_mode is None:
                        logits_p = self.apply_fn(self.params, seqs)
                        probs_p_seq = torch.nn.functional.softmax(logits_p, dim=-1)
                    # Move just this batch's probs to GPU
                    elif self.config.cache_mode == "cpu":
                        probs_p_seq = self.probs_p[i:i+self.config.data_batch_size].to("cuda")
                    elif self.config.cache_mode == "gpu":
                        probs_p_seq = self.probs_p[i:i+self.config.data_batch_size]
                    else:
                        raise ValueError(f"Invalid cache mode: {self.config.cache_mode}")
                    
                    kl_seq = torch.nn.functional.kl_div(logprobs_q, probs_p_seq, reduction="none").sum(dim=-1)
                    mask = seqs != self.tokenizer.pad_token_id
                    kl_seq_masked = kl_seq[mask]
                    if self.config.reduction == "mean":
                        kl_sum += torch.sum(kl_seq_masked)
                        count += torch.sum(mask)
                    elif self.config.reduction is None:
                        kl_sum[i:i+self.config.data_batch_size] = kl_seq_masked.sum(dim=-1)
                        count[i:i+self.config.data_batch_size] = mask.sum(dim=-1)

                kl_term = kl_sum / count
                if self.config.l2_reg:
                    b_sq = b @ b if b else 0
                    l2_term = 1/2 * self.config.l2_reg * b_sq
                else:
                    l2_term = 0
                return kl_term * multiplier + l2_term
            return kl_fn

        # Set up KL functions based on available datasets
        if self.config.dataset2 is None and self.config.dataset_ref is None:
            self.kl_fns = {"marginal": kl_fn_factory(self.val_data, self.probs_p)}
        elif self.config.dataset2 is not None and self.config.dataset_ref is not None:
            kl_fn_ref = kl_fn_factory(self.val_data_ref, self.probs_p_ref, multiplier=self.config.scale_ref)
            kl_fn1 = lambda a, b: torch.maximum(kl_fn_factory(self.val_data, self.probs_p)(a, b), kl_fn_ref(a, b))
            kl_fn2 = lambda a, b: torch.maximum(kl_fn_factory(self.val_data2, self.probs_p2)(a, b), kl_fn_ref(a, b))
            kl_fn_joint = lambda a, b: torch.maximum(kl_fn1(a, b), kl_fn2(a, b))
            self.kl_fns = {"joint": kl_fn_joint, "marginal1": kl_fn1, "marginal2": kl_fn2, "ref": kl_fn_ref}
        elif self.config.dataset2 is not None:
            kl_fn1 = kl_fn_factory(self.val_data, self.probs_p)
            kl_fn2 = kl_fn_factory(self.val_data2, self.probs_p2)
            kl_fn_joint = lambda a, b: torch.maximum(kl_fn1(a, b), kl_fn2(a, b))
            self.kl_fns = {"joint": kl_fn_joint, "marginal1": kl_fn1, "marginal2": kl_fn2}
        elif self.config.dataset_ref is not None:
            kl_fn_ref = kl_fn_factory(self.val_data_ref, self.probs_p_ref, multiplier=self.config.scale_ref)
            kl_fn1 = lambda a, b: torch.maximum(kl_fn_factory(self.val_data, self.probs_p)(a, b), kl_fn_ref(a, b))
            self.kl_fns = {"marginal1": kl_fn1, "ref": kl_fn_ref}

    def load_adam_vector(self):
        raise NotImplementedError("CausalLMEstimator does not support ADAM preconditioning")


class PythiaEstimator(VolumeEstimator):
    def set_defaults(self):
        if self.config.model_name is None:
            self.config.model_name = "31m"
        if self.config.checkpoint_step is None:
            steps = get_pythia_checkpoint_steps(self.config.model_name)
            self.config.checkpoint_step = steps[-1]
        if self.config.val_size is None:
            self.config.val_size = 10
        if self.config.model_batch_size is None:
            self.config.model_batch_size = 1
        if self.config.preconditioner_eps is None:
            self.config.preconditioner_eps = 1e-5
        if self.config.preconditioner_exponent is None:
            self.config.preconditioner_exponent = 0.5
        if self.config.sigma is None:
            self.config.sigma = 0.03997834
        if self.config.val_size_ref is None:
            self.config.val_size_ref = self.config.val_size
        if self.config.val_size2 is None:
            self.config.val_size2 = self.config.val_size

    def _prepare_dataset(self, dataset, text_key: str, val_size: int):
        # First, extract the validation subset
        if hasattr(dataset, "__len__"):
            val_dataset = dataset.select(range(min(len(dataset), val_size)))
        else:
            # For IterableDataset, take the first val_size examples
            from datasets import Dataset
            val_examples = list(itertools.islice(dataset, val_size))
            val_dataset = Dataset.from_dict({k: [ex[k] for ex in val_examples] 
                                           for k in val_examples[0].keys()})
        
        if self.config.chunking:
            tokens = chunk_and_tokenize(
                val_dataset, self.tokenizer, max_seq_len=self.config.max_seq_len, text_key=text_key
            )
            if hasattr(tokens, "__len__"):
                tokens = tokens["input_ids"]
        else:
            tokens = get_strings_and_tokenize(
                val_dataset, text_key, self.tokenizer,
                max_seq_len=self.config.max_seq_len,
                padding='max_length',
                truncation=True,
                format="torch"
            )
        
        val_data = tokens.to("cuda")
        probs_p = None
        if self.config.cache_mode:
            probs_list = []
            for i in range(0, val_data.shape[0], self.config.data_batch_size):
                seqs = val_data[i:i+self.config.data_batch_size]
                logits = self.apply_fn(self.params, seqs)
                probs = torch.nn.functional.softmax(logits, dim=-1)
                if self.config.cache_mode == "cpu":
                    probs_list.append(probs.to("cpu"))
                elif self.config.cache_mode == "gpu":
                    probs_list.append(probs)
                else:
                    raise ValueError(f"Invalid cache mode: {self.config.cache_mode}")
            probs_p = torch.cat(probs_list, dim=0)
        return val_data, probs_p

    def setup_model(self):
        if self.config.implicit_vectors:
            raise NotImplementedError("Implicit vectors not yet supported for Pythia")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(f"EleutherAI/pythia-{self.config.model_name}")
        self.tokenizer.pad_token_id = 1
        self.tokenizer.eos_token_id = 0
        
        # Get model checkpoint
        self.model = load_pythia_checkpoint(self.config.checkpoint_step, self.config.model_name)
            
        # Convert params to JAX
        trained_params_t = torch.nn.utils.parameters_to_vector(self.model.parameters()).detach()
        self.params = trained_params_t
        
        # Set up apply_fn and kl_fn
        def apply_fn(params, x):
            params_t = torch.from_dlpack(params)
            torch.nn.utils.vector_to_parameters(params_t, self.model.parameters())
            return self.model(x).logits.detach()
            
        self.apply_fn = apply_fn

        # Primary validation data
        if not self.config.dataset_ref:
            self.val_data_ref = load_pythia_val_data(self.tokenizer, n_seqs=self.config.val_size)
            logits_pref = self.apply_fn(self.params, self.val_data_ref)
            probs_pref = torch.nn.functional.softmax(logits_pref, dim=-1)
        else:
            self.val_data_ref, probs_pref = self._prepare_dataset(self.config.dataset_ref, self.config.text_key_ref, self.config.val_size_ref)

        
        if self.config.dataset is not None:
            self.val_data, self.probs_p = self._prepare_dataset(self.config.dataset, self.config.text_key, self.config.val_size)
        if self.config.dataset2 is not None:
            self.val_data2, self.probs_p2 = self._prepare_dataset(self.config.dataset2, self.config.text_key2, self.config.val_size2)

        def kl_fn_factory(val_data, probs, multiplier=1):
            def kl_fn(a, b, mults=None):
                if mults:
                    if mults.shape[0] != 1:
                        raise ValueError(f"Invalid mults: {mults}; reduction = None not supported for pythia")
                    b = mults[0] * b
                if not isinstance(b, torch.Tensor):
                    b = torch.tensor(b)
                params_q = a + b
                kl_sum = 0.0
                count = 0
                for i in range(0, val_data.shape[0], self.config.data_batch_size):
                    seqs = val_data[i:i+self.config.data_batch_size]
                    logits_q = self.apply_fn(params_q, seqs)
                    logprobs_q = torch.nn.functional.log_softmax(logits_q, dim=-1)
                    if self.config.cache_mode is None:
                        logits_p = self.apply_fn(self.params, seqs)
                        probs_p_seq = torch.nn.functional.softmax(logits_p, dim=-1)
                    elif self.config.cache_mode == "cpu":
                        probs_p_seq = probs[i:i+self.config.data_batch_size].to("cuda")
                    elif self.config.cache_mode == "gpu":
                        probs_p_seq = probs[i:i+self.config.data_batch_size]
                    else:
                        raise ValueError(f"Invalid cache mode: {self.config.cache_mode}")
                    kl_seq = torch.nn.functional.kl_div(logprobs_q, probs_p_seq, reduction="none").sum(dim=-1)
                    mask = seqs != self.tokenizer.pad_token_id
                    kl_sum += torch.sum(kl_seq[mask])
                    count += torch.sum(mask)
                kl_term = kl_sum / count
                if self.config.l2_reg:
                    b_sq = b @ b if b else 0
                    l2_term = 0.5 * self.config.l2_reg * b_sq
                else:
                    l2_term = 0
                return kl_term * multiplier + l2_term
            return kl_fn

        if self.config.dataset is None and self.config.dataset2 is None:
            self.kl_fns = {"ref": kl_fn_factory(self.val_data_ref, probs_pref)}
        elif self.config.dataset is not None and self.config.dataset2 is not None:
            # Process datasets
            kl_fn_ref = kl_fn_factory(self.val_data_ref, probs_pref, multiplier=self.config.scale_ref)
            kl_fn_1_base = kl_fn_factory(self.val_data, self.probs_p)
            kl_fn_2_base = kl_fn_factory(self.val_data2, self.probs_p2)
            kl_fn1 = lambda a, b: torch.maximum(kl_fn_1_base(a, b), kl_fn_ref(a, b))
            kl_fn2 = lambda a, b: torch.maximum(kl_fn_2_base(a, b), kl_fn_ref(a, b))
            kl_fn_joint = lambda a, b: torch.maximum(kl_fn1(a, b), kl_fn2(a, b))
            self.kl_fns = {"joint": kl_fn_joint, "marginal1": kl_fn1, "marginal2": kl_fn2, "ref": kl_fn_ref}
        else:
            raise ValueError(f"Invalid dataset configuration: either both aux datasets or neither must be provided")

    def load_adam_vector(self):
        adam_states = load_pythia_checkpoint_states(self.config.checkpoint_step, self.config.model_name)
        adam1, adam2 = build_pythia_adam_vectors(self.model, adam_states)
        self.adam1 = adam1
        self.adam2 = adam2


class ConvNextEstimator(VolumeEstimator):
    def set_defaults(self):
        if self.config.model_name is None:
            self.config.model_name = "b16pai_p001"
        if self.config.checkpoint_step is None:
            self.config.checkpoint_step = 2**16
        if self.config.val_size is None:
            self.config.val_size = 1024
        if self.config.model_batch_size is None:
            self.config.model_batch_size = 1
        if self.config.preconditioner_eps is None:
            self.config.preconditioner_eps = 1e-5
        if self.config.preconditioner_exponent is None:
            self.config.preconditioner_exponent = 0.5
        if self.config.sigma is None:
            self.config.sigma = 0.03358687
        if self.config.split is None:
            self.config.split = "val"
        # Check for multiple datasets
        if self.config.dataset2 is not None or self.config.dataset_ref is not None:
            raise NotImplementedError("ConvNextEstimator does not support multiple datasets")

    def setup_model(self):
        if self.config.implicit_vectors:
            raise NotImplementedError("Implicit vectors not yet supported for ConvNext")
        
        # Load model checkpoint
        self.model = load_convnext_checkpoint(
            f"{BASIN_VOLUME_DIR}/runs/{self.config.model_name}/checkpoint-{self.config.checkpoint_step}"
        )
        
        # Convert params to JAX
        trained_params_t = torch.nn.utils.parameters_to_vector(self.model.parameters())
        trained_params_t = trained_params_t.to(torch.float32).detach()
        self.params = torch.from_dlpack(trained_params_t)
        
        # Load evaluation data
        splits = load_cifar10_splits(size=self.config.val_size)
        self.val_data = splits[self.config.split]
        
        # Set up apply_fn and kl_fn
        def apply_fn(params, x):
            params_t = torch.from_dlpack(params).to(torch.float16)
            return get_convnext_logits(params_t, x, self.model)
            
        self.apply_fn = apply_fn
        
        logits_p = self.apply_fn(self.params, self.val_data)
        probs_p = torch.nn.functional.softmax(logits_p, dim=-1)
        
        def kl_fn(a, b, mults=None):
            if mults:
                if mults.shape[0] != 1:
                    raise ValueError(f"Invalid mults: {mults}; reduction = None not supported for convnext")
                b = mults[0] * b
            params_q = a + b
            logits_q = self.apply_fn(params_q, self.val_data)
            logprobs_q = torch.nn.functional.log_softmax(logits_q, dim=-1)
            # equivalent to batchmean but written out for easy comparison
            kl_term = torch.nn.functional.kl_div(logprobs_q, probs_p, reduction="none").sum(dim=-1).mean()
            if self.config.l2_reg:
                b_sq = b @ b if b else 0
                l2_term = 1/2 * self.config.l2_reg * b_sq
            else:
                l2_term = 0
            return kl_term + l2_term
            
        self.kl_fn = kl_fn

    def load_adam_vector(self):
        adam1, adam2 = load_convnext_adam_vectors(
            self.model,
            self.config.model_name,
            self.config.checkpoint_step
        )
        self.adam1 = adam1
        self.adam2 = adam2