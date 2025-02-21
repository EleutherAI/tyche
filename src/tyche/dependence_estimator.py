import torch
from dataclasses import dataclass
from typing import Optional, Dict
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from .estimator import VolumeConfig, VolumeEstimator
from .volume import get_estimates_vectorized_gauss
from .data import chunk_and_tokenize
from .vectors import ImplicitParamVector

@dataclass
class DependenceVolumeConfig(VolumeConfig):
    dataset2: Optional[Dataset] = None
    text_key2: Optional[str] = None
    val_size2: Optional[int] = None
    dataset_ref: Optional[Dataset] = None
    text_key_ref: Optional[str] = None
    val_size_ref: Optional[int] = None
    scale_ref: float = 0.8

@dataclass
class DependenceResult:
    """
    Results of dependence estimation. Each dict has four keys:
     - marginal1: marginal estimate for dataset1 and reference dataset
     - marginal2: marginal estimate for dataset2 and reference dataset
     - joint: joint estimate for both datasets and reference dataset
     - reference: marginal estimate for reference dataset alone

    Attributes:
        estimates: The estimated log-probability.
        props: Lengths of proposal vectors.
        mults: Multipliers (relative to `props`) for the basin radius.
        deltas: Difference between fn at basin edge and center.
        gaussint: Log of Gaussian integral term.
    """
    estimates: Dict[str, torch.Tensor]
    props: Dict[str, torch.Tensor]
    mults: Dict[str, torch.Tensor]
    deltas: Dict[str, torch.Tensor]
    gaussint: Dict[str, torch.Tensor]

class DependenceEstimator(VolumeEstimator):
    def __init__(self, config: DependenceVolumeConfig):
        super().__init__(config)
        self.set_defaults()

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
        if self.config.dataset2 is None:
            raise ValueError("dataset2 must be provided for dependence estimation")
        if self.config.text_key2 is None:
            self.config.text_key2 = self.config.text_key
        if self.config.val_size2 is None:
            self.config.val_size2 = self.config.val_size
        if self.config.text_key_ref is None:
            self.config.text_key_ref = self.config.text_key
        if self.config.val_size_ref is None:
            self.config.val_size_ref = self.config.val_size
        if self.config.dataset_ref is None:
            raise ValueError("dataset_ref must be provided for dependence estimation")

    def _prepare_dataset(self, dataset, text_key: str, val_size: int):
        if self.config.chunking:
            tokens = chunk_and_tokenize(
                dataset, self.tokenizer, max_seq_len=self.config.max_seq_len, text_key=text_key
            )["input_ids"]
        else:
            tokens = self.tokenizer(
                dataset[text_key],
                padding=True,
                truncation=True,
                max_length=self.config.max_seq_len,
                return_tensors="pt"
            )["input_ids"]
        tokens = tokens[:val_size]
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
        self.dataset1 = self.config.dataset
        self.dataset2 = self.config.dataset2
        self.dataset_ref = self.config.dataset_ref
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

        self.val_data1, self.probs_p1 = self._prepare_dataset(self.dataset1, self.config.text_key, self.config.val_size)
        self.val_data2, self.probs_p2 = self._prepare_dataset(self.dataset2, self.config.text_key2, self.config.val_size2)
        self.val_data_ref, self.probs_p_ref = self._prepare_dataset(self.dataset_ref, self.config.text_key_ref, self.config.val_size_ref)

        def kl_fn_factory(val_data, probs_p, multiplier=1):
            def kl_fn(a, b):
                params_q = a + b if not self.config.implicit_vectors else a
                kl_sum = 0.0
                count = 0
                for i in range(0, val_data.shape[0], self.config.data_batch_size):
                    seqs = val_data[i:i+self.config.data_batch_size]
                    if self.config.implicit_vectors:
                        if b is not None:
                            a.add_(b)
                            logits_q = self.apply_fn(a, seqs)
                            a.sub_(b)
                        else:
                            logits_q = self.apply_fn(a, seqs)
                    else:
                        logits_q = self.apply_fn(params_q, seqs)
                    logprobs_q = torch.nn.functional.log_softmax(logits_q, dim=-1)
                    if self.config.cache_mode is None:
                        logits_p = self.apply_fn(self.params, seqs)
                        probs_p_seq = torch.nn.functional.softmax(logits_p, dim=-1)
                    elif self.config.cache_mode == "cpu":
                        probs_p_seq = probs_p[i:i+self.config.data_batch_size].to("cuda")
                    elif self.config.cache_mode == "gpu":
                        probs_p_seq = probs_p[i:i+self.config.data_batch_size]
                    else:
                        raise ValueError(f"Invalid cache mode: {self.config.cache_mode}")
                    kl_seq = torch.nn.functional.kl_div(logprobs_q, probs_p_seq, reduction="none").sum(dim=-1)
                    mask = seqs != self.tokenizer.pad_token_id
                    kl_sum += torch.sum(kl_seq[mask])
                    count += torch.sum(mask)
                kl_term = kl_sum / count
                l2_term = 0
                if self.config.l2_reg:
                    b_sq = (b @ b) if b is not None else 0
                    l2_term = 0.5 * self.config.l2_reg * b_sq
                return kl_term * multiplier + l2_term
            return kl_fn

        self.kl_fn_ref = kl_fn_factory(self.val_data_ref, self.probs_p_ref, multiplier=self.config.scale_ref)

        def kl_fn1(a, b):
            kl_fn1 = kl_fn_factory(self.val_data1, self.probs_p1)
            return torch.maximum(kl_fn1(a, b), self.kl_fn_ref(a, b))
        self.kl_fn1_ref = kl_fn1

        def kl_fn2(a, b):
            kl_fn2 = kl_fn_factory(self.val_data2, self.probs_p2)
            return torch.maximum(kl_fn2(a, b), self.kl_fn_ref(a, b))
        self.kl_fn2_ref = kl_fn2

        def kl_fn_joint(a, b):
            loss1 = self.kl_fn1_ref(a, b)
            loss2 = self.kl_fn2_ref(a, b)
            return torch.maximum(loss1, loss2)
        self.kl_fn_joint = kl_fn_joint

        self.kl_fns = {
            'joint': self.kl_fn_joint,
            'marginal1': self.kl_fn1_ref,
            'marginal2': self.kl_fn2_ref,
            'ref': self.kl_fn_ref,
        }

    def load_adam_vector(self):
        raise NotImplementedError("DependenceEstimator does not support ADAM preconditioning")

    @torch.inference_mode()
    def run(self) -> DependenceResult:
        if self.config.sigma is None:
            self.config.sigma = torch.sqrt((self.params @ self.params) / self.params.shape[0])
        if self.config.debug:
            print(f"sigma = {self.config.sigma}")
        results = {}
        for k, fn in self.kl_fns.items():
            print(f"Estimating {k} dependence volume")
            results[k] = get_estimates_vectorized_gauss(
                n=self.config.n_samples,
                batch_size=self.config.model_batch_size,
                sigma=self.config.sigma,
                preconditioner=self.preconditioner,
                fn=fn,
                params=self.params,
                tol=self.config.tol,
                y_tol=self.config.y_tol,
                seed=self.config.seed, # Same seed so we test the same perturbations for each volume
                cutoff=self.config.cutoff,
                with_tqdm=self.config.tqdm,
                debug=self.config.debug,
            )

        return DependenceResult(
            estimates = {k: results[k].estimates for k in self.kl_fns.keys()},
            props = {k: results[k].props for k in self.kl_fns.keys()},
            mults = {k: results[k].mults for k in self.kl_fns.keys()},
            deltas = {k: results[k].deltas for k in self.kl_fns.keys()},
            gaussint = {k: results[k].gaussint for k in self.kl_fns.keys()},
        )