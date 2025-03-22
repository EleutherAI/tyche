from basin_volume.convnext import load_cifar10_splits, load_convnext_checkpoint
from basin_volume.estimator import VolumeConfig, VolumeEstimator
import torch as t

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


device = t.device("cuda:7") if t.cuda.is_available() else t.device("cpu")


model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-14m")
model.cuda()
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-14m")
tokenizer.pad_token_id = 1  # pythia-specific
tokenizer.eos_token_id = 0  # pythia-specific
dataset = load_dataset(
    "EleutherAI/lambada_openai", name="en", split="test", trust_remote_code=True
)


cfg = VolumeConfig(
    model=model,
    tokenizer=tokenizer,
    dataset=dataset,
    text_key="text",  # must match dataset field
    n_samples=10,  # number of MC samples
    cutoff=1e-2,  # KL-divergence cutoff (nats)
    max_seq_len=8,  # sequence length for chunking dataset
    val_size=10,  # number of sequences or chunks to use in estimation
    data_batch_size=1,
    cache_mode=None,
    chunking=True,
    implicit_vectors=False,
    debug=False,
)
estimator = VolumeEstimator.from_config(cfg)

result = estimator.run()
print(result)
