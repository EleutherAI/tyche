import argparse
import random
import string
import os
import json
import csv
import sys
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Literal, Dict, Any
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from tyche import VolumeConfig, VolumeEstimator

def generate_random_tokens(n_samples: int = 50, length: int = 50) -> Dataset:
    """Generate a dataset of random tokens."""
    data = {"text": [''.join(random.choices(string.ascii_lowercase, k=length)) 
                    for _ in range(n_samples)]}
    return Dataset.from_dict(data)

TASK_TO_DATASET = {
    "anthropic_hh": ("Anthropic/hh-rlhf", "chosen", "text"),
    "smoltalk": ("HuggingFaceTB/smoltalk", "train", "text"),
    "finemath": ("HuggingFaceTB/finemath", "train", "text"),
    "gsm8k": ("gsm8k", "train", "question"),
    "mmlu": ("cais/mmlu", "auxiliary_train", "question"),
    "winogrande": ("winogrande", "train", "sentence"),
    "truthful_qa": ("truthful_qa", "validation", "question"),
    "bbq": ("bbq", "train", "question"),
    "xsum": ("xsum", "train", "document"),
    "code_search_net": ("code_search_net", "train", "docstring"),
    "ethics": ("hendrycks/ethics", "commonsense/train", "text"),
    "flores": ("facebook/flores", "dev", "sentence"),
    "chatgpt_prompts": ("fka/awesome-chatgpt-prompts", "train", "prompt"),
    "natural_reasoning": ("facebook/natural_reasoning", "train", "text"),
    "ui_reasoning": ("smirki/UI_Reasoning_Dataset", "train", "text"),
    "lambada": ("EleutherAI/lambada_openai", "test", "text"),
    "quirky_alice": ("EleutherAI/quirky_hemisphere_alice", "test", "statement")
}

def load_task_dataset(task_name: str) -> tuple[Dataset, str]:
    """Load a dataset for a given task name."""
    if task_name not in TASK_TO_DATASET:
        raise ValueError(f"Unknown task {task_name}. Available tasks: {list(TASK_TO_DATASET.keys())}")
    
    dataset_name, split, text_key = TASK_TO_DATASET[task_name]
    try:
        dataset = load_dataset(dataset_name, split=split)
        return dataset, text_key
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset {dataset_name}: {e}")

def load_reference_dataset(ref_type: Literal["random", "pile", "task"], 
                         task_name: Optional[str] = None,
                         n_samples: int = 50) -> tuple[Dataset, str]:
    """Load a reference dataset."""
    if ref_type == "random":
        return generate_random_tokens(n_samples), "text"
    elif ref_type == "pile":
        dataset = load_dataset("EleutherAI/pile", split="train", streaming=True)
        # Take first n_samples examples
        data = {"text": []}
        for item in dataset:
            if len(data["text"]) >= n_samples:
                break
            data["text"].append(item["text"])
        return Dataset.from_dict(data), "text"
    elif ref_type == "task":
        if task_name is None:
            raise ValueError("task_name must be provided when ref_type is 'task'")
        return load_task_dataset(task_name)
    else:
        raise ValueError(f"Unknown reference type {ref_type}")

PYTHIA_MODELS = {
    "14m": "EleutherAI/pythia-14m",
    "160m": "EleutherAI/pythia-160m",
    "1.4b": "EleutherAI/pythia-1.4b"
}

SMOLLM2_MODELS = {
    "135m": "HuggingFaceTB/SmolLM2-135M",
    "360m": "HuggingFaceTB/SmolLM2-360M",
    "1.7b": "HuggingFaceTB/SmolLM2-1.7B",
    # Instruct versions
    "135m-instruct": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "360m-instruct": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "1.7b-instruct": "HuggingFaceTB/SmolLM2-1.7B-Instruct"
}

def get_model_path(model_family: str, model_size: str) -> str:
    """Get the full model path for a given family and size."""
    if model_family == "pythia":
        if model_size not in PYTHIA_MODELS:
            raise ValueError(f"Unknown Pythia model size: {model_size}. Available sizes: {list(PYTHIA_MODELS.keys())}")
        return PYTHIA_MODELS[model_size]
    elif model_family == "smollm2":
        if model_size not in SMOLLM2_MODELS:
            raise ValueError(f"Unknown SmolLM2 model size: {model_size}. Available sizes: {list(SMOLLM2_MODELS.keys())}")
        return SMOLLM2_MODELS[model_size]
    else:
        raise ValueError(f"Unknown model family: {model_family}")

def get_git_hash() -> str:
    """Get the current git commit hash."""
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                              capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        raise RuntimeError("Failed to get git hash. Make sure you're in a git repository.")

def check_git_status() -> None:
    """Check if there are uncommitted changes."""
    result = subprocess.run(['git', 'status', '--porcelain'], 
                          capture_output=True, text=True, check=True)
    if result.stdout.strip():
        raise RuntimeError("There are uncommitted changes. Please commit before running experiments.")

def save_experiment_result(result: Dict[str, Any], args: argparse.Namespace) -> str:
    """Save detailed experiment result to a file and return the filepath."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_str = f"{args.model_family}_{args.model_size}"
    tasks_str = f"{args.task1}_{args.task2}"
    filename = f"{timestamp}_{model_str}_{tasks_str}.json"
    
    filepath = os.path.join("dependence_results", "detailed_results", filename)
    
    # Convert result object to dictionary
    result_dict = {
        "estimates": {k: v.tolist() for k, v in result.estimates.items()},
        "props": {k: v.tolist() for k, v in result.props.items()},
        "mults": {k: v.tolist() for k, v in result.mults.items()},
        "deltas": {k: v.tolist() for k, v in result.deltas.items()},
        "gaussint": {k: v.tolist() for k, v in result.gaussint.items()}
    }
    
    with open(filepath, 'w') as f:
        json.dump(result_dict, f)
    
    return filepath

def log_experiment(args: argparse.Namespace, result: Dict[str, Any], 
                  result_filepath: str, command: str) -> None:
    """Log experiment details to the CSV file."""
    log_file = os.path.join("dependence_results", "dependence_experiment_logs.csv")
    file_exists = os.path.exists(log_file)
    
    # Calculate metrics
    joint_ref = (result.estimates['joint'] - result.estimates['ref']).mean().item()
    marginal_ref = (result.estimates['marginal1'] + result.estimates['marginal2'] - 
                   2 * result.estimates['ref']).mean().item()
    min_marginal = min(result.estimates['marginal1'].mean().item(), 
                      result.estimates['marginal2'].mean().item())
    normalized_dependence = 1 - (joint_ref - marginal_ref)/(marginal_ref - min_marginal)
    
    # Prepare row data
    row = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'git_hash': get_git_hash(),
        'model_family': args.model_family,
        'model_size': args.model_size,
        'task1': args.task1,
        'task2': args.task2,
        'ref_type': args.ref_type,
        'ref_task': args.ref_task,
        'n_samples': args.n_samples,
        'val_size': args.val_size,
        'cache_mode': args.cache_mode,
        'use_preconditioner': args.use_preconditioner,
        'preconditioner_eps': args.preconditioner_eps,
        'preconditioner_exp': args.preconditioner_exp,
        'checkpoint_step': args.checkpoint_step,
        'command': command,
        'result_file': result_filepath,
        'joint_ref': joint_ref,
        'marginal_ref': marginal_ref,
        'min_marginal': min_marginal,
        'normalized_dependence': normalized_dependence
    }
    
    # Write to CSV
    with open(log_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def main():
    parser = argparse.ArgumentParser(description="Measure task dependence between two tasks")
    
    # Model configuration
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument("--model-family", choices=["pythia", "smollm2"], required=True,
                          help="Model family to use")
    model_group.add_argument("--model-size", required=True,
                          help=f"Model size. Pythia: {list(PYTHIA_MODELS.keys())}, SmolLM2: {list(SMOLLM2_MODELS.keys())}")
    model_group.add_argument("--checkpoint-step", type=int,
                          help="Checkpoint step (for Pythia models only)")
    
    # Task configuration  
    task_group = parser.add_argument_group("Task Configuration")
    task_group.add_argument("--task1", required=True, choices=list(TASK_TO_DATASET.keys()),
                         help="First task to measure")
    task_group.add_argument("--task2", required=True, choices=list(TASK_TO_DATASET.keys()),
                         help="Second task to measure")
    
    # Reference dataset configuration
    ref_group = parser.add_argument_group("Reference Dataset")
    ref_group.add_argument("--ref-type", choices=["random", "pile", "task"], default="random",
                        help="Type of reference dataset to use")
    ref_group.add_argument("--ref-task", choices=list(TASK_TO_DATASET.keys()),
                        help="Reference task (if ref-type is 'task')")
    
    # Estimation parameters
    est_group = parser.add_argument_group("Estimation Parameters")
    est_group.add_argument("--n-samples", type=int, default=50,
                        help="Number of MC samples for volume estimation")
    est_group.add_argument("--val-size", type=int, default=10,
                        help="Number of validation examples to use from each dataset")
    est_group.add_argument("--cache-mode", choices=[None, "cpu", "gpu"], default=None,
                        help="Cache mode for probability computations")
    
    # Pythia-specific parameters
    pythia_group = parser.add_argument_group("Pythia-specific Parameters")
    pythia_group.add_argument("--use-preconditioner", action="store_true",
                           help="Use ADAM preconditioner (Pythia only)")
    pythia_group.add_argument("--preconditioner-eps", type=float, default=1e-5,
                           help="Epsilon for ADAM preconditioner")
    pythia_group.add_argument("--preconditioner-exp", type=float, default=0.5,
                           help="Exponent for ADAM preconditioner")

    misc_group = parser.add_argument_group("Miscellaneous")
    misc_group.add_argument("--ignore-commit-check", action="store_true",
                           help="Ignore git commit check")
    
    args = parser.parse_args()
    
    # Check git status
    if not args.ignore_commit_check:
        check_git_status()
    
    # Record the command used
    command = f"python {' '.join(sys.argv)}"
    
    # Get model path and load model
    try:
        model_path = get_model_path(args.model_family, args.model_size)
    except ValueError as e:
        parser.error(str(e))
        
    print(f"Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Set pythia-specific tokenizer settings
    if args.model_family == "pythia":
        tokenizer.pad_token_id = 1
        tokenizer.eos_token_id = 0
    
    # Load datasets
    print(f"Loading task datasets")
    dataset1, text_key1 = load_task_dataset(args.task1)
    dataset2, text_key2 = load_task_dataset(args.task2)
    ref_dataset, text_key_ref = load_reference_dataset(args.ref_type, args.ref_task)
    
    model_type = 'causal' if args.model_family == 'smollm2' else 'pythia'
    implicit_vectors = model_type == 'causal'
    # Configure estimator
    cfg = VolumeConfig(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset1,
        dataset2=dataset2,
        dataset_ref=ref_dataset,
        text_key=text_key1,
        text_key2=text_key2,
        text_key_ref=text_key_ref,
        n_samples=args.n_samples,
        val_size=args.val_size,
        cache_mode=args.cache_mode,
        model_type=model_type,
        checkpoint_step=args.checkpoint_step if args.model_family == "pythia" else None,
        preconditioner_type="adam" if args.use_preconditioner and args.model_family == "pythia" else None,
        preconditioner_eps=args.preconditioner_eps,
        preconditioner_exponent=args.preconditioner_exp,
        implicit_vectors=implicit_vectors,
        data_batch_size=1,
        tol=1
    )
    
    # Run estimation
    print("Running dependence estimation...")
    estimator = VolumeEstimator.from_config(cfg)
    result = estimator.run()
    
    # Save detailed results and log experiment
    result_filepath = save_experiment_result(result, args)
    log_experiment(args, result, result_filepath, command)
    
    # Print results
    print("\nResults:")
    print(f"Joint volume: {result.estimates['joint'].mean():.2f}")
    print(f"Marginal volumes: {result.estimates['marginal1'].mean():.2f}, {result.estimates['marginal2'].mean():.2f}")
    print(f"Reference volume: {result.estimates['ref'].mean():.2f}")
    
    # Calculate dependence measures
    joint_ref = (result.estimates['joint'] - result.estimates['ref']).mean()
    marginal_ref = (result.estimates['marginal1'] + result.estimates['marginal2'] - 2 * result.estimates['ref']).mean()
    min_marginal = min(result.estimates['marginal1'].mean(), result.estimates['marginal2'].mean())
    normalized_dependence = 1 - (joint_ref - marginal_ref)/(marginal_ref - min_marginal)
    
    print("\nDependence measures:")
    print(f"Joint - Ref: {joint_ref:.2f}")
    print(f"(Marginal1 + Marginal2 - 2*Ref): {marginal_ref:.2f}")
    print(f"Normalized dependence: {normalized_dependence:.2f}")

if __name__ == "__main__":
    main() 