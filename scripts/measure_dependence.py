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
import psutil
import gc
import tracemalloc
from functools import lru_cache

from tyche import VolumeConfig, VolumeEstimator

def generate_random_tokens(n_samples: int = 50, length: int = 50) -> Dataset:
    """Generate a dataset of random tokens."""
    data = {"text": [''.join(random.choices(string.ascii_lowercase, k=length)) 
                    for _ in range(n_samples)]}
    return Dataset.from_dict(data)

TASK_TO_DATASET = {
    "anthropic_hh": ("Anthropic/hh-rlhf", None, "train", "chosen"),
    "smoltalk": ("HuggingFaceTB/smoltalk", "everyday-conversations", "train", "messages"),
    "finemath": ("HuggingFaceTB/finemath", "finemath-3plus", "train", "text"),
    "gsm8k": ("gsm8k", "main", "train", "question"),
    "mmlu": ("cais/mmlu", None, "auxiliary_train", "question"),
    "truthful_qa": ("truthfulqa/truthful_qa", "generation", "validation", "question"),
    "bbq": ("elfsong/bbq", None, None, "question"),
    "xsum": ("EdinburghNLP/xsum", None, "train", "document"),
    "code_search_net": ("Nan-Do/code-search-net-python", None, "train", "docstring"),
    "ethics": ("hendrycks/ethics", "commonsense", "train", "input"),
    "flores": ("facebook/flores", None, "dev", "sentence"),
    "chatgpt_prompts": ("fka/awesome-chatgpt-prompts", None, "train", "prompt"),
    "natural_reasoning": ("facebook/natural_reasoning", None, "train", "question"),
    "ui_reasoning": ("smirki/UI_Reasoning_Dataset", None, "train", "question"),
    "lambada": ("EleutherAI/lambada_openai", None, "test", "text"),
    "quirky_alice": ("EleutherAI/quirky_hemisphere_alice", None, "test", "statement")
}

def log_memory_usage(stage: str):
    """Log current memory usage."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"Memory usage at {stage}: {memory_info.rss / (1024 * 1024):.2f} MB")

@lru_cache(maxsize=16)
def load_task_dataset(task_name: str) -> tuple[Dataset, str]:
    """Load a dataset for a given task name with caching."""
    log_memory_usage(f"Before loading dataset {task_name}")
    
    if task_name not in TASK_TO_DATASET:
        raise ValueError(f"Unknown task {task_name}. Available tasks: {list(TASK_TO_DATASET.keys())}")
    
    dataset_name, subset, split, text_key = TASK_TO_DATASET[task_name]
    try:
        # Special case for bbq dataset which is a DatasetDict with multiple splits
        if dataset_name == "elfsong/bbq" and split is None:
            dataset_dict = load_dataset(dataset_name)
            # Use 'age' split as default
            dataset = dataset_dict['age']
            log_memory_usage(f"After loading dataset {task_name}")
            return dataset, text_key
        
        if subset is not None:
            dataset = load_dataset(dataset_name, subset, split=split, streaming=True)
        else:
            dataset = load_dataset(dataset_name, split=split, streaming=True)
        
        log_memory_usage(f"After loading dataset {task_name}")
        return dataset, text_key
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset {dataset_name}: {e}")

@lru_cache(maxsize=8)
def load_reference_dataset(ref_type: Optional[Literal["random", "pile", "task"]], 
                         task_name: Optional[str] = None,
                         n_samples: int = 50) -> Optional[tuple[Dataset, str]]:
    """Load a reference dataset with caching."""
    log_memory_usage(f"Before loading reference dataset {ref_type}")
    
    if ref_type is None or ref_type.lower() == "none":
        return None
        
    if ref_type == "random":
        result = generate_random_tokens(n_samples), "text"
        log_memory_usage(f"After loading reference dataset {ref_type}")
        return result
    elif ref_type == "pile":
        dataset = load_dataset("EleutherAI/pile", split="train", streaming=True)
        # Take first n_samples examples
        data = {"text": []}
        for item in dataset:
            if len(data["text"]) >= n_samples:
                break
            data["text"].append(item["text"])
        result = Dataset.from_dict(data), "text"
        log_memory_usage(f"After loading reference dataset {ref_type}")
        return result
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
    min_marginal = min(result.estimates['marginal1'].mean().item() - result.estimates['ref'].mean().item(), 
                      result.estimates['marginal2'].mean().item() - result.estimates['ref'].mean().item())
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
    ref_group.add_argument("--ref-type", type=str, choices=["random", "pile", "task", "none"],
                        default="random", help="Type of reference dataset to use ('none' for no reference)")
    ref_group.add_argument("--ref-task", choices=list(TASK_TO_DATASET.keys()),
                        help="Reference task (if ref-type is 'task')")
    ref_group.add_argument("--ref-div-multiplier", type=float, default=0.75,
                           help="Multiplier for reference KL divergence (we want the joint basin to be a subset of the reference basin)")
    
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
    misc_group.add_argument("--debug-memory", action="store_true",
                           help="Enable detailed memory tracking")
    
    args = parser.parse_args()
    
    # Start memory tracking if requested
    if args.debug_memory:
        tracemalloc.start()
    
    log_memory_usage("Script start")
    
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
    
    log_memory_usage("Before loading model")
    print(f"Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    log_memory_usage("After loading model")
    
    # Set pythia-specific tokenizer settings
    if args.model_family == "pythia":
        tokenizer.pad_token_id = 1
        tokenizer.eos_token_id = 0
    
    # Load datasets
    print(f"Loading task datasets")
    dataset1, text_key1 = load_task_dataset(args.task1)
    dataset2, text_key2 = load_task_dataset(args.task2)
    
    ref_result = load_reference_dataset(args.ref_type.lower() if args.ref_type else None, args.ref_task)
    if ref_result is not None:
        ref_dataset, text_key_ref = ref_result
    else:
        ref_dataset, text_key_ref = None, None
    
    log_memory_usage("After loading all datasets")
    
    if args.debug_memory:
        print("\nTop 10 memory allocations:")
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        for stat in top_stats[:10]:
            print(f"{stat.size / (1024 * 1024):.1f} MB: {stat.traceback.format()[0]}")
    
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
        tol=1,
        scale_ref=args.ref_div_multiplier
    )
    
    # Run estimation
    print("Running dependence estimation...")
    log_memory_usage("Before estimation")
    estimator = VolumeEstimator.from_config(cfg)
    result = estimator.run()
    log_memory_usage("After estimation")
    
    # Clean up to reduce memory usage
    del estimator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
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
    
    log_memory_usage("End of script")

if __name__ == "__main__":
    main() 