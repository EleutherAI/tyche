import argparse
import yaml
import subprocess
import os
import time
import psutil
import csv
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from datetime import datetime
import itertools
from collections import defaultdict

@dataclass
class Experiment:
    name: str
    gpu: Optional[int]  # Now optional since we can dynamically assign
    params: dict
    status: str = "pending"  # pending, running, completed, failed

class GPUManager:
    def __init__(self, n_gpus: Optional[int] = None):
        if n_gpus is None:
            # Try to detect available GPUs using nvidia-smi
            try:
                result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
                self.n_gpus = len(result.stdout.strip().split("\n"))
            except FileNotFoundError:
                raise RuntimeError("Could not detect GPUs. Please specify --n-gpus")
        else:
            self.n_gpus = n_gpus
            
        self.gpu_processes: Dict[int, Optional[subprocess.Popen]] = {i: None for i in range(self.n_gpus)}
        
    def get_free_gpu(self) -> Optional[int]:
        for gpu_id, process in self.gpu_processes.items():
            if process is None or process.poll() is not None:
                # GPU is free or process has completed
                self.gpu_processes[gpu_id] = None
                return gpu_id
        return None
    
    def assign_process(self, gpu_id: int, process: subprocess.Popen):
        self.gpu_processes[gpu_id] = process
        
    def any_running(self) -> bool:
        return any(p is not None and p.poll() is None for p in self.gpu_processes.values())
    
    def cleanup(self):
        for process in self.gpu_processes.values():
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()

def merge_params(defaults, experiment_params):
    return {**defaults, **experiment_params}

def build_command(experiment: Experiment, defaults: dict) -> str:
    params = merge_params(defaults, experiment.params)
    cmd_parts = ["python", "scripts/measure_dependence.py"]
    
    for key, value in params.items():
        if isinstance(value, bool):
            if value:
                cmd_parts.append(f"--{key}")
        else:
            cmd_parts.append(f"--{key}")
            cmd_parts.append(str(value))
    
    return " ".join(cmd_parts)

def experiment_already_run(experiment: Experiment, defaults: dict) -> Tuple[bool, str]:
    """
    Check if an experiment with the same parameters has already been run.
    
    Returns:
        Tuple[bool, str]: (True if experiment has been run, reason message)
    """
    log_file = Path("dependence_results/dependence_experiment_logs.csv")
    if not log_file.exists():
        return False, "No experiment log file found"
    
    # Get the parameters for this experiment
    params = merge_params(defaults, experiment.params)
    
    # Define the key parameters that identify a unique experiment
    key_params = [
        'model_family', 'model_size', 'task1', 'task2', 
        'ref_type', 'ref_task', 'n_samples', 'val_size',
        'use_preconditioner', 'preconditioner_eps', 'preconditioner_exp',
        'checkpoint_step'
    ]
    
    try:
        with open(log_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Check if all key parameters match
                match = True
                for param in key_params:
                    # Convert boolean strings to actual booleans for comparison
                    if param in ['use_preconditioner']:
                        row_value = row[param].lower() == 'true'
                        param_value = params.get(param.replace('_', '-'), False)
                    else:
                        row_value = row[param]
                        # Convert parameter name format from snake_case to kebab-case
                        param_value = str(params.get(param.replace('_', '-'), ''))
                    
                    if str(row_value) != str(param_value):
                        match = False
                        break
                
                if match:
                    return True, f"Experiment already run at {row['timestamp']}, result file: {row['result_file']}"
    except Exception as e:
        return False, f"Error checking experiment log: {str(e)}"
    
    return False, "No matching experiment found"

def ensure_log_file_exists():
    """
    Ensure the experiment log file exists with proper headers.
    """
    log_file = Path("dependence_results/dependence_experiment_logs.csv")
    if not log_file.exists():
        # Create the file with headers
        headers = [
            'timestamp', 'git_hash', 'model_family', 'model_size', 
            'task1', 'task2', 'ref_type', 'ref_task', 'n_samples', 
            'val_size', 'cache_mode', 'use_preconditioner', 
            'preconditioner_eps', 'preconditioner_exp', 'checkpoint_step',
            'command', 'result_file', 'joint_ref', 'marginal_ref',
            'min_marginal', 'normalized_dependence'
        ]
        with open(log_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
        print(f"Created new experiment log file: {log_file}")

def run_experiment(experiment: Experiment, gpu: int, defaults: dict) -> subprocess.Popen:
    cmd = build_command(experiment, defaults)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{experiment.name}_{timestamp}.log"
    
    print(f"\nStarting experiment: {experiment.name}")
    print(f"GPU: {gpu}")
    print(f"Command: {cmd}")
    print(f"Log file: {log_file}\n")
    
    with open(log_file, "w") as f:
        process = subprocess.Popen(
            cmd,
            shell=True,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True
        )
    return process

def group_experiments_by_datasets(experiments, defaults):
    """Group experiments by the datasets they use to avoid loading the same datasets in parallel."""
    dataset_groups = defaultdict(list)
    
    for exp in experiments:
        params = merge_params(defaults, exp.params)
        # Create a key based on the datasets used
        datasets = (params.get('task1'), params.get('task2'), 
                   params.get('ref-type'), params.get('ref-task'))
        dataset_groups[datasets].append(exp)
    
    # Sort groups by size (largest first) to optimize scheduling
    return sorted(dataset_groups.values(), key=len, reverse=True)

def main():
    parser = argparse.ArgumentParser(description="Run multiple dependence experiments from config")
    parser.add_argument("--config", type=str, default="configs/experiments.yaml",
                      help="Path to experiment configuration file")
    parser.add_argument("--experiment", type=str, default=None,
                      help="Run specific experiment by name (optional)")
    parser.add_argument("--n-gpus", type=int, default=None,
                      help="Number of GPUs to use (default: auto-detect)")
    parser.add_argument("--check-interval", type=float, default=10,
                      help="Interval in seconds to check for completed experiments")
    parser.add_argument("--recompute", action="store_true",
                      help="Recompute experiments even if they have already been run")
    parser.add_argument("--verbose", action="store_true",
                      help="Print verbose information about experiments")
    parser.add_argument("--group-by-dataset", action="store_true", default=True,
                      help="Group experiments by datasets to avoid parallel loading")
    args = parser.parse_args()

    # Ensure results directories exist
    Path("dependence_results").mkdir(exist_ok=True)
    Path("dependence_results/detailed_results").mkdir(exist_ok=True)
    ensure_log_file_exists()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    defaults = config.get("defaults", {})
    
    # Convert config experiments to Experiment objects
    experiments = [
        Experiment(
            name=exp["name"],
            gpu=exp.get("gpu"),  # GPU assignment is now optional
            params=exp["params"]
        )
        for exp in config["experiments"]
    ]
    
    if args.experiment:
        experiments = [exp for exp in experiments if exp.name == args.experiment]
        if not experiments:
            raise ValueError(f"No experiment found with name {args.experiment}")

    # Filter out experiments that have already been run unless --recompute is specified
    if not args.recompute:
        filtered_experiments = []
        skipped_count = 0
        for exp in experiments:
            already_run, reason = experiment_already_run(exp, defaults)
            if already_run:
                skipped_count += 1
                if args.verbose:
                    print(f"Skipping experiment '{exp.name}': {reason}")
            else:
                filtered_experiments.append(exp)
        
        if skipped_count > 0:
            print(f"Skipped {skipped_count} experiments that have already been run. Use --recompute to run them again.")
        
        experiments = filtered_experiments
        
        if not experiments:
            print("All experiments have already been run. Use --recompute to run them again.")
            return

    gpu_manager = GPUManager(args.n_gpus)
    print(f"Running experiments using {gpu_manager.n_gpus} GPUs")
    
    # Group experiments by datasets if requested
    if args.group_by_dataset:
        experiment_groups = group_experiments_by_datasets(experiments, defaults)
        print(f"Grouped {len(experiments)} experiments into {len(experiment_groups)} dataset groups")
        # Flatten groups but keep the grouping order
        experiments = list(itertools.chain.from_iterable(experiment_groups))
    
    try:
        # Main experiment loop
        while experiments or gpu_manager.any_running():
            # Try to start new experiments on free GPUs
            while experiments:
                free_gpu = gpu_manager.get_free_gpu()
                if free_gpu is None:
                    break
                    
                experiment = experiments.pop(0)  # Get next experiment
                # Add debug flag to experiment parameters
                experiment.params["debug-memory"] = True
                process = run_experiment(experiment, free_gpu, defaults)
                gpu_manager.assign_process(free_gpu, process)
            
            time.sleep(args.check_interval)
            
    except KeyboardInterrupt:
        print("\nReceived interrupt, cleaning up...")
        gpu_manager.cleanup()
        # Add interrupted experiments back to queue
        for gpu_id, process in gpu_manager.gpu_processes.items():
            if process is not None and process.poll() is None:
                print(f"Experiment on GPU {gpu_id} was interrupted")
    
    print("\nAll experiments completed")

if __name__ == "__main__":
    main() 