import argparse
import yaml
import subprocess
import os
import time
import psutil
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from datetime import datetime

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
    args = parser.parse_args()

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

    gpu_manager = GPUManager(args.n_gpus)
    print(f"Running experiments using {gpu_manager.n_gpus} GPUs")
    
    try:
        # Main experiment loop
        while experiments or gpu_manager.any_running():
            # Try to start new experiments on free GPUs
            while experiments:
                free_gpu = gpu_manager.get_free_gpu()
                if free_gpu is None:
                    break
                    
                experiment = experiments.pop(0)  # Get next experiment
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