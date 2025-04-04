#!/usr/bin/env python
"""
Script to upload Adam buffer files to Hugging Face Hub.
"""

import os
import argparse
from pathlib import Path
from huggingface_hub import HfApi
import re


def parse_args():
    parser = argparse.ArgumentParser(description="Upload Adam buffer files to Hugging Face Hub")
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/mnt/ssd-1/adam/tiny-pythia/ckpts",
        help="Base directory containing model checkpoints",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["pythia-14m", "pythia-31m"],
        required=True,
        help="Model to upload files for",
    )
    parser.add_argument(
        "--token",
        type=str,
        required=True,
        help="Hugging Face API token",
    )
    parser.add_argument(
        "--start_from_step",
        type=int,
        help="Skip all steps before this number (inclusive)",
    )
    return parser.parse_args()


def get_step_branches(api, repo_id):
    """Get all branches that start with 'step' and extract their step numbers."""
    branches = api.list_repo_refs(repo_id).branches
    step_branches = []
    for branch in branches:
        if branch.name.startswith("step"):
            step_num = int(branch.name[4:])  # Extract number after "step"
            step_branches.append(step_num)
        else:
            print(f"Skipping branch {branch.name}: does not start with 'step'")
    return sorted(step_branches)


def main():
    args = parse_args()
    
    # Initialize Hugging Face API
    api = HfApi(token=args.token)
    
    # Set repo ID based on model
    repo_id = f"EleutherAI/{args.model}"
    
    # Get all step branches
    step_numbers = get_step_branches(api, repo_id)
    print(f"Found {len(step_numbers)} step branches for {repo_id}")
    
    # Filter steps if start_from_step is specified
    if args.start_from_step is not None:
        step_numbers = [s for s in step_numbers if s >= args.start_from_step]
        print(f"Filtered to {len(step_numbers)} steps starting from step {args.start_from_step}")
    
    # Base directory for this model
    model_dir = Path(args.base_dir) / args.model
    
    # Upload files for each step
    for step in step_numbers:
        checkpoint_dir = model_dir / f"global_step{step}"
        adam_file = checkpoint_dir / "mp_rank_00_model_states.pt"
        
        if not adam_file.exists():
            raise ValueError(f"Adam file not found at {adam_file}")
            
        api.upload_file(
            path_or_fileobj=str(adam_file),
            path_in_repo="mp_rank_00_model_states.pt",
            repo_id=repo_id,
            repo_type="model",
            revision=f"step{step}",
        )
        print(f"Uploaded Adam file for step {step}")
    
    print("Upload complete!")


if __name__ == "__main__":
    main()
