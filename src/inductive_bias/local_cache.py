import os
import torch
import numpy as np
import functools
import hashlib
import contextlib
import time
from typing import Optional, Any, Dict, Union, Callable, Tuple

# Module-level variable to track the current cache directory
_current_cache_dir = ".cache"


class CacheContext:
    """
    Context manager to temporarily set the cache directory for cached functions.

    Example:
        with CacheContext('./cache/run_123'):
            result = my_cached_function(x)
    """

    def __init__(self, cache_dir: str):
        self.new_cache_dir = cache_dir
        self.previous_cache_dir = None

    def __enter__(self) -> str:
        # Save the current directory and set the new one
        global _current_cache_dir
        self.previous_cache_dir = _current_cache_dir
        _current_cache_dir = self.new_cache_dir

        # Create the directory if it doesn't exist
        os.makedirs(self.new_cache_dir, exist_ok=True)
        return self.new_cache_dir

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore the previous directory
        global _current_cache_dir
        _current_cache_dir = self.previous_cache_dir


def cache(
    func: Optional[Callable] = None, verbose: bool = False, save_inputs: bool = False
) -> Callable:
    """
    Decorator that caches function results using torch.save.
    Works with PyTorch tensors, NumPy arrays, and dictionaries containing them.
    Uses the current cache directory set by CacheContext.

    Args:
        func: The function to decorate
        verbose: Whether to print cache hit/miss messages (default: False)
        save_inputs: Whether to save the function inputs alongside results (default: False)
                     When True, the cache file will contain a dict with "input" and "output" keys
                     instead of just the raw output

    Example:
        @cache
        def my_function(tensor):
            # Expensive computation
            return result

        @cache(verbose=True, save_inputs=True)
        def debug_function(tensor):
            # Cache messages will be printed and inputs will be saved
            return result
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            global _current_cache_dir
            nonlocal verbose, save_inputs

            # Create cache key based on function and arguments
            key_parts = [func.__name__]

            # Process args for key generation
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    # For tensors, use shape and content hash
                    tensor_hash = str(hash(arg.detach().cpu().numpy().tobytes()))
                    key_parts.append(f"tensor-{tuple(arg.shape)}-{tensor_hash[:10]}")
                elif isinstance(arg, np.ndarray):
                    # For numpy arrays, use shape and content hash
                    array_hash = str(hash(arg.tobytes()))
                    key_parts.append(f"ndarray-{arg.shape}-{array_hash[:10]}")
                else:
                    # For other types, use their string representation
                    key_parts.append(str(arg)[:50])  # Limit length

            # Process kwargs for key generation
            for k, v in sorted(kwargs.items()):
                if isinstance(v, torch.Tensor):
                    tensor_hash = str(hash(v.detach().cpu().numpy().tobytes()))
                    key_parts.append(f"{k}=tensor-{tuple(v.shape)}-{tensor_hash[:10]}")
                elif isinstance(v, np.ndarray):
                    array_hash = str(hash(v.tobytes()))
                    key_parts.append(f"{k}=ndarray-{v.shape}-{array_hash[:10]}")
                else:
                    key_parts.append(f"{k}={str(v)[:50]}")  # Limit length

            # Create deterministic hash of key parts to avoid filename length issues
            key = "-".join(key_parts)
            key_hash = hashlib.md5(key.encode()).hexdigest()
            cache_file = os.path.join(
                _current_cache_dir, f"{func.__name__}_{key_hash}.pt"
            )

            # Check if result is already cached
            if os.path.exists(cache_file):
                try:
                    # Try loading with weights_only=False since we know we're dealing with custom objects
                    cached_data = torch.load(cache_file, weights_only=False)

                    # If the cached data is a dict with 'input' and 'output' keys, this is the new format
                    if (
                        save_inputs
                        and isinstance(cached_data, dict)
                        and "output" in cached_data
                    ):
                        result = cached_data["output"]
                        if verbose:
                            print(
                                f"Loaded cached result (with inputs) for {func.__name__} from {cache_file}"
                            )
                    else:
                        # Legacy format or save_inputs=False: the entire cache is just the result
                        result = cached_data
                        if verbose:
                            print(
                                f"Loaded cached result for {func.__name__} from {cache_file}"
                            )

                    return result
                except Exception as e:
                    if verbose:
                        print(f"Error loading cache: {e}, computing fresh result")

            # Compute the result
            result = func(*args, **kwargs)

            # Save to cache
            try:
                # Create directory if it doesn't exist (in case it was deleted)
                os.makedirs(_current_cache_dir, exist_ok=True)

                # Prepare data to be saved
                if save_inputs:
                    # Create a dictionary with both inputs and outputs
                    cache_data = {
                        "input": {
                            "args": args,
                            "kwargs": kwargs,
                            "timestamp": time.time(),
                            "function": func.__name__,
                        },
                        "output": result,
                    }

                    # Save combined data
                    torch.save(
                        cache_data,
                        cache_file,
                        _use_new_zipfile_serialization=True,
                    )

                    if verbose:
                        print(
                            f"Cached result with inputs for {func.__name__} to {cache_file}"
                        )
                else:
                    # Save just the result (original behavior)
                    torch.save(
                        result,
                        cache_file,
                        _use_new_zipfile_serialization=True,
                    )

                    if verbose:
                        print(f"Cached result for {func.__name__} to {cache_file}")

            except Exception as e:
                if verbose:
                    print(f"Error saving to cache: {e}")

            return result

        return wrapper

    # This allows the decorator to be used with or without arguments
    if func is not None and callable(func):
        return decorator(func)
    else:
        return decorator


# Convenience function to get cached inputs for a specific function call
def get_cached_inputs(func_name: str, key_hash: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve the cached inputs for a specific function call.

    Args:
        func_name: Name of the function
        key_hash: Hash key for the specific function call

    Returns:
        Dict containing the inputs or None if not found
    """
    global _current_cache_dir
    cache_file = os.path.join(_current_cache_dir, f"{func_name}_{key_hash}.pt")

    # First try the new combined format
    if os.path.exists(cache_file):
        try:
            data = torch.load(cache_file, weights_only=False)
            if isinstance(data, dict) and "input" in data:
                return data["input"]
        except Exception:
            pass

    # For backward compatibility, try the old separate input file
    input_cache_file = os.path.join(
        _current_cache_dir, f"{func_name}_{key_hash}_inputs.pt"
    )

    if os.path.exists(input_cache_file):
        try:
            return torch.load(input_cache_file, weights_only=False)
        except Exception:
            pass

    return None


# Helper function to compute the hash key for a function and its arguments
def compute_cache_key(func_name: str, *args, **kwargs) -> str:
    """
    Compute the cache key hash for a function call with given args and kwargs.
    This is useful when you need to retrieve inputs for a function call you know about.

    Returns:
        Hash string that can be used with get_cached_inputs
    """
    key_parts = [func_name]

    # Process args for key generation
    for arg in args:
        if isinstance(arg, torch.Tensor):
            tensor_hash = str(hash(arg.detach().cpu().numpy().tobytes()))
            key_parts.append(f"tensor-{tuple(arg.shape)}-{tensor_hash[:10]}")
        elif isinstance(arg, np.ndarray):
            array_hash = str(hash(arg.tobytes()))
            key_parts.append(f"ndarray-{arg.shape}-{array_hash[:10]}")
        else:
            key_parts.append(str(arg)[:50])

    # Process kwargs for key generation
    for k, v in sorted(kwargs.items()):
        if isinstance(v, torch.Tensor):
            tensor_hash = str(hash(v.detach().cpu().numpy().tobytes()))
            key_parts.append(f"{k}=tensor-{tuple(v.shape)}-{tensor_hash[:10]}")
        elif isinstance(v, np.ndarray):
            array_hash = str(hash(v.tobytes()))
            key_parts.append(f"{k}=ndarray-{v.shape}-{array_hash[:10]}")
        else:
            key_parts.append(f"{k}={str(v)[:50]}")

    key = "-".join(key_parts)
    return hashlib.md5(key.encode()).hexdigest()


# Convenience function to set and get the current cache directory
def get_cache_dir() -> str:
    """Get the currently active cache directory."""
    global _current_cache_dir
    return _current_cache_dir


def set_cache_dir(directory: str) -> None:
    """
    Set the current cache directory globally.
    Note: Using CacheContext is preferred for better control.
    """
    global _current_cache_dir
    _current_cache_dir = directory
    os.makedirs(_current_cache_dir, exist_ok=True)
