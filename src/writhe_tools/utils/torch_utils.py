#!/usr/bin/env python

import numpy as np
import torch
import time
import functools
import gc
from collections import OrderedDict
from .sorting import multireplace


def cleanup():
    gc.collect()  # Clean up unreferenced memory
    with torch.no_grad():
        torch.cuda.empty_cache()


def catch_cuda_oom(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            print("Error occurred during CUDA enabled operation, cleaning up tensors on GPU:", e)
            cleanup()  # Clear the CUDA cache to free memory
            # Optionally, reattempt the function, raise an error, or handle it as needed
    return wrapper


def profile_function(algorithm, args:dict, track_gpu=True, device=0):
    """
    Function to profile execution time and GPU memory usage for functions with PyTorch tensors.

    returns : executation_time (seconds), max_gpu_memory_used (GB)

    """
    # Reset GPU memory stats (if using GPU)
    if track_gpu and device and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    # Time execution
    start_time = time.time()
    result = algorithm(**args)  # Execute the function
    end_time = time.time()

    # Track GPU memory after execution (if using GPU)
    if track_gpu:
        torch.cuda.synchronize()  # Make sure all GPU operations are done
        max_gpu_memory_used = torch.cuda.max_memory_allocated() / (1024 ** 2) / 1e3  # Peak GPU memory
    else:
        max_gpu_memory_used = 0

    # Compute execution time
    execution_time = end_time - start_time

    return execution_time, max_gpu_memory_used


def gpu_stats():
    def list_tensors_on_gpu():
        all_tensors = [obj for obj in gc.get_objects() if isinstance(obj, torch.Tensor) and obj.is_cuda]
        for tensor in all_tensors:
            print(f"Tensor ID: {id(tensor)}, Size: {tensor.size()}, Device: {tensor.device}")
    list_tensors_on_gpu()
    allocated_memory = torch.cuda.memory_allocated()
    print(f"Allocated memory: {allocated_memory / (1024 ** 2):.2f} MB")
    # Total memory reserved by PyTorch
    reserved_memory = torch.cuda.memory_reserved()
    print(f"Reserved memory: {reserved_memory / (1024 ** 2):.2f} MB")
    print(torch.cuda.memory_summary(device=None, abbreviated=False))


def get_available_cuda_memory(device: int = 0):
    """return VRAM available on a device in GB"""

    assert torch.cuda.is_available(), "CUDA is not available"

    return (torch.cuda.get_device_properties(device).total_memory
            - torch.cuda.memory_allocated(device)) / 1024 ** 3

def array_size(array):
    """
    Calculate the memory usage of a NumPy array or PyTorch tensor in gigabytes.

    Args:
        array (np.ndarray or torch.Tensor): The array or tensor to analyze.

    Returns:
        float: Memory usage in gigabytes (GB).
    """
    if isinstance(array, np.ndarray):
        total_bytes = array.nbytes
    elif isinstance(array, torch.Tensor):
        total_bytes = array.element_size() * array.nelement()
    else:
        raise TypeError("Input must be a NumPy array or PyTorch tensor")

    return total_bytes / (1024 ** 3)


def array_item_size(array):
    """
    Get the number of bytes per element for a NumPy array or PyTorch tensor.

    Args:
        array (np.ndarray or torch.Tensor): The array or tensor to analyze.

    Returns:
        int: Number of bytes per element.
    """
    if isinstance(array, np.ndarray):
        return array.itemsize
    elif isinstance(array, torch.Tensor):
        return array.element_size()
    else:
        raise TypeError("Input must be a NumPy array or PyTorch tensor")


def estimate_segment_batch_size(xyz):
    mem_avail = (get_available_cuda_memory() - array_size(xyz) - 1) * 1e9
    n, d = xyz.shape[:2]
    # m is the bytes per item, it will always be 4 for float32, the default for torch
    m = 4 # array_item_size(xyz)
    # the guessed batch size has been cut in half from what is combinatorically predicted

    return int(((mem_avail / (3 * n * m)) - d) / 24)


# def estimate_segment_batch_size(n_samples: int):
#
#     return int((700 * 29977 * get_available_cuda_memory()) / (7.79229736328125 * n_samples))


def get_metrics(path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    event_accumulator = EventAccumulator(path)
    event_accumulator.Reload()
    steps = {x.step for x in event_accumulator.Scalars("epoch")}
    epoch = list(range(len(steps)))
    train_loss, val_loss = [-np.array([x.value for x in event_accumulator.Scalars(_key) if x.step in steps]) for _key in
                            ["train_loss","val_loss"]]
    return np.array(epoch),train_loss, val_loss


def plain_state_dict(d, badwords=["module.","model."]):
    replacements= dict(zip(badwords,[""]*len(badwords)))
    new = OrderedDict()
    for k, v in d.items():
        name = multireplace(string=k, replacements=replacements, ignore_case=True)
        new[name] = v
    return new


def load_state_dict(model, file):
    try:
        model.load_state_dict(plain_state_dict(torch.load(file)))
    #if we're trying to load from lightning state dict
    except:
        model.load_state_dict(plain_state_dict(torch.load(file)["state_dict"]))
    return model