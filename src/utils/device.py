"""Device management utilities for CPU/CUDA/MPS support."""

from typing import Union, Dict, Any
import torch


def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Auto-detect and return the best available device.

    Priority: CUDA > MPS > CPU

    Args:
        prefer_gpu: If False, always return CPU device.

    Returns:
        torch.device: The selected device.

    Example:
        >>> device = get_device()
        >>> model = model.to(device)
    """
    if not prefer_gpu:
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def to_device(
    data: Union[torch.Tensor, Dict[str, torch.Tensor], list, tuple],
    device: torch.device
) -> Union[torch.Tensor, Dict[str, torch.Tensor], list, tuple]:
    """
    Move tensor(s) to the specified device.

    Handles single tensors, dictionaries of tensors, and nested structures.

    Args:
        data: Tensor or collection of tensors to move.
        device: Target device.

    Returns:
        Data moved to the specified device.

    Example:
        >>> batch = {"input_ids": tensor1, "labels": tensor2}
        >>> batch = to_device(batch, device)
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_device(item, device) for item in data]
    elif isinstance(data, tuple):
        return tuple(to_device(item, device) for item in data)
    else:
        return data


def get_device_info() -> Dict[str, Any]:
    """
    Get detailed information about available devices.

    Returns:
        Dictionary containing device information:
        - device: Selected device name
        - cuda_available: Whether CUDA is available
        - cuda_device_count: Number of CUDA devices
        - cuda_device_name: Name of CUDA device (if available)
        - mps_available: Whether MPS is available
        - pytorch_version: PyTorch version

    Example:
        >>> info = get_device_info()
        >>> print(f"Using: {info['device']}")
    """
    info = {
        "device": str(get_device()),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_device_name": None,
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        "pytorch_version": torch.__version__,
    }

    if info["cuda_available"] and info["cuda_device_count"] > 0:
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_memory_total"] = torch.cuda.get_device_properties(0).total_memory
        info["cuda_memory_allocated"] = torch.cuda.memory_allocated(0)

    return info
