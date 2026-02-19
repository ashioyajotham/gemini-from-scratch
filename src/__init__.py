"""
Gemini from Scratch - Educational transformer implementation.

This package provides modular components for building transformer-based
language models from scratch using PyTorch.
"""

__version__ = "0.1.0"

# Submodules are imported lazily to avoid import errors
# when optional dependencies are not installed
__all__ = ["utils", "data", "models", "training", "generation", "__version__"]


def __getattr__(name):
    """Lazy import submodules."""
    if name == "utils":
        from . import utils
        return utils
    elif name == "data":
        from . import data
        return data
    elif name == "models":
        from . import models
        return models
    elif name == "training":
        from . import training
        return training
    elif name == "generation":
        from . import generation
        return generation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
