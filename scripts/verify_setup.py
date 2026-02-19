#!/usr/bin/env python
"""
Verification script for the Gemini workshop environment.

Run this script to verify that all dependencies are installed correctly
and the environment is properly configured.

Usage:
    python scripts/verify_setup.py
"""

import sys
from pathlib import Path


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_status(name: str, status: bool, details: str = "") -> None:
    """Print a status line with checkmark or X."""
    symbol = "[OK]" if status else "[X]"
    print(f"  {symbol} {name}")
    if details:
        print(f"      {details}")


def check_python_version() -> bool:
    """Check Python version is 3.9+."""
    version = sys.version_info
    is_ok = version.major >= 3 and version.minor >= 9
    print_status(
        f"Python {version.major}.{version.minor}.{version.micro}",
        is_ok,
        "Requires Python 3.9+" if not is_ok else ""
    )
    return is_ok


def check_pytorch() -> bool:
    """Check PyTorch installation and version."""
    try:
        import torch
        version = torch.__version__
        is_ok = int(version.split(".")[0]) >= 2
        print_status(f"PyTorch {version}", is_ok, "Requires PyTorch 2.0+" if not is_ok else "")
        return is_ok
    except ImportError:
        print_status("PyTorch", False, "Not installed")
        return False


def check_device() -> dict:
    """Check available compute devices."""
    import torch

    info = {
        "cpu": True,
        "cuda": torch.cuda.is_available(),
        "mps": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
    }

    print_status("CPU", True, "Always available")

    if info["cuda"]:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_count = torch.cuda.device_count()
        print_status(f"CUDA ({gpu_count} device(s))", True, gpu_name)
    else:
        print_status("CUDA", False, "Not available")

    if info["mps"]:
        print_status("MPS (Apple Silicon)", True)
    else:
        print_status("MPS", False, "Not available")

    return info


def check_dependencies() -> dict:
    """Check all required dependencies."""
    dependencies = {
        "numpy": "numpy",
        "scipy": "scipy",
        "sentencepiece": "sentencepiece",
        "tokenizers": "tokenizers",
        "datasets": "datasets",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "pyyaml": "yaml",
        "tqdm": "tqdm",
    }

    results = {}
    for name, import_name in dependencies.items():
        try:
            module = __import__(import_name)
            version = getattr(module, "__version__", "unknown")
            print_status(f"{name} ({version})", True)
            results[name] = True
        except ImportError:
            print_status(name, False, "Not installed")
            results[name] = False

    return results


def check_package_imports() -> bool:
    """Check that the package can be imported."""
    try:
        from src.utils import get_device, load_config, set_seed
        from src.data import BPETokenizer, TextDataset, create_dataloader
        print_status("src.utils", True)
        print_status("src.data", True)
        return True
    except ImportError as e:
        print_status("Package imports", False, str(e))
        return False


def check_config_loading() -> bool:
    """Check that config files can be loaded."""
    try:
        from src.utils import load_config, Config

        config_path = Path("configs/small_model.yaml")
        if config_path.exists():
            config = Config.from_yaml(config_path)
            print_status(
                "Config loading",
                True,
                f"d_model={config.model.d_model}, n_layers={config.model.n_layers}"
            )
            return True
        else:
            print_status("Config loading", False, "configs/small_model.yaml not found")
            return False
    except Exception as e:
        print_status("Config loading", False, str(e))
        return False


def run_tensor_test() -> bool:
    """Run a simple tensor operation test."""
    try:
        import torch
        from src.utils import get_device, to_device

        device = get_device()

        # Create and move tensors
        x = torch.randn(10, 10)
        x = to_device(x, device)

        # Simple operation
        y = torch.matmul(x, x.T)

        print_status(
            f"Tensor operations on {device}",
            True,
            f"Created {x.shape} tensor, computed matmul"
        )
        return True
    except Exception as e:
        print_status("Tensor operations", False, str(e))
        return False


def main():
    """Run all verification checks."""
    print("\n" + "=" * 60)
    print("  Gemini Workshop - Environment Verification")
    print("=" * 60)

    all_passed = True

    # Python version
    print_header("Python Environment")
    all_passed &= check_python_version()

    # PyTorch
    print_header("PyTorch Installation")
    all_passed &= check_pytorch()

    # Devices
    print_header("Compute Devices")
    check_device()

    # Dependencies
    print_header("Dependencies")
    dep_results = check_dependencies()
    all_passed &= all(dep_results.values())

    # Package imports
    print_header("Package Imports")
    all_passed &= check_package_imports()

    # Config loading
    print_header("Configuration")
    all_passed &= check_config_loading()

    # Tensor test
    print_header("Tensor Operations")
    all_passed &= run_tensor_test()

    # Summary
    print_header("Summary")
    if all_passed:
        print("  All checks passed! Your environment is ready.")
        print("  You can start the workshop with:")
        print("    jupyter notebook notebooks/00_setup_and_verification.ipynb")
    else:
        print("  Some checks failed. Please review the errors above.")
        print("  Try reinstalling with: pip install -e .")

    print()
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
