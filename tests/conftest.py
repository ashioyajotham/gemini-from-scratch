"""Pytest configuration and fixtures."""

import pytest
import torch


@pytest.fixture(autouse=True)
def reset_seed():
    """Reset random seeds before each test for reproducibility."""
    torch.manual_seed(42)
    yield


@pytest.fixture
def device():
    """Return the available device for testing."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
