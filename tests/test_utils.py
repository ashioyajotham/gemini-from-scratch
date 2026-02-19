"""Tests for utility modules."""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from src.utils.device import get_device, to_device, get_device_info
from src.utils.config import load_config, merge_configs, Config, ModelConfig
from src.utils.helpers import set_seed, count_parameters, format_number, get_logger
from src.utils.checkpointing import save_checkpoint, load_checkpoint
from src.utils.metrics import compute_perplexity, compute_accuracy, MetricsTracker


class TestDevice:
    """Tests for device utilities."""

    def test_get_device_returns_device(self):
        """get_device should return a torch.device."""
        device = get_device()
        assert isinstance(device, torch.device)

    def test_get_device_cpu_only(self):
        """get_device with prefer_gpu=False should return CPU."""
        device = get_device(prefer_gpu=False)
        assert device.type == "cpu"

    def test_to_device_tensor(self):
        """to_device should move a tensor to the specified device."""
        device = get_device(prefer_gpu=False)
        tensor = torch.randn(5, 5)
        moved = to_device(tensor, device)
        assert moved.device.type == device.type

    def test_to_device_dict(self):
        """to_device should move a dict of tensors."""
        device = get_device(prefer_gpu=False)
        data = {"a": torch.randn(3), "b": torch.randn(4)}
        moved = to_device(data, device)
        assert all(v.device.type == device.type for v in moved.values())

    def test_to_device_nested(self):
        """to_device should handle nested structures."""
        device = get_device(prefer_gpu=False)
        data = [torch.randn(2), {"x": torch.randn(3)}]
        moved = to_device(data, device)
        assert moved[0].device.type == device.type
        assert moved[1]["x"].device.type == device.type

    def test_get_device_info(self):
        """get_device_info should return expected keys."""
        info = get_device_info()
        assert "device" in info
        assert "cuda_available" in info
        assert "pytorch_version" in info


class TestConfig:
    """Tests for configuration utilities."""

    def test_load_config(self, tmp_path):
        """load_config should load a YAML file."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text("model:\n  d_model: 256\n  n_layers: 4\n")

        config = load_config(config_file)
        assert config["model"]["d_model"] == 256
        assert config["model"]["n_layers"] == 4

    def test_load_config_not_found(self):
        """load_config should raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")

    def test_merge_configs(self):
        """merge_configs should deep merge dictionaries."""
        base = {"model": {"d_model": 256, "n_layers": 4}, "training": {"lr": 1e-4}}
        override = {"model": {"n_layers": 6}}

        merged = merge_configs(base, override)
        assert merged["model"]["d_model"] == 256  # Preserved
        assert merged["model"]["n_layers"] == 6  # Overridden
        assert merged["training"]["lr"] == 1e-4  # Preserved

    def test_model_config_from_dict(self):
        """ModelConfig should be creatable from dict."""
        d = {"d_model": 512, "n_heads": 8, "unknown_key": "ignored"}
        config = ModelConfig.from_dict(d)
        assert config.d_model == 512
        assert config.n_heads == 8

    def test_config_from_yaml(self, tmp_path):
        """Config should be loadable from YAML."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "model:\n  d_model: 512\ntraining:\n  batch_size: 64\n"
        )

        config = Config.from_yaml(config_file)
        assert config.model.d_model == 512
        assert config.training.batch_size == 64


class TestHelpers:
    """Tests for helper utilities."""

    def test_set_seed_reproducibility(self):
        """set_seed should make random operations reproducible."""
        set_seed(42)
        a = torch.randn(5)

        set_seed(42)
        b = torch.randn(5)

        assert torch.allclose(a, b)

    def test_count_parameters(self):
        """count_parameters should count model parameters."""
        model = nn.Linear(10, 5)  # 10*5 + 5 = 55 parameters
        assert count_parameters(model) == 55

    def test_count_parameters_non_trainable(self):
        """count_parameters should respect trainable_only flag."""
        model = nn.Linear(10, 5)
        for p in model.parameters():
            p.requires_grad = False

        assert count_parameters(model, trainable_only=True) == 0
        assert count_parameters(model, trainable_only=False) == 55

    def test_format_number(self):
        """format_number should format large numbers."""
        assert format_number(1500) == "1.50K"
        assert format_number(1500000) == "1.50M"
        assert format_number(1500000000) == "1.50B"
        assert format_number(100) == "100"

    def test_get_logger(self):
        """get_logger should return a configured logger."""
        logger = get_logger("test")
        assert logger.name == "test"


class TestCheckpointing:
    """Tests for checkpointing utilities."""

    def test_save_load_checkpoint(self, tmp_path):
        """save_checkpoint and load_checkpoint should round-trip."""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())

        # Save
        path = tmp_path / "checkpoint.pt"
        save_checkpoint(
            model,
            path,
            optimizer=optimizer,
            step=100,
            loss=2.5,
            config={"d_model": 256}
        )

        # Load into new model
        new_model = nn.Linear(10, 5)
        new_optimizer = torch.optim.Adam(new_model.parameters())

        metadata = load_checkpoint(path, new_model, new_optimizer)

        assert metadata["step"] == 100
        assert metadata["loss"] == 2.5
        assert metadata["config"]["d_model"] == 256

        # Check weights match
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2)

    def test_load_checkpoint_not_found(self):
        """load_checkpoint should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_checkpoint("nonexistent.pt")


class TestMetrics:
    """Tests for metrics utilities."""

    def test_compute_perplexity(self):
        """compute_perplexity should return exp(loss)."""
        import math
        assert abs(compute_perplexity(1.0) - math.e) < 1e-5
        assert abs(compute_perplexity(0.0) - 1.0) < 1e-5

    def test_compute_accuracy(self):
        """compute_accuracy should compute token-level accuracy."""
        # Perfect predictions
        logits = torch.tensor([[[10.0, 0.0], [0.0, 10.0]]])  # predicts [0, 1]
        targets = torch.tensor([[0, 1]])
        assert compute_accuracy(logits, targets) == 1.0

        # Wrong predictions
        targets = torch.tensor([[1, 0]])
        assert compute_accuracy(logits, targets) == 0.0

    def test_compute_accuracy_with_ignore(self):
        """compute_accuracy should ignore specified tokens."""
        logits = torch.tensor([[[10.0, 0.0], [0.0, 10.0], [10.0, 0.0]]])
        targets = torch.tensor([[0, 1, -100]])  # Last token ignored
        acc = compute_accuracy(logits, targets, ignore_index=-100)
        assert acc == 1.0

    def test_metrics_tracker(self):
        """MetricsTracker should track and average metrics."""
        tracker = MetricsTracker()
        tracker.update("loss", 1.0)
        tracker.update("loss", 2.0)
        tracker.update("loss", 3.0)

        assert tracker.get_average("loss") == 2.0

        tracker.reset()
        assert tracker.get_average("loss") is None
