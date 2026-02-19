"""Configuration management utilities."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union
from pathlib import Path
import yaml


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        path: Path to the YAML config file.

    Returns:
        Dictionary containing configuration values.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If YAML parsing fails.

    Example:
        >>> config = load_config("configs/small_model.yaml")
        >>> print(config["model"]["d_model"])
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two configuration dictionaries.

    Values from override take precedence over base.

    Args:
        base: Base configuration dictionary.
        override: Override configuration dictionary.

    Returns:
        Merged configuration dictionary.

    Example:
        >>> base = {"model": {"d_model": 256, "n_layers": 4}}
        >>> override = {"model": {"n_layers": 6}}
        >>> merged = merge_configs(base, override)
        >>> print(merged["model"]["n_layers"])  # 6
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


@dataclass
class ModelConfig:
    """Configuration for transformer model architecture."""

    vocab_size: int = 8000
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 1024
    max_seq_len: int = 512
    dropout: float = 0.1
    layer_norm_eps: float = 1e-6
    tie_weights: bool = True

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelConfig":
        """Create config from dictionary, ignoring unknown keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""

    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 10000
    gradient_clip: float = 1.0
    log_interval: int = 100
    eval_interval: int = 500
    save_interval: int = 1000

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingConfig":
        """Create config from dictionary, ignoring unknown keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class DataConfig:
    """Configuration for data processing."""

    seq_length: int = 512
    train_split: float = 0.9
    num_workers: int = 4
    pin_memory: bool = True

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DataConfig":
        """Create config from dictionary, ignoring unknown keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class Config:
    """
    Complete configuration combining model, training, and data configs.

    This provides type-safe access to all configuration values.

    Example:
        >>> config = Config.from_yaml("configs/small_model.yaml")
        >>> print(config.model.d_model)
        >>> print(config.training.learning_rate)
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        """Create Config from a dictionary."""
        return cls(
            model=ModelConfig.from_dict(d.get("model", {})),
            training=TrainingConfig.from_dict(d.get("training", {})),
            data=DataConfig.from_dict(d.get("data", {})),
        )

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        """Load Config from a YAML file."""
        d = load_config(path)
        return cls.from_dict(d)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        from dataclasses import asdict
        return asdict(self)

    def save(self, path: Union[str, Path]) -> None:
        """Save config to YAML file."""
        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
