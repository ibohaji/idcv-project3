from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path
from typing import Dict, List
import yaml


class Settings(BaseSettings):
    """General application settings."""
    # Data settings
    data_dir: str = "./data"
    train_dir: str = "./data/train"
    val_dir: str = "./data/val"
    test_dir: str = "./data/test"
    
    # Training settings
    device: str = "cuda"
    num_workers: int = 4
    pin_memory: bool = True
    
    # Output settings
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    save_best_only: bool = True
    
    # Logging
    log_interval: int = 10
    save_interval: int = 5
    
    # Ablation study configurations
    models: List[str] = ["EncoderDecoder", "Unet"]
    datasets: List[str] = ["PH2", "DRIVE"]
    losses: List[str] = ["BCE", "Focal", "WeightedBCE"]
    optimizers: List[str] = ["adam", "sgd"]
    
    # Dataset normalization statistics
    normalization: Dict[str, Dict[str, List[float]]] = {
        "PH2": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
        "DRIVE": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}
    }

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class Parameters(BaseSettings):
    """Training hyperparameters (shared across all experiments)."""
    # Training hyperparameters
    batch_size: int = 32
    epochs: int = 10
    weight_decay: float = 0.0001
    momentum: float = 0.9
    
    # Optimizer-specific learning rates
    learning_rates: Dict[str, float] = {"adam": 0.001, "sgd": 0.01}
    
    # Scheduler settings
    scheduler: str = "step"
    scheduler_step_size: int = 30
    scheduler_gamma: float = 0.1
    
    # Validation
    val_split: float = 0.2
    early_stopping_patience: int = 5

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def load_settings_from_yaml(yaml_path: Path) -> Settings:
    """Load settings from a YAML file."""
    with open(yaml_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return Settings(**config_dict)


def load_parameters_from_yaml(yaml_path: Path) -> Parameters:
    """Load parameters from a YAML file."""
    with open(yaml_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return Parameters(**config_dict)
