"""
Configuration for experiments.
Each experiment (M0, M1, M2, M3) has its own config.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ExperimentConfig:
    """
    Configuration for a single experiment.

    This allows you to define all hyperparameters in one place
    and ensures reproducibility.
    """

    # Experiment metadata
    name: str  # e.g., 'M0', 'M1', 'M2', 'M3'
    description: str

    # Model architecture
    hidden_layers: list[int]  # e.g., [128, 64]
    batch_size: int

    # Training
    epochs: int
    learning_rate: float
    optimizer: str  # 'sgd' or 'adam'

    # Optimizer params
    momentum: float = 0.0  # For SGD
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999

    # Learning rate scheduling
    use_lr_scheduler: bool = False
    lr_scheduler_type: Optional[str] = None  # 'linear' or 'exponential'
    lr_decay_rate: Optional[float] = None
    lr_gamma: Optional[float] = None

    # Regularization
    use_l2: bool = False
    l2_lambda: float = 0.01

    use_early_stopping: bool = False
    early_stopping_patience: int = 10

    random_seed: int = 42


# Define experiment configurations
M0_CONFIG = ExperimentConfig(
    name="M0",
    description="Baseline model: 2 hidden layers [128, 64], standard GD",
    hidden_layers=[128, 64],
    batch_size=256,
    epochs=50,
    learning_rate=0.001,
    optimizer="sgd",
    use_lr_scheduler=False,
    use_l2=False,
    use_early_stopping=False,
)

# TODO: Define M1, M2, M3 configs after experiments
