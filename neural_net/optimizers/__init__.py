from enum import Enum

from .adam import Adam
from .base import BaseOptimizer
from .config import OptimizerConfig
from .factory import create_optimizer
from .sgd import SGD

__all__ = [
    "BaseOptimizer",
    "SGD",
    "Adam",
    "OptimizerType",
    "create_optimizer",
    "OptimizerConfig",
]


class OptimizerType(Enum):
    """Types of optimizers available."""

    SGD = SGD
    ADAM = Adam
