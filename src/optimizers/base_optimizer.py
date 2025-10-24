"""
Abstract base class for optimizers.
"""
from abc import ABC, abstractmethod


class BaseOptimizer(ABC):
    """
    Abstract base class for optimization algorithms.
    """

    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    @abstractmethod
    def step(self, gradients):
        """
        Compute parameter updates given gradients.

        Args:
            gradients: list of (grad_weights, grad_bias) tuples for each layer

        Returns:
            list of (weight_update, bias_update) tuples
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset optimizer state (e.g., momentum buffers)"""
        pass
