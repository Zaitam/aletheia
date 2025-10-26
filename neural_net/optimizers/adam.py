from .base import BaseOptimizer


class Adam(BaseOptimizer):
    """Adam optimizer (Adaptive Moment Estimation)."""

    def __init__(
        self,
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        # TODO: Add state variables (m, v, t) when implementing

    @classmethod
    def from_config(cls, config):
        """Create Adam optimizer from configuration."""
        return cls(
            learning_rate=config.learning_rate,
            beta1=config.beta1,
            beta2=config.beta2,
            epsilon=config.epsilon,
        )

    def step(self, gradients):
        """Compute parameter updates using Adam algorithm."""
        # TODO: Implement Adam update logic
        raise NotImplementedError("Adam optimizer not yet implemented")

    def reset(self):
        """Reset Adam state (moving averages and timestep)."""
        # TODO: Reset m, v, t when implementing
        pass
