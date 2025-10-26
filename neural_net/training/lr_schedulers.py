from abc import ABC, abstractmethod


class LRScheduler(ABC):
    """Base class for learning rate schedulers"""

    def __init__(self, optimizer, initial_lr: float):
        self.optimizer = optimizer
        self.initial_lr = initial_lr

    @abstractmethod
    def step(self, epoch: int):
        """Update learning rate based on epoch"""

    @classmethod
    @abstractmethod
    def from_config(cls, config, optimizer, initial_lr: float):
        """Create from configuration"""


class LinearScheduler(LRScheduler):
    """
    Linear learning rate decay with saturation.

    lr = max(min_lr, initial_lr - decay_rate * epoch)
    """

    def __init__(
        self, optimizer, initial_lr: float, decay_rate: float, min_lr: float = 1e-6
    ):
        super().__init__(optimizer, initial_lr)
        self.decay_rate = decay_rate
        self.min_lr = min_lr

    @classmethod
    def from_config(cls, config, optimizer, initial_lr: float):
        """Create from configuration."""
        return cls(
            optimizer=optimizer,
            initial_lr=initial_lr,
            decay_rate=config.decay_rate,
            min_lr=config.min_lr if hasattr(config, "min_lr") else 1e-6,
        )

    def step(self, epoch: int):
        """Update learning rate linearly"""
        new_lr = max(self.min_lr, self.initial_lr - self.decay_rate * epoch)
        self.optimizer.learning_rate = new_lr


class ExponentialScheduler(LRScheduler):
    """
    Exponential learning rate decay.

    lr = initial_lr * gamma^epoch
    """

    def __init__(self, optimizer, initial_lr: float, gamma: float = 0.95):
        super().__init__(optimizer, initial_lr)
        self.gamma = gamma

    @classmethod
    def from_config(cls, config, optimizer, initial_lr: float):
        """Create from configuration."""
        return cls(
            optimizer=optimizer,
            initial_lr=initial_lr,
            gamma=config.decay_rate,
        )

    def step(self, epoch: int):
        """Update learning rate exponentially"""
        new_lr = self.initial_lr * (self.gamma**epoch)
        self.optimizer.learning_rate = new_lr


class StepScheduler(LRScheduler):
    """
    Step learning rate decay.

    lr = initial_lr * gamma^(epoch // step_size)
    """

    def __init__(
        self, optimizer, initial_lr: float, step_size: int = 10, gamma: float = 0.1
    ):
        super().__init__(optimizer, initial_lr)
        self.step_size = step_size
        self.gamma = gamma

    @classmethod
    def from_config(cls, config, optimizer, initial_lr: float):
        """Create from configuration."""
        return cls(
            optimizer=optimizer,
            initial_lr=initial_lr,
            step_size=config.step_size,
            gamma=config.gamma,
        )

    def step(self, epoch: int):
        """Update learning rate in steps"""
        new_lr = self.initial_lr * (self.gamma ** (epoch // self.step_size))
        self.optimizer.learning_rate = new_lr


class CosineAnnealingScheduler(LRScheduler):
    """
    Cosine annealing learning rate decay.

    lr = eta_min + 0.5 * (initial_lr - eta_min) * (1 + cos(pi * epoch / T_max))
    """

    def __init__(
        self, optimizer, initial_lr: float, T_max: int = 50, eta_min: float = 0.0
    ):
        super().__init__(optimizer, initial_lr)
        self.T_max = T_max
        self.eta_min = eta_min

    @classmethod
    def from_config(cls, config, optimizer, initial_lr: float):
        """Create from configuration."""
        return cls(
            optimizer=optimizer,
            initial_lr=initial_lr,
            T_max=config.T_max,
            eta_min=config.eta_min,
        )

    def step(self, epoch: int):
        """Update learning rate with cosine annealing"""
        import math

        new_lr = self.eta_min + 0.5 * (self.initial_lr - self.eta_min) * (
            1 + math.cos(math.pi * epoch / self.T_max)
        )
        self.optimizer.learning_rate = new_lr
