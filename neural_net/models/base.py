from abc import ABC, abstractmethod


class BaseMLP(ABC):
    """Abstract base class for Multi-Layer Perceptron models."""

    def __init__(self, input_dim: int, output_dim: int, hidden_layers: list[int]):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers

    @abstractmethod
    def forward(self, X):
        """Forward pass through the network"""

    @abstractmethod
    def backward(self, y_true):
        """Backward pass to compute gradients"""

    @abstractmethod
    def get_gradients(self):
        """Return current gradients for optimizer"""

    @abstractmethod
    def update_parameters(self, updates):
        """Update parameters given updates from optimizer"""

    @abstractmethod
    def save_weights(self, filepath: str):
        """Save model weights to file"""

    @abstractmethod
    def load_weights(self, filepath: str):
        """Load model weights from file"""

    @abstractmethod
    def predict(self, X):
        """Make predictions (forward pass without storing gradients)"""
