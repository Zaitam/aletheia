from abc import ABC, abstractmethod


class BaseMLP(ABC):
    """
    Abstract base class for Multi-Layer Perceptron models.
    All MLP implementations (CuPy, PyTorch) should inherit from this class.
    """

    def __init__(
        self, input_dim: int, output_dim: int, hidden_layers: list[int], batch_size: int
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.batch_size = batch_size

    @abstractmethod
    def forward(self, X):
        """Forward pass through the network"""
        pass

    @abstractmethod
    def backward(self, y_true):
        """Backward pass to compute gradients"""
        pass

    @abstractmethod
    def get_gradients(self):
        """Return current gradients for optimizer"""
        pass

    @abstractmethod
    def update_parameters(self, updates):
        """Update parameters given updates from optimizer"""
        pass

    @abstractmethod
    def save_weights(self, filepath: str):
        """Save model weights to file"""
        pass

    @abstractmethod
    def load_weights(self, filepath: str):
        """Load model weights from file"""
        pass

    @abstractmethod
    def predict(self, X):
        """Make predictions (forward pass without storing gradients)"""
        pass
