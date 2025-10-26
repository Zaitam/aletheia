from abc import ABC, abstractmethod
import cupy as cp
from enum import Enum


class LayerType(Enum):
    LINEAR = "linear"
    RELU = "relu"
    SOFTMAX = "softmax"
    DROPOUT = "dropout"


class Layer(ABC):
    """Base class for all layers"""

    def __init__(self, input_dim: int = None, output_dim: int = None):
        self.input_dim = input_dim
        self.output_dim = output_dim

    @abstractmethod
    def forward(self, x: cp.ndarray):
        """
        Forward pass through the layer.
        """

    @abstractmethod
    def evaluate(self, x: cp.ndarray):
        """
        Evaluate layer output without storing intermediate values.
        """

    @abstractmethod
    def backward(self, prev_grad: cp.ndarray):
        """
        Backward pass through the layer.
        """


class Linear(Layer):
    """Fully connected linear layer"""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__(input_dim, output_dim)
        # Initialize weights and biases
        # It does not depend on batch size but gradients do, hence we set it during backward
        # and not force set the input size as (input_dim, batch_size) here, to allow broadcasting.
        # He initialization for ReLU networks -> https://arxiv.org/pdf/1502.01852.pdf
        self.weights = cp.random.randn(output_dim, input_dim).astype(
            cp.float32
        ) * cp.sqrt(2.0 / input_dim)
        # Bias as a column vector [output_dim, 1] -> broadcasted during addition
        self.bias = cp.zeros((output_dim, 1), dtype=cp.float32)

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        # weight: (output_dim, input_dim) @ x: (input_dim, batch_size) -> (output_dim, batch_size)
        # bias: (output_dim, batch_size)
        # return: (output_dim, batch_size)
        self.input = x
        return self.weights @ x + self.bias

    def evaluate(self, x: cp.ndarray) -> cp.ndarray:
        return self.weights @ x + self.bias

    def backward(self, prev_grad: cp.ndarray) -> cp.ndarray:
        # prev_grad: (output_dim, batch_size), input: (input_dim, batch_size)
        # Weights: DL/DZ3 * (a^i-1)^T
        self.grad_weights = prev_grad @ self.input.T  # (output_dim, input_dim)
        # Bias gradient is the sum over the batch dimension -> (output_dim, 1)
        self.grad_bias = cp.sum(prev_grad, axis=1, keepdims=True)

        # Layer is independant of the batch size (which is handled in the optimizer)

        return self.weights.T @ prev_grad


class Relu(Layer):
    """ReLU activation function"""

    def forward(self, x: cp.ndarray):
        self.mask = cp.array(x > 0, dtype=cp.float32)
        return x * self.mask

    def evaluate(self, x: cp.ndarray) -> cp.ndarray:
        return cp.maximum(0, x)

    def backward(self, prev_grad: cp.ndarray):
        return prev_grad * self.mask


class CESoftmax(Layer):
    """Combined Cross-Entropy Loss + Softmax activation

    This combines softmax activation with cross-entropy loss for numerical stability.
    The backward pass computes the gradient of CE loss w.r.t. logits.
    """

    def forward(self, logits: cp.ndarray):
        # Subtract max for numerical stability
        exp_input = cp.exp(logits - cp.max(logits, axis=0, keepdims=True))
        # Returns only softmax output
        self.output = exp_input / cp.sum(exp_input, axis=0, keepdims=True)
        return self.output

    def evaluate(self, logits: cp.ndarray) -> cp.ndarray:
        exp_input = cp.exp(logits - cp.max(logits, axis=0, keepdims=True))
        return exp_input / cp.sum(exp_input, axis=0, keepdims=True)

    def backward(self, prev_grad: cp.ndarray):
        # NOTE: This backward is the CrossEntropy + Softmax derivative (Not pure Softmax)
        return self.output - prev_grad


class Dropout(Layer):
    """Dropout layer for regularization"""

    def __init__(self, drop_prob: float):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: cp.ndarray):
        # Randomly drop units during training
        self.mask = (cp.random.rand(*x.shape) > self.drop_prob).astype(cp.float32)
        return x * self.mask / (1.0 - self.drop_prob)  # Scale to keep expected value

    def evaluate(self, x: cp.ndarray) -> cp.ndarray:
        return x  # No dropout during evaluation

    def backward(self, prev_grad: cp.ndarray):
        return (
            prev_grad * self.mask / (1.0 - self.drop_prob)
        )  # Scale gradient accordingly
