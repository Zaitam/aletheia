"""
Neural network layers (Linear, ReLU, Softmax, etc.)
These are the building blocks for different MLP architectures.
"""
import cupy as cp


class Layer:
    """Base class for all layers"""
    def forward(self, _):
        raise NotImplementedError

    def backward(self, _):
        raise NotImplementedError


class Linear(Layer):
    """Fully connected linear layer"""
    def __init__(self, input_dim: int, output_dim: int, batch_size: int):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size

        # He initialization for ReLU networks -> https://arxiv.org/pdf/1502.01852.pdf
        self.weights = cp.random.randn(output_dim, input_dim).astype(
            cp.float32
        ) * cp.sqrt(2.0 / input_dim)
        # Bias as a column vector [output_dim, 1] -> broadcasted during addition
        self.bias = cp.zeros((output_dim, 1), dtype=cp.float32)

        # cache para backward
        self.input = cp.zeros((input_dim, batch_size), dtype=cp.float32)
        self.grad_weights = cp.zeros((output_dim, input_dim), dtype=cp.float32)
        self.grad_bias = cp.zeros((output_dim, 1), dtype=cp.float32)

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        # weight: (output_dim, input_dim) @ x: (input_dim, batch_size) -> (output_dim, batch_size)
        # bias: (output_dim, batch_size)
        # return: (output_dim, batch_size)
        self.input = x
        return self.weights @ x + self.bias

    def backward(self, prev_grad: cp.ndarray) -> cp.ndarray:
        # prev_grad: (output_dim, batch_size), input: (input_dim, batch_size)
        # Weights: DL/DZ3 * (a^i-1)^T
        self.grad_weights = (
            prev_grad @ self.input.T
        ) / self.batch_size  # (output_dim, input_dim)
        # Bias gradient is the sum over the batch dimension -> (output_dim, 1)
        self.grad_bias = cp.sum(prev_grad, axis=1, keepdims=True) / self.batch_size

        return self.weights.T @ prev_grad


class Relu(Layer):
    """ReLU activation function"""
    def forward(self, x: cp.ndarray):
        self.mask = cp.array(x > 0, dtype=cp.float32)
        return x * self.mask

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

    def backward(self, prev_grad: cp.ndarray):
        # NOTE: This backward is the CrossEntropy + Softmax derivative (Not pure Softmax)
        return self.output - prev_grad
