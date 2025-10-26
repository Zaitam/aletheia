from .config import ModelConfig
from .mlp import MLP


def create_mlp(
    model_config: ModelConfig,
    input_dim: int,
    output_dim: int,
) -> MLP:
    """Create an MLP model from configuration."""
    return MLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_layers=model_config.hidden_layers,
        activation=model_config.activation,
        output_activation=model_config.output_activation,
        dropout_rate=model_config.dropout_rate,
    )
