from dataclasses import dataclass

from neural_net.models import ModelConfig
from neural_net.training import TrainingConfig


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment with metadata."""

    name: str
    description: str
    model: ModelConfig
    training: TrainingConfig

    def __str__(self) -> str:
        return f"{self.name}: {self.description}"

    def __repr__(self) -> str:
        return (
            f"ExperimentConfig(\n"
            f"  name='{self.name}',\n"
            f"  description='{self.description}',\n"
            f"  model={self.model},\n"
            f"  training={self.training}\n"
            f")"
        )
