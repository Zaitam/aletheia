from neural_net.models import ModelConfig
from neural_net.optimizers import OptimizerConfig, OptimizerType
from neural_net.training import (
    EarlyStoppingConfig,
    RegularizerConfig,
    TrainingConfig,
)

from .base import ExperimentConfig

# ============================================================================
# Baseline Models
# ============================================================================

M0_CONFIG = ExperimentConfig(
    name="M0",
    description="Baseline: 2 hidden layers [128, 64], SGD, no regularization",
    model=ModelConfig(hidden_layers=[128, 64]),
    training=TrainingConfig(
        batch_size=256,
        epochs=50,
        optimizer=OptimizerConfig(
            type=OptimizerType.SGD, learning_rate=0.001, momentum=0.0
        ),
        regularizer=RegularizerConfig(use_l2=False),
        early_stopping=EarlyStoppingConfig(enabled=False),
    ),
)

M1_CONFIG = ExperimentConfig(
    name="M1",
    description="With momentum: SGD with momentum=0.9",
    model=ModelConfig(hidden_layers=[128, 64]),
    training=TrainingConfig(
        batch_size=256,
        epochs=50,
        optimizer=OptimizerConfig(
            type=OptimizerType.SGD, learning_rate=0.001, momentum=0.9
        ),
        regularizer=RegularizerConfig(use_l2=False),
        early_stopping=EarlyStoppingConfig(enabled=False),
    ),
)

M2_CONFIG = ExperimentConfig(
    name="M2",
    description="With L2 regularization: lambda=0.01",
    model=ModelConfig(hidden_layers=[128, 64]),
    training=TrainingConfig(
        batch_size=256,
        epochs=50,
        optimizer=OptimizerConfig(
            type=OptimizerType.SGD, learning_rate=0.001, momentum=0.9
        ),
        regularizer=RegularizerConfig(use_l2=True, l2_lambda=0.01),
        early_stopping=EarlyStoppingConfig(enabled=False),
    ),
)

M3_CONFIG = ExperimentConfig(
    name="M3",
    description="With early stopping: patience=10",
    model=ModelConfig(hidden_layers=[128, 64]),
    training=TrainingConfig(
        batch_size=256,
        epochs=100,
        optimizer=OptimizerConfig(
            type=OptimizerType.SGD, learning_rate=0.001, momentum=0.9
        ),
        regularizer=RegularizerConfig(use_l2=True, l2_lambda=0.01),
        early_stopping=EarlyStoppingConfig(
            enabled=True, patience=10, monitor="val_loss"
        ),
    ),
)

# ============================================================================
# Alternative Optimizers
# ============================================================================

ADAM_CONFIG = ExperimentConfig(
    name="M_ADAM",
    description="Using Adam optimizer",
    model=ModelConfig(hidden_layers=[128, 64]),
    training=TrainingConfig(
        batch_size=256,
        epochs=50,
        optimizer=OptimizerConfig(
            type=OptimizerType.ADAM, learning_rate=0.001, beta1=0.9, beta2=0.999
        ),
        regularizer=RegularizerConfig(use_l2=False),
        early_stopping=EarlyStoppingConfig(enabled=False),
    ),
)

# ============================================================================
# Cross-Validation
# ============================================================================

CV_CONFIG = ExperimentConfig(
    name="M_CV",
    description="5-Fold stratified cross-validation",
    model=ModelConfig(hidden_layers=[128, 64]),
    training=TrainingConfig(
        batch_size=256,
        epochs=50,
        optimizer=OptimizerConfig(
            type=OptimizerType.SGD, learning_rate=0.001, momentum=0.9
        ),
        regularizer=RegularizerConfig(use_l2=True, l2_lambda=0.01),
        early_stopping=EarlyStoppingConfig(enabled=False),
        use_cross_validation=True,
        cv_folds=5,
        cv_stratified=True,
    ),
)
