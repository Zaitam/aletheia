from .config import EarlyStoppingConfig, RegularizerConfig, SchedulerConfig
from .early_stopping import EarlyStopping
from .lr_schedulers import LRScheduler
from .regularizers import L1Regularizer, L2Regularizer, Regularizer


def create_scheduler(
    config: SchedulerConfig, optimizer, initial_lr: float
) -> LRScheduler:
    """Create a learning rate scheduler from configuration."""
    scheduler_class = config.type.value
    return scheduler_class.from_config(config, optimizer, initial_lr)


def create_regularizer(config: RegularizerConfig) -> Regularizer | None:
    """
    Create a regularizer from configuration.

    Note: Currently only supports one type of regularization at a time.
    If multiple are enabled, L2 takes precedence over L1.
    """
    if config.use_l2:
        return L2Regularizer(lambda_=config.l2_lambda)
    elif config.use_l1:
        return L1Regularizer(lambda_=config.l1_lambda)
    # Note: Dropout is handled at the model level, not here
    return None


def create_early_stopping(config: EarlyStoppingConfig) -> EarlyStopping | None:
    """Create an early stopping instance from configuration."""
    if not config.enabled:
        return None

    return EarlyStopping(
        patience=config.patience,
        monitor=config.monitor,
        min_delta=config.min_delta,
        restore_best_weights=config.restore_best_weights,
    )
