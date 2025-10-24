"""
Experiment runner.
Ties together model, optimizer, trainer, and evaluator.
"""

import cupy as cp
from ..models import MLP
from ..optimizers import SGD, Adam
from ..training import (
    Trainer,
    LinearScheduler,
    ExponentialScheduler,
    L2Regularizer,
    EarlyStopping,
)
from ..evaluation import Evaluator
from .config import ExperimentConfig


class Experiment:
    """
    Runs a complete experiment given a configuration.

    This is the high-level interface that puts everything together.
    """

    def __init__(self, config: ExperimentConfig, dataset):
        self.config = config
        self.dataset = dataset

        # Set random seed
        cp.random.seed(config.random_seed)

        # Initialize components
        self.model = None
        self.optimizer = None
        self.trainer = None
        self.evaluator = Evaluator()

    def build_model(self):
        """Build model based on config"""
        input_dim = self.dataset.X_train.shape[1] * self.dataset.X_train.shape[2]
        output_dim = len(cp.unique(self.dataset.y_train))

        self.model = MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_layers=self.config.hidden_layers,
            batch_size=self.config.batch_size,
        )

    def build_optimizer(self):
        """Build optimizer based on config"""
        if self.config.optimizer == "sgd":
            self.optimizer = SGD(
                learning_rate=self.config.learning_rate, momentum=self.config.momentum
            )
        elif self.config.optimizer == "adam":
            self.optimizer = Adam(
                learning_rate=self.config.learning_rate,
                beta1=self.config.adam_beta1,
                beta2=self.config.adam_beta2,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def build_trainer(self):
        """Build trainer with all components"""
        # Learning rate scheduler
        lr_scheduler = None
        if self.config.use_lr_scheduler:
            if self.config.lr_scheduler_type == "linear":
                lr_scheduler = LinearScheduler(
                    self.optimizer, self.config.learning_rate, self.config.lr_decay_rate
                )
            elif self.config.lr_scheduler_type == "exponential":
                lr_scheduler = ExponentialScheduler(
                    self.optimizer, self.config.learning_rate, self.config.lr_gamma
                )

        # Regularizer
        regularizer = None
        if self.config.use_l2:
            regularizer = L2Regularizer(self.config.l2_lambda)

        # Early stopping
        early_stopping = None
        if self.config.use_early_stopping:
            early_stopping = EarlyStopping(patience=self.config.early_stopping_patience)

        self.trainer = Trainer(
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=lr_scheduler,
            regularizer=regularizer,
            early_stopping=early_stopping,
        )

    def run(self):
        """Run the complete experiment"""
        print(f"\n{'='*60}")
        print(f"Running experiment: {self.config.name}")
        print(f"Description: {self.config.description}")
        print(f"{'='*60}\n")

        # Build components
        self.build_model()
        self.build_optimizer()
        self.build_trainer()

        # Get data
        X_train, y_train = self.dataset.get_train()
        X_val, y_val = self.dataset.get_val()
        X_test, y_test = self.dataset.get_test()

        # Train
        print("Training...")
        history = self.trainer.train(
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            verbose=True,
        )

        # Evaluate on test
        print("\nEvaluating on test set...")
        test_metrics = self.evaluator.evaluate(
            self.model, X_test, y_test, self.config.name
        )

        print(f"\nTest Results:")
        print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  F1-Score: {test_metrics['f1_macro']:.4f}")

        return {"history": history, "test_metrics": test_metrics, "model": self.model}
