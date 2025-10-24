import cupy as cp


class Dataset:
    """
    Handles data loading and splitting into train/val/test.

    This ensures ALL experiments use the same data splits,
    making comparisons fair.
    """

    def __init__(
        self,
        X_path: str,
        y_path: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        normalize: bool = True,
        random_seed: int = 42,
    ):
        """
        Load and split dataset.

        Args:
            X_path: Path to X_images.npy
            y_path: Path to y_images.npy
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            test_ratio: Fraction for testing
            normalize: Whether to normalize images to [0, 1]
            random_seed: Random seed for reproducibility
        """
        assert (
            abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        ), "Ratios must sum to 1.0"

        self.X = cp.load(X_path)
        self.y = cp.load(y_path)

        if normalize:
            self.X = self.X.astype(cp.float32) / 255.0

        cp.random.seed(random_seed)
        n_samples = self.X.shape[0]
        indices = cp.random.permutation(n_samples)

        train_end = int(n_samples * train_ratio)
        val_end = train_end + int(n_samples * val_ratio)

        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        self.X_train = self.X[train_idx]
        self.y_train = self.y[train_idx]

        self.X_val = self.X[val_idx]
        self.y_val = self.y[val_idx]

        self.X_test = self.X[test_idx]
        self.y_test = self.y[test_idx]

        print(f"Dataset loaded:")
        print(f"  Train: {len(self.X_train)} samples")
        print(f"  Val: {len(self.X_val)} samples")
        print(f"  Test: {len(self.X_test)} samples")

    def get_train(self):
        """Get training data"""
        return self.X_train, self.y_train

    def get_val(self):
        """Get validation data"""
        return self.X_val, self.y_val

    def get_test(self):
        """Get test data"""
        return self.X_test, self.y_test
