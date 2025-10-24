"""
Simple test script to verify new interface works.
This is a minimal example to test the complete pipeline.
"""
import cupy as cp
from models import MLP
from optimizers import SGD
from training import Trainer

# Load a small subset of data for quick testing
print("Loading data...")
X_images = cp.load("../data/X_images.npy")[:1000]  # Only 1000 samples for quick test
y_images = cp.load("../data/y_images.npy")[:1000]

# Normalize
X_images = X_images.astype(cp.float32) / 255.0

# Split manually (simple split, no stratification for now)
n_train = 700
X_train = X_images[:n_train]
y_train = y_images[:n_train]
X_val = X_images[n_train:]
y_val = y_images[n_train:]

print(f"Train: {X_train.shape}, Val: {X_val.shape}")

# Create model
print("\nBuilding model...")
input_dim = 28 * 28
output_dim = len(cp.unique(y_images))
hidden_layers = [128, 64]
batch_size = 128

model = MLP(
    input_dim=input_dim,
    output_dim=output_dim,
    hidden_layers=hidden_layers,
    batch_size=batch_size
)

# Create optimizer
optimizer = SGD(learning_rate=0.001, momentum=0.0)

# Create trainer
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    lr_scheduler=None,
    regularizer=None,
    early_stopping=None
)

# Train
print("\nTraining...")
history = trainer.train(
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    epochs=3,  # Only 3 epochs for quick test
    batch_size=batch_size,
    verbose=True
)

print("\nTraining complete!")
print(f"Final train loss: {history['train_loss'][-1]:.4f}")
print(f"Final train acc: {history['train_acc'][-1]:.4f}")
print(f"Final val loss: {history['val_loss'][-1]:.4f}")
print(f"Final val acc: {history['val_acc'][-1]:.4f}")
