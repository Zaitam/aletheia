"""
Visualization utilities for plotting results.
"""
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Get style file path (one directory up from src/)
STYLE_PATH = Path(__file__).parent.parent.parent / "figs.mplstyle"


def plot_training_history(history, save_path=None):
    """
    Plot training and validation loss/accuracy.

    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Optional path to save figure
    """
    # Use custom style if available
    if STYLE_PATH.exists():
        plt.style.use(str(STYLE_PATH))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_confusion_matrix(cm, class_names=None, save_path=None):
    """
    Plot confusion matrix.

    Args:
        cm: Confusion matrix (numpy array)
        class_names: Optional list of class names
        save_path: Optional path to save figure
    """
    # Use custom style if available
    if STYLE_PATH.exists():
        plt.style.use(str(STYLE_PATH))

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def visualize_sample_images(X_images, y_images, num_samples=3, figsize=(12, 4)):
    """
    Visualize sample images from the dataset.

    Args:
        X_images: Image data (n_samples, 28, 28) or (n_samples, 784)
        y_images: Labels (n_samples,)
        num_samples: Number of samples to visualize
        figsize: Figure size
    """
    # Use custom style if available
    if STYLE_PATH.exists():
        plt.style.use(str(STYLE_PATH))
    plt.rcParams["figure.dpi"] = 300

    fig, axes = plt.subplots(1, num_samples, figsize=figsize)

    for i in range(num_samples):
        img = X_images[i].reshape(28, 28)

        if num_samples == 1:
            ax = axes
        else:
            ax = axes[i]

        ax.imshow(img, cmap="gray")
        ax.set_title(f"Sample {i+1}\nClass: {y_images[i]}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_one_per_class(X, y, save_path='figures/one_per_class.png', figsize=(16, 12)):
    """
    Visualize one image per class in a grid.

    Args:
        X: Image data (n_samples, 28, 28) or (n_samples, 784)
        y: Labels (n_samples,)
        save_path: Path to save figure
        figsize: Figure size
    """
    import numpy as np
    from pathlib import Path

    # Use custom style if available
    if STYLE_PATH.exists():
        plt.style.use(str(STYLE_PATH))
    plt.rcParams["figure.dpi"] = 300

    # Ensure figures directory exists
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Get unique classes
    unique_classes = np.unique(y)
    n_classes = len(unique_classes)

    # Create grid (7x7 = 49, good for 47 classes)
    n_rows = 7
    n_cols = 7

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    # Plot one image per class
    for idx, class_label in enumerate(unique_classes):
        # Find first occurrence of this class
        class_indices = np.where(y == class_label)[0]
        sample_idx = class_indices[0]

        # Get image
        img = X[sample_idx].reshape(28, 28)

        # Plot
        axes[idx].imshow(img, cmap='gray')
        axes[idx].set_title(f'Class {int(class_label)}', fontsize=8)
        axes[idx].axis('off')

    # Hide unused subplots
    for idx in range(n_classes, n_rows * n_cols):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")
    plt.show()


def plot_class_distribution(y, save_path='figures/class_distribution.png', figsize=(14, 6)):
    """
    Plot the distribution of classes in the dataset.

    Args:
        y: Labels (n_samples,)
        save_path: Path to save figure
        figsize: Figure size
    """
    import numpy as np
    from pathlib import Path

    # Use custom style if available
    if STYLE_PATH.exists():
        plt.style.use(str(STYLE_PATH))
    plt.rcParams["figure.dpi"] = 300

    # Ensure figures directory exists
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Count samples per class
    unique_classes, counts = np.unique(y, return_counts=True)

    # Create bar plot
    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.bar(unique_classes, counts, color='#4165c0', alpha=0.8, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Class Label', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('Class Distribution in EMNIST Dataset', fontsize=14)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")
    plt.show()


def plot_pixel_statistics(X, save_path='figures/pixel_statistics.png', figsize=(12, 5)):
    """
    Plot mean and std of pixel values across the dataset.

    Args:
        X: Image data (n_samples, 28, 28) or (n_samples, 784)
        save_path: Path to save figure
        figsize: Figure size
    """
    import numpy as np
    from pathlib import Path

    # Use custom style if available
    if STYLE_PATH.exists():
        plt.style.use(str(STYLE_PATH))
    plt.rcParams["figure.dpi"] = 300

    # Ensure figures directory exists
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Reshape if needed
    if X.ndim == 2 and X.shape[1] == 784:
        X_reshaped = X.reshape(-1, 28, 28)
    else:
        X_reshaped = X

    # Compute statistics
    mean_img = np.mean(X_reshaped, axis=0)
    std_img = np.std(X_reshaped, axis=0)

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Mean
    im1 = ax1.imshow(mean_img, cmap='viridis')
    ax1.set_title('Mean Pixel Values', fontsize=12)
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # Std
    im2 = ax2.imshow(std_img, cmap='viridis')
    ax2.set_title('Std Pixel Values', fontsize=12)
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")
    plt.show()
