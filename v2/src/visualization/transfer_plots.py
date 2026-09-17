"""Shared visualizations for all V2 Phase 2 transfer experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_transfer_training_curves(
    history: Sequence[dict[str, Any]],
    loss_path: str | Path,
    accuracy_path: str | Path,
    title_prefix: str,
) -> None:
    epochs = [row["epoch"] for row in history]
    loss_output = Path(loss_path)
    loss_output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, [row["train_loss"] for row in history], label="Train loss")
    axis.plot(
        epochs,
        [row["validation_loss"] for row in history],
        label="Validation loss",
    )
    axis.set(
        title=f"{title_prefix} loss",
        xlabel="Epoch",
        ylabel="Cross-entropy loss",
    )
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(loss_output, dpi=150)
    plt.close(figure)

    accuracy_output = Path(accuracy_path)
    accuracy_output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        epochs,
        [row["train_accuracy"] for row in history],
        label="Train accuracy",
    )
    axis.plot(
        epochs,
        [row["validation_accuracy"] for row in history],
        label="Validation accuracy",
    )
    axis.set(title=f"{title_prefix} accuracy", xlabel="Epoch", ylabel="Accuracy")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(accuracy_output, dpi=150)
    plt.close(figure)


def plot_transfer_confusion_matrix(
    matrix: np.ndarray,
    class_names: Sequence[str],
    output_path: str | Path,
    title: str,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
    figure.colorbar(image, ax=axis)
    positions = np.arange(len(class_names))
    axis.set(
        xticks=positions,
        yticks=positions,
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="Actual class",
        xlabel="Predicted class",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                format(int(matrix[row, column]), "d"),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)
