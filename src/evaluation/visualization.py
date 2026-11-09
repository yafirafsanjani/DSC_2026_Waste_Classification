"""Plot learning curves only from metrics emitted by a real training run."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence


def load_history(path: Path | str) -> list[dict[str, object]]:
    history_path = Path(path)
    if not history_path.is_file():
        raise FileNotFoundError(f"Metrics history does not exist: {history_path}")
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if not isinstance(history, list) or not history:
        raise ValueError("Metrics history is empty; no real learning curve is available.")
    return history


def plot_learning_curves(history: Sequence[Mapping[str, object]] | Path | str, output_path: Path | str,
                         *, title: str = "Learning Curves", display_figure: bool = False) -> Path:
    """Create loss, accuracy, LR, and macro-F1 plots from saved epoch metrics."""
    if isinstance(history, (str, Path)):
        history = load_history(history)
    if not history:
        raise ValueError("Metrics history is empty; refusing to create a synthetic chart.")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [int(record["epoch"]) + 1 for record in history]
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    axes[0, 0].plot(epochs, [float(row["train_loss"]) for row in history], label="train")
    axes[0, 0].plot(epochs, [float(row["val_loss"]) for row in history], label="validation")
    axes[0, 0].set_title("Loss")
    axes[0, 1].plot(epochs, [float(row["train_accuracy"]) for row in history], label="train")
    axes[0, 1].plot(epochs, [float(row["val_accuracy"]) for row in history], label="validation")
    axes[0, 1].set_title("Accuracy")
    axes[1, 0].plot(epochs, [float(row["learning_rates"][0]) for row in history])
    axes[1, 0].set_title("Learning Rate")
    axes[1, 1].plot(epochs, [float(row["val_macro_f1"]) for row in history])
    axes[1, 1].set_title("Validation Macro F1")
    best_epoch = min(history, key=lambda row: float(row["val_loss"]))["epoch"]
    for axis in axes.flat:
        axis.axvline(int(best_epoch) + 1, color="tab:green", linestyle="--", label="best val loss")
        if any(bool(row.get("early_stopping")) for row in history):
            stopped = next(row for row in history if row.get("early_stopping"))
            axis.axvline(int(stopped["epoch"]) + 1, color="tab:red", linestyle=":", label="early stop")
        axis.set_xlabel("Epoch")
        axis.legend()
        axis.grid(alpha=0.25)
    figure.suptitle(title)
    figure.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    if display_figure:
        try:
            from IPython.display import display
            display(figure)
        except ImportError:
            pass
    plt.close(figure)
    return destination
