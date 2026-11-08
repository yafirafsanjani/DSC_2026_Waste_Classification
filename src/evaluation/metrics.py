"""Classification metrics computed from complete prediction arrays."""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support


def ordered_class_names(class_mapping: Mapping[str, int]) -> list[str]:
    """Return names in the exact target-index order used by a model."""
    indices = sorted(class_mapping.values())
    if indices != list(range(len(indices))):
        raise ValueError("class_mapping indices must be contiguous from zero.")
    return [name for name, _ in sorted(class_mapping.items(), key=lambda item: item[1])]


def compute_classification_metrics(targets: Sequence[int] | np.ndarray, predictions: Sequence[int] | np.ndarray, class_mapping: Mapping[str, int]) -> dict[str, object]:
    """Return global and per-class metrics from every sample, not batch averages."""
    target_array, prediction_array = np.asarray(targets, dtype=np.int64), np.asarray(predictions, dtype=np.int64)
    if target_array.ndim != 1 or prediction_array.ndim != 1 or len(target_array) != len(prediction_array):
        raise ValueError("targets and predictions must be equally sized one-dimensional arrays.")
    if not len(target_array):
        raise ValueError("Cannot compute metrics for zero samples.")
    names, labels = ordered_class_names(class_mapping), list(range(len(class_mapping)))
    precision, recall, f1, support = precision_recall_fscore_support(target_array, prediction_array, labels=labels, zero_division=0)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(target_array, prediction_array, labels=labels, average="macro", zero_division=0)
    return {
        "accuracy": float(accuracy_score(target_array, prediction_array)),
        "macro_precision": float(macro_precision), "macro_recall": float(macro_recall), "macro_f1": float(macro_f1),
        "per_class": {name: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])} for i, name in enumerate(names)},
        "confusion_matrix": confusion_matrix(target_array, prediction_array, labels=labels).tolist(),
        "classification_report": classification_report(target_array, prediction_array, labels=labels, target_names=names, output_dict=True, zero_division=0),
        "sample_count": int(len(target_array)),
    }
