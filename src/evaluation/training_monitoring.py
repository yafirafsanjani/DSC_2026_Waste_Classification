"""Persistent training-history and notebook-friendly monitoring utilities."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


HISTORY_CSV_COLUMNS = (
    "experiment_id", "epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy",
    "val_macro_precision", "val_macro_recall", "val_macro_f1", "learning_rate", "learning_rates",
    "previous_learning_rates", "learning_rate_changed", "epoch_seconds", "best_epoch", "best_validation_loss",
    "early_stopping_counter", "early_stopping", "early_stopping_reason", "checkpoint_status",
    "checkpoint_best_saved", "checkpoint_last",
)


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="")
    temporary.replace(path)


def deduplicate_history(history: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep one latest record per epoch, sorted, so resume cannot duplicate rows."""
    by_epoch: dict[int, dict[str, Any]] = {}
    for raw_record in history:
        record = dict(raw_record)
        if "epoch" not in record:
            raise ValueError("Training history record is missing epoch.")
        epoch = int(record["epoch"])
        if epoch < 0:
            raise ValueError("Training history epoch cannot be negative.")
        record["epoch"] = epoch
        by_epoch[epoch] = record
    return [by_epoch[epoch] for epoch in sorted(by_epoch)]


def persist_training_history(
    history: Sequence[Mapping[str, Any]], *, json_path: Path | str, csv_path: Path | str, jsonl_path: Path | str,
) -> list[dict[str, Any]]:
    """Atomically replace JSON, CSV, and JSONL history from the same deduplicated epoch records."""
    normalized = deduplicate_history(history)
    _atomic_write(Path(json_path), json.dumps(normalized, indent=2) + "\n")

    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=HISTORY_CSV_COLUMNS, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for record in normalized:
        row = dict(record)
        for key in ("learning_rates", "previous_learning_rates"):
            if key in row and not isinstance(row[key], str):
                row[key] = json.dumps(row[key])
        writer.writerow(row)
    _atomic_write(Path(csv_path), csv_buffer.getvalue())
    _atomic_write(Path(jsonl_path), "".join(json.dumps(record) + "\n" for record in normalized))
    return normalized


def load_training_history(path: Path | str) -> list[dict[str, Any]]:
    """Load actual persisted JSON or CSV history; no synthetic fallback is created."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Training history does not exist: {source}")
    if source.suffix.lower() == ".json":
        loaded = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(loaded, list):
            raise ValueError("JSON training history must be a list.")
        return deduplicate_history(loaded)
    if source.suffix.lower() == ".csv":
        with source.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        records: list[dict[str, Any]] = []
        integer_fields = {"epoch", "best_epoch", "early_stopping_counter"}
        float_fields = {
            "train_loss", "train_accuracy", "val_loss", "val_accuracy", "val_macro_precision", "val_macro_recall",
            "val_macro_f1", "learning_rate", "epoch_seconds", "best_validation_loss",
        }
        boolean_fields = {"learning_rate_changed", "early_stopping", "checkpoint_best_saved"}
        for row in rows:
            record: dict[str, Any] = dict(row)
            for key in integer_fields:
                if record.get(key) not in {None, ""}:
                    record[key] = int(record[key])
            for key in float_fields:
                if record.get(key) not in {None, ""}:
                    record[key] = float(record[key])
            for key in boolean_fields:
                if record.get(key) not in {None, ""}:
                    record[key] = str(record[key]).strip().lower() == "true"
            for key in ("learning_rates", "previous_learning_rates"):
                if record.get(key):
                    record[key] = json.loads(record[key])
            records.append(record)
        return deduplicate_history(records)
    raise ValueError(f"Unsupported training history format: {source.suffix}")


def format_epoch_summary(record: Mapping[str, Any], max_epochs: int | None = None) -> str:
    """Return a compact single-line epoch summary suitable for notebooks and terminals."""
    epoch_label = str(int(record["epoch"]) + 1)
    if max_epochs is not None:
        epoch_label = f"{epoch_label}/{max_epochs}"
    learning_rate = float(record.get("learning_rate", record.get("learning_rates", [0.0])[0]))
    checkpoint_status = record.get("checkpoint_status", "LAST UPDATED")
    return (
        f"Epoch {epoch_label} | train_loss={float(record['train_loss']):.4f} "
        f"train_acc={float(record['train_accuracy']):.2%} | val_loss={float(record['val_loss']):.4f} "
        f"val_acc={float(record['val_accuracy']):.2%} | macro_f1={float(record['val_macro_f1']):.4f} "
        f"lr={learning_rate:.2e} | {float(record['epoch_seconds']):.1f}s | "
        f"best_epoch={int(record['best_epoch']) + 1} best_val_loss={float(record['best_validation_loss']):.4f} | "
        f"early_stop={int(record['early_stopping_counter'])} | {checkpoint_status}"
    )


def history_review(history: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    """Return complete, best-five, last, LR-change, and early-stopping review tables."""
    if not history:
        raise ValueError("Training history is empty.")
    import pandas as pd

    frame = pd.DataFrame(deduplicate_history(history)).sort_values("epoch").reset_index(drop=True)
    lr_changes = frame[frame["learning_rate_changed"].astype(bool)] if "learning_rate_changed" in frame else frame.iloc[0:0]
    early_stop = frame[frame["early_stopping"].astype(bool)] if "early_stopping" in frame else frame.iloc[0:0]
    return {
        "all_epochs": frame,
        "best_five_by_val_loss": frame.nsmallest(min(5, len(frame)), "val_loss"),
        "last_epoch": frame.tail(1),
        "learning_rate_changes": lr_changes,
        "early_stopping": early_stop,
    }
