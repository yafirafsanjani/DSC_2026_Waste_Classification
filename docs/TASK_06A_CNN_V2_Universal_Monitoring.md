# TASK 06A - CNN From Scratch V2 Universal Monitoring Implementation

## 1. Scope

TASK 06A implements the project-wide monitoring and per-model submission standard first in `notebooks/01b_cnn_from_scratch_v2.ipynb`. The notebook is unexecuted: it does not claim a V2 training result, checkpoint, validation metric, or Kaggle CSV.

## 2. Reusable Implementation

- `src/training/train.py` provides one Jupyter-compatible batch progress bar per epoch with smoothed loss, running accuracy, learning rate, batch/total, elapsed time, and ETA.
- `src/callbacks/training_manager.py` emits a complete epoch summary and atomically persists deduplicated JSON, CSV, and JSONL history after every actual epoch.
- `src/evaluation/training_monitoring.py` loads history, prevents duplicate epoch records, formats epoch summaries, and produces all-epoch, best-five, last-epoch, LR-change, and early-stopping reviews.
- `src/evaluation/visualization.py` renders loss, accuracy, macro F1, and LR curves with best/early-stop markers from stored history.

## 3. CNN V2 Notebook Workflow

The V2 notebook contains manual-training, validation, history/curves, classification report/confusion matrix, standalone inference, submission generation, and CSV verification sections. Its flags start as `False`:

- `RUN_TRAINING`
- `RUN_VALIDATION_EVALUATION`
- `RUN_TEST_INFERENCE`

The standalone inference cell can run first in a fresh kernel. It does not require the V2 training cell; it resolves project root and all inference context itself. Since V2 is not yet trained, it reports a missing `best.pt` without attempting inference. After manual V2 training produces `models/checkpoints/cnn_scratch_v2/best.pt`, the output target is `Final Dataset for Submission to Kaggle/submission_cnn_v2.csv`.

## 4. Verification

Synthetic monitoring tests verify batch postfix fields, atomic JSON/CSV/JSONL persistence, history deduplication, complete epoch summary fields, and review functions. The V2 standalone setup is smoke-tested from a clean process and notebook working directory without executing training or generating a submission.

## 5. Follow-On Models

MobileNetV3, EfficientNet, ResNet50, and YOLOv8 have not been implemented in this TASK. Their notebooks must reuse the contracts in `STANDARD_Training_Monitoring_and_Kaggle_Submission.md`, retain model-specific preprocessing, and write the designated per-model submission filename.
