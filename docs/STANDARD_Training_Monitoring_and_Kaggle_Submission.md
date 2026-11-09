# Standard - Training Monitoring and Per-Model Kaggle Submission

## 1. Scope

This is the mandatory project standard for every model experiment: CNN From Scratch V2, MobileNetV3, EfficientNet, ResNet50, YOLOv8 Classification, and any future architecture. It standardizes observable training, durable history, checkpoint selection, validation review, standalone inference, and a separate Kaggle CSV for each trained model.

The standard never permits actual training, internal-test evaluation, competition inference, or Kaggle upload to run automatically. Every executable stage remains manual opt-in in its notebook.

## 2. Required Notebook Workflow

Every model notebook must preserve this order:

1. Manual training with `RUN_TRAINING = False` by default.
2. Validation evaluation from `best.pt` with `RUN_VALIDATION_EVALUATION = False` by default.
3. Persistent training-history review and learning curves.
4. Validation classification report.
5. Validation confusion matrix.
6. Standalone competition-test inference setup.
7. Kaggle CSV generation with `RUN_TEST_INFERENCE = False` by default.
8. Optional CSV verification without inference rerun.

The 383-image validation split is the only split used for checkpoint selection and iterative experiment decisions. The 191-image internal test is a separate holdout and must not be used for submission or iterative tuning. The 480-image official test at `data/competition/test/` is the only image source for submissions.

## 3. Batch-Level Monitoring

Custom PyTorch training uses one `tqdm.auto` bar per epoch with `leave=False`, preventing repeated notebook lines at refresh. The description reports `Epoch current/max train`; postfix reports smoothed batch loss, running training accuracy, and current learning rate. tqdm provides batch/total, elapsed time, and ETA.

Monitoring is observational only. It does not alter AMP scaler behavior, loss scaling, optimizer stepping, gradient accumulation, clipping, or scheduler semantics.

## 4. Epoch-Level Monitoring and Persistent History

At each completed epoch, `TrainingManager` records and prints a single summary containing:

- epoch/current maximum, train loss and accuracy;
- validation loss, accuracy, macro precision, macro recall, and macro F1;
- learning rate and change status, epoch duration;
- current best epoch and best validation loss;
- early-stopping counter/status; and
- checkpoint status: `BEST SAVED` or `LAST UPDATED`.

Each experiment writes the same deduplicated actual history after every epoch:

- `outputs/metrics/{experiment_id}.json`
- `outputs/metrics/{experiment_id}.csv`
- `outputs/logs/{experiment_id}.jsonl`

The writer atomically replaces all files and keeps one latest record per epoch. Resume must start after the final persisted epoch; duplicate epoch writes are rejected. Required history fields include experiment ID, train/validation metrics, macro metrics, learning rate, duration, checkpoint status, and early-stopping counter.

`src.evaluation.training_monitoring` provides loading from JSON/CSV, full history, five epochs with lowest validation loss, final epoch, learning-rate changes, and early-stopping review.

## 5. Learning Curves

`plot_learning_curves` reads persisted history and writes `outputs/figures/{experiment_id}_learning_curves.png`. It includes train/validation loss, train/validation accuracy, validation macro F1, learning rate, a best-validation-loss marker, and an early-stopping marker where applicable. With `display_figure=True`, the same figure is displayed in a Jupyter notebook and then closed safely.

Plots must be regenerated from stored actual history after a kernel restart; no illustrative or synthetic metric plot may be shown as a training result.

## 6. Checkpoints and Model Adapters

For custom PyTorch models:

- `best.pt` is selected only by minimum `val_loss`.
- `last.pt` stores the latest epoch and resume state.
- early stopping monitors `val_loss`.
- validation evaluation and competition inference load `best.pt`.
- resume is manual only and restores history without duplicate records.

Checkpoint metadata must include model identifier, class mapping, model state, epoch, best epoch/metric, monitor/mode, training configuration, callback state, and history. Training accuracy never replaces the best checkpoint criterion.

YOLOv8 Classification must use an adapter for Ultralytics training/checkpoint formats rather than forcing the custom PyTorch manager. The adapter must normalize actual YOLO validation metrics into the project history schema, document metric-definition differences, expose an equivalent best/last selection policy where available, and return the same ordered class mapping for submission inference.

## 7. Standalone Inference and CSV Rules

Standalone inference must work from a new kernel without prior notebook cells. Its setup cell must resolve project root, add it to `sys.path` before `src` imports, load model config/class mapping/best checkpoint metadata from disk, choose CUDA or CPU, build validation-equivalent deterministic preprocessing, audit the official test/template pair, and resolve all output paths.

No standalone inference may depend on training variables such as `PROJECT_ROOT`, `EXPERIMENT_ID`, `CONFIG`, `DEVICE`, model objects, or prior loader state. The sample submission supplies only column names, ID representation, ID order, and row count. Its category values are never labels or predictions.

Each trained model writes only its own output in `Final Dataset for Submission to Kaggle/`:

| Model | Required output |
| --- | --- |
| CNN From Scratch V2 | `submission_cnn_v2.csv` |
| MobileNetV3 | `submission_mobilenetv3.csv` |
| EfficientNet | `submission_efficientnet.csv` |
| ResNet50 | `submission_resnet50.csv` |
| YOLOv8 Classification | `submission_yolov8.csv` |

Every CSV must have 480 rows with `image_id,waste_category`, preserve leading zeros and template order, contain no empty or duplicate IDs, use only the eight training classes, and use `overwrite=False` by default. Users upload to Kaggle manually; no auto-submit is allowed.

## 8. Model-Specific Compatibility

Pretrained models must use legitimate weights and their own validation-equivalent preprocessing from official library metadata/configuration. They must not inherit CNN From Scratch normalization by default. Fine-tuning phases are explicit, use a new experiment ID, and preserve the eight-class head/mapping.

The reusable history, plotting, checkpoint-review, competition audit, submission validation, and CSV writing contracts apply to all models. Only architecture construction, training backend, checkpoint adapter, and preprocessing may differ by model.

## 9. Test Requirements

Synthetic/unit tests must verify batch loss/accuracy/LR progress, complete epoch summaries, durable deduplicated history, plotting from history, best-checkpoint `val_loss` selection, early-stopping status, standalone setup without training variables, strict official-test-only data access, and valid CSV alignment. Tests must not run competition training, full 480-image inference, or internal-test evaluation.

## 10. Current Adoption Status

CNN From Scratch V2 (`notebooks/01b_cnn_from_scratch_v2.ipynb`) is the first notebook implementing this standard. CNN V1 remains preserved as a completed historical experiment. MobileNetV3, EfficientNet, ResNet50, and YOLOv8 notebooks are not yet implemented and must reuse this standard when added.
