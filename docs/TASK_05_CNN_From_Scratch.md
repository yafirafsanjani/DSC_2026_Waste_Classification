# TASK 05B - Final CNN From Scratch Training Results

## 1. Scope and Evidence

Dokumen ini mencatat hasil eksperimen yang sudah dijalankan secara manual oleh pengguna melalui `notebooks/01_cnn_from_scratch.ipynb`. Tidak ada training, resume, continued training, inference, atau evaluasi internal test yang dijalankan saat TASK 05B. Angka di bawah diverifikasi dari:

- output eksekusi notebook, termasuk validation classification report dan confusion matrix;
- `outputs/metrics/cnn_scratch_baseline_v1.json` dengan 75 record epoch;
- `models/checkpoints/cnn_scratch_baseline_v1/best.pt` dan `models/checkpoints/cnn_scratch_baseline_v1/last.pt`; dan
- `outputs/figures/cnn_scratch_baseline_v1_learning_curves.png`.

Ringkasan terstruktur untuk notebook perbandingan model tersedia di `outputs/metrics/cnn_scratch_baseline_v1_summary.json`.

## 2. Dataset and Split

Eksperimen memakai dataset V1 final melalui manifest aktif. Semua source record tervalidasi berasal dari `data/extracted/v1/train/`; tidak ada data `v1/test` maupun `v5` yang digunakan.

| Split | Samples |
| --- | ---: |
| Train | 1,340 |
| Validation | 383 |
| Internal test | 191 (tidak digunakan) |
| Total | 1,914 |

Kelas yang digunakan: `cardboard`, `electronic`, `foodwaste`, `glass`, `metal`, `paper`, `plastic`, dan `textile`.

## 3. Actual Training Configuration

| Item | Actual value |
| --- | --- |
| Experiment ID | `cnn_scratch_baseline_v1` |
| Seed | 42 |
| Device recorded in notebook | CUDA (NVIDIA GeForce RTX 3050 Laptop GPU) |
| Input | RGB, direct resize 224 x 224, bilinear, float32, scaling [0, 1], no additional normalization |
| Train augmentation | Enabled, profile `cnn_standard` |
| Validation augmentation | Disabled; deterministic transform |
| Batch size | 16 |
| Loss | CrossEntropyLoss |
| Optimizer | AdamW, learning rate 0.001, weight decay 0.0001 |
| AMP | Enabled |
| Gradient accumulation | 1 step |
| Gradient clipping | Norm 1.0 |
| Scheduler | ReduceLROnPlateau on `val_loss`, factor 0.5, patience 4, minimum LR 0.000001 |
| Early stopping | `val_loss`, mode `min`, patience 10, min_delta 0.001 |
| Requested maximum | 100 epochs |
| Completed epochs | 75 epochs (stored as epoch 0 through 74) |

The learning rate stepped down from 0.001 to 0.0000078125. The per-epoch history records the actual rate, losses, accuracies, macro metrics, duration, checkpoint status, and early-stopping status.

## 4. CNN Architecture and Parameters

`CNNFromScratch` is trained without pretrained weights. It contains four blocks with channel progression 32 -> 64 -> 128 -> 192. Each block consists of two 3 x 3 bias-free convolutions, BatchNorm, ReLU, and MaxPool2d. The classifier is AdaptiveAvgPool2d(1, 1), Flatten, Linear(192, 256), ReLU, Dropout(0.40), and Linear(256, 8).

- Trainable parameters: **892,648**.
- Adaptive pooling keeps the classifier small and supports the 224 x 224 input configuration.
- The saved checkpoint confirms model identifier `cnn_from_scratch`, eight-class mapping, and seed 42.

## 5. Training Outcome and Checkpoint Selection

The checkpoint monitor is `val_loss` in `min` mode. Therefore `best.pt` represents the minimum validation loss, not necessarily the epoch with the maximum validation accuracy or macro F1.

| Metric | Best checkpoint (epoch 64) | Last epoch (epoch 74) |
| --- | ---: | ---: |
| Train loss | 1.036100 | 1.021970 |
| Train accuracy | 65.67% | 64.40% |
| Validation loss | **1.248915** | 1.302490 |
| Validation accuracy | 59.01% | 59.27% |
| Validation macro precision | 59.36% | 60.45% |
| Validation macro recall | 59.12% | 59.28% |
| Validation macro F1 | **58.69%** | 58.86% |
| Learning rate | 0.00003125 | 0.0000078125 |

The best checkpoint is epoch index 64 (the 65th completed epoch when counted from one). Its checkpoint payload records `best_metric = 1.2489149981316947` and a history length of 65. The last checkpoint is epoch index 74 and includes all 75 records.

Early stopping occurred at epoch 74 because `val_loss` did not improve by at least 0.001 for 10 consecutive epochs after the epoch-64 best result. Consequently, the configured 100-epoch maximum was not reached.

### Difference Between Best Checkpoint and Peak Accuracy/F1

Validation accuracy reaches its maximum of **59.79%** at epoch 72, and validation macro F1 reaches its maximum of **59.38%** at epoch 68. Neither epoch improved `val_loss` below 1.248915, so neither replaced `best.pt`. The validation report and confusion matrix in this document are from `best.pt` at epoch 64, which is the intended selection under the configured monitor.

## 6. Learning Curves

The recorded training/validation curves are stored at `outputs/figures/cnn_scratch_baseline_v1_learning_curves.png`.

![CNN From Scratch learning curves](../outputs/figures/cnn_scratch_baseline_v1_learning_curves.png)

The curves show substantial early progress: validation loss declines from 1.839558 at epoch 0 to 1.248915 at epoch 64, while validation accuracy rises from 33.68% to 59.01%. After the best epoch, validation loss fluctuates upward and finishes 0.053575 higher at 1.302490. The scheduler progressively reduces the learning rate, but no further `val_loss` improvement passes the early-stopping threshold.

## 7. Best-Checkpoint Validation Classification Report

This report is the validation evaluation already recorded in the notebook for `best.pt`; it covers 383 validation samples.

| Class | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| cardboard | 0.611111 | 0.846154 | 0.709677 | 52 |
| electronic | 0.738095 | 0.756098 | 0.746988 | 41 |
| foodwaste | 0.750000 | 0.673469 | 0.709677 | 49 |
| glass | 0.593750 | 0.463415 | 0.520548 | 41 |
| metal | 0.380000 | 0.395833 | 0.387755 | 48 |
| paper | 0.518519 | 0.622222 | 0.565657 | 45 |
| plastic | 0.632653 | 0.484375 | 0.548673 | 64 |
| textile | 0.525000 | 0.488372 | 0.506024 | 43 |
| accuracy | 0.590078 | 0.590078 | 0.590078 | 383 |
| macro average | 0.593641 | 0.591242 | 0.586875 | 383 |
| weighted average | 0.594704 | 0.590078 | 0.586390 | 383 |

`electronic` has the highest F1-score (0.746988). `metal` is the weakest class (F1 0.387755), indicating the largest opportunity for representation improvement.

## 8. Best-Checkpoint Validation Confusion Matrix

Rows are actual labels and columns are predicted labels. This is transcribed from the confusion-matrix figure saved in the executed notebook output. Diagonal entries total 226 of 383 samples, consistent with the recorded 59.01% validation accuracy.

| Actual \ Predicted | cardboard | electronic | foodwaste | glass | metal | paper | plastic | textile |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cardboard | 44 | 0 | 1 | 1 | 2 | 3 | 1 | 0 |
| electronic | 0 | 31 | 0 | 3 | 1 | 1 | 3 | 2 |
| foodwaste | 5 | 3 | 33 | 0 | 3 | 1 | 0 | 4 |
| glass | 6 | 1 | 3 | 19 | 4 | 5 | 3 | 0 |
| metal | 5 | 2 | 1 | 5 | 19 | 4 | 5 | 7 |
| paper | 5 | 2 | 0 | 0 | 8 | 28 | 1 | 1 |
| plastic | 2 | 2 | 5 | 3 | 8 | 8 | 31 | 5 |
| textile | 5 | 1 | 1 | 1 | 5 | 4 | 5 | 21 |

The dominant errors are consistent with the per-class report: `metal` is confused with `textile` (7 samples) and several other material classes; `plastic` is confused with `metal` and `paper` (8 samples each); and `glass` is often predicted as `cardboard` (6) or `paper` (5).

## 9. Generalization Analysis

There is a mild-to-moderate overfitting signal after epoch 64, rather than a severe collapse or a clear underfitting-only pattern:

- At the best epoch, validation loss exceeds training loss by 0.212815 and training accuracy exceeds validation accuracy by 6.66 percentage points.
- At the last epoch, the validation/train loss gap grows to 0.280521 and validation loss is 0.053575 above the best value, while validation accuracy remains near 59%.
- Training accuracy remains modest (maximum 66.64%), and macro F1 peaks below 60%; the model is not simply memorizing the training set. Its limited feature capacity and the small dataset also leave room for under-capacity behavior on difficult classes.
- The correct operational choice is to retain `best.pt`, because it has the best configured validation objective and precedes the post-best validation-loss degradation.

## 10. Artifact Locations

| Artifact | Location |
| --- | --- |
| Best checkpoint | `models/checkpoints/cnn_scratch_baseline_v1/best.pt` |
| Last checkpoint | `models/checkpoints/cnn_scratch_baseline_v1/last.pt` |
| Per-epoch metric history | `outputs/metrics/cnn_scratch_baseline_v1.json` |
| Baseline comparison summary | `outputs/metrics/cnn_scratch_baseline_v1_summary.json` |
| Learning curves | `outputs/figures/cnn_scratch_baseline_v1_learning_curves.png` |
| Executed experiment and stored validation outputs | `notebooks/01_cnn_from_scratch.ipynb` |

## 11. Limitations of This Baseline

- The model learns all visual features from only 1,340 training images; it has no pretrained visual representation.
- The eight classes include visually similar materials, as reflected in the metal, plastic, paper, glass, and textile confusions.
- The baseline uses a fixed direct-resize input and simple CNN capacity; it may not preserve or represent fine material texture as effectively as a pretrained backbone.
- Results are validation-only for selection. The internal test split remains untouched and must not be used for iterative model choice.
- This is one seed/configuration run. Improvements should be compared under the same split and validation protocol before any final internal-test decision.

## 12. Recommended Next Pretrained Experiment

Run the next model as a separate experiment ID using a pretrained backbone such as MobileNetV3, EfficientNet, or ResNet50. Keep the V1 final manifest, class mapping, validation split, and no-internal-test-tuning rule unchanged. Record the same `val_loss`, validation accuracy, macro precision/recall/F1, per-class report, confusion matrix, and checkpoint monitor in the comparison notebook.

For a fair comparison, select each candidate checkpoint by the predeclared validation monitor, distinguish that checkpoint from its peak accuracy/F1 epoch when they differ, and compare per-class improvements with special attention to `metal`, `plastic`, `glass`, and `textile`.
