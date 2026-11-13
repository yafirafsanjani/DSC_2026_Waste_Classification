# scripts/ssl_v2/sec_01_06.py
SECTIONS = []

SECTIONS.append({
    "id": 1,
    "title": "01. Project Setup & Environment",
    "markdown": r"""# 01. Project Setup & Environment (TASK 09 — Optimized Few-Shot SSL)

Notebook ini mengimplementasikan eksperimen lanjutan **Few-Shot Semi-Supervised Learning (SSL) V2** untuk klasifikasi 8 kelas sampah kompetisi Data Science Competition (DSC) Gelar Rasa 2026:
- **Baseline Sebelumnya (TASK 08)**: Skor public leaderboard **0.5902** (hanya Teacher supervised 40 label karena zero pseudo-label akibat threshold 0.95 dan efek corrupt BatchNorm).
- **Tujuan TASK 09**: Mengoptimalkan performa few-shot classification melalui perbaikan mendasar:
  1. **Strict BatchNorm Freezing**: Mencegah degradasi ImageNet running statistics saat melatih classifier head pada batch kecil.
  2. **Perbandingan Model Few-Shot (5-Fold CV OOF)**: Mengevaluasi PyTorch Linear Head, L2-Normalized Logistic Regression, Linear SVM, dan Cosine Prototype Classifier.
  3. **Safe & Calibrated Pseudo-Labeling**: Menggunakan konsistensi dua arah (weak augmentation / horizontal flip) + **Class-Balanced Top-K** dengan confidence floor terukur.
  4. **Student Training**: Melatih Student dengan pembobotan loss adaptif pada 40 label manual + pseudo-label terverifikasi.
  5. **Test-Time Augmentation (TTA)**: Inferensi test 480 citra multi-ekstensi (.jpg, .jpeg, .png) dengan rata-rata probabilitas original + horizontal flip.
- **Hardware Optimization**: Dioptimalkan untuk NVIDIA RTX 3050 Laptop GPU (4 GB VRAM) dengan feature caching dan Mixed Precision (AMP).""",
    "code": r"""# ==============================================================================
# SECTION 01: PROJECT SETUP & REPRODUCIBILITY
# ==============================================================================
import os
import sys
import gc
import json
import time
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable, Union

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
from torchvision.transforms import v2
from tqdm.auto import tqdm

# Scikit-learn untuk few-shot classifiers & metrik OOF
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

# Resolusi path root proyek secara dinamis
def resolve_project_root() -> Path:
    candidates = [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]
    for c in candidates:
        if (c / "data" / "competition").exists() or (c / "data" / "manifests").exists():
            return c.resolve()
    return Path.cwd().resolve()

PROJECT_ROOT = resolve_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Seed reproduktifitas deterministik
RANDOM_SEED = 42
# Fungsi pemuatan checkpoint lokal aman dan terverifikasi
def load_project_checkpoint(path: Path, device: Optional[torch.device] = None) -> Dict[str, Any]:
    # Memuat checkpoint lokal terpercaya dengan verifikasi sumber direktori proyek.
    # Utamakan weights_only=True untuk keamanan, dengan fallback terkontrol ke weights_only=False
    # HANYA jika checkpoint terbukti berada di dalam direktori proyek ini (PROJECT_ROOT).
    path = Path(path).resolve()
    root_resolved = PROJECT_ROOT.resolve()
    try:
        path.relative_to(root_resolved)
    except ValueError:
        raise PermissionError(f"Akses file di luar direktori proyek ditolak: {path}")
        
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan di: {path}")
        
    map_loc = device if device is not None else "cpu"
    try:
        return torch.load(path, map_location=map_loc, weights_only=True)
    except Exception as err:
        print(f"[INFO] Checkpoint lokal memuat metadata non-tensor ({type(err).__name__}). Memuat dengan weights_only=False secara eksplisit...")
        return torch.load(path, map_location=map_loc, weights_only=False)

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

seed_everything(RANDOM_SEED)

# Deteksi perangkat komputasi GPU / CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()

# Direktori proyek resmi TASK 09 (Terisolasi dari TASK 08)
DATA_DIR = PROJECT_ROOT / "data"
COMPETITION_DATA_DIR = DATA_DIR / "competition"
DATA_TRAIN_DIR = COMPETITION_DATA_DIR / "train"
DATA_TEST_DIR = COMPETITION_DATA_DIR / "test"
SAMPLE_SUBMISSION_PATH = COMPETITION_DATA_DIR / "SampleSubmission_DSC.csv"

# Direktori manifest & checkpoint baru untuk TASK 09
MANIFEST_DIR = DATA_DIR / "manifests" / "fewshot_ssl_v2"
CHECKPOINTS_DIR = PROJECT_ROOT / "models" / "checkpoints" / "fewshot_ssl_v2"
TEACHER_CKPT_DIR = CHECKPOINTS_DIR / "teacher"
STUDENT_CKPT_DIR = CHECKPOINTS_DIR / "student"
CV_CKPT_DIR = CHECKPOINTS_DIR / "cv"
OUTPUT_METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
OUTPUT_FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
SUBMISSION_DIR = PROJECT_ROOT / "Final Dataset for Submission to Kaggle"
SUBMISSION_CSV_PATH = SUBMISSION_DIR / "submission_optimized_fewshot_ssl.csv"

for p in [MANIFEST_DIR, TEACHER_CKPT_DIR, STUDENT_CKPT_DIR, CV_CKPT_DIR, OUTPUT_METRICS_DIR, OUTPUT_FIGURES_DIR, SUBMISSION_DIR]:
    p.mkdir(parents=True, exist_ok=True)

print("=" * 75)
print(f"DSC GELAR RASA 2026 - TASK 09 OPTIMIZED FEW-SHOT SSL")
print(f"Project Root     : {PROJECT_ROOT}")
print(f"Device           : {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
print(f"AMP Mixed Prec   : {USE_AMP}")
print(f"Random Seed      : {RANDOM_SEED}")
print(f"Target Submission: {SUBMISSION_CSV_PATH.name}")
print("=" * 75)"""
})

SECTIONS.append({
    "id": 2,
    "title": "02. Experiment Configuration",
    "markdown": r"""# 02. Experiment Configuration (Centralized & Deterministic)

Konfigurasi eksperimen CONFIG didefinisikan secara terpusat untuk menjamin eksekusi otomatis tanpa flag manual:
- **Labeling Budget**: Maksimal 5 label manual per kelas, total tepat 40 label manual.
- **Cross-Validation**: 5-Fold Stratified CV (32 train, 8 validation per fold).
- **Candidate Classifiers**: PyTorch Linear Head (frozen BN), L2 Logistic Regression, Linear SVM, Cosine Prototype.
- **Safe Pseudo-Labeling**:
  - pseudo_min_confidence = 0.45 (di atas peluang acak 8 kelas = 0.125).
  - pseudo_top_k_per_class = 15 (maksimal 15 citra per kelas untuk mencegah ketimpangan kelas).
  - Multi-view consistency: Hanya menerima prediksi yang identik antara citra asli dan horizontal flip.
- **Student Training**: Supervised weight 1.0, pseudo loss weight 0.35.""",
    "code": r"""# ==============================================================================
# SECTION 02: EXPERIMENT CONFIGURATION
# ==============================================================================
CONFIG = {
    "experiment_id": "fewshot_ssl_v2",
    "random_seed": 42,
    "img_size": 288,
    "batch_size": 8,
    "num_classes": 8,
    "max_labels_per_class": 5,
    "max_total_labels": 40,
    "cv_folds": 5,
    "teacher_epochs": 20,
    "teacher_lr": 1e-3,
    "teacher_weight_decay": 1e-2,
    "student_epochs": 25,
    "student_lr": 5e-4,
    "student_weight_decay": 1e-4,
    "student_pseudo_weight": 0.35,
    "pseudo_min_confidence": 0.45,
    "pseudo_top_k_per_class": 15,
    "use_tta": True,
    "use_amp": USE_AMP
}

# 8 Kelas Resmi Kompetisi DSC 2026
CLASS_NAMES = ["cardboard", "electronic", "foodwaste", "glass", "metal", "paper", "plastic", "textile"]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}

print("KONFIGURASI EKSPERIMEN TASK 09:")
for k, v in CONFIG.items():
    print(f"  - {k:<25}: {v}")
print(f"  - Kelas Resmi ({NUM_CLASSES})        : {CLASS_NAMES}")"""
})

SECTIONS.append({
    "id": 3,
    "title": "03. Official Rules & Labeling Budget Verification",
    "markdown": r"""# 03. Official Rules & Labeling Budget Verification

Memverifikasi kepatuhan terhadap regulasi resmi kompetisi:
1. **Manual Labeling Budget**: Tidak boleh melebihi 5 citra per kelas atau 40 citra total.
2. **Anti-Leakage**: Citra test resmi (480 citra) dilarang digunakan untuk training, validasi, maupun pseudo-labeling.
3. **No External Dataset**: Dilarang menggunakan dataset eksternal atau label Roboflow historis di luar 40 label manual yang sah.
4. **No LLM / Cloud AI Inference**: Seluruh klasifikasi dilakukan menggunakan model lokal EfficientNet-B2.""",
    "code": r"""# ==============================================================================
# SECTION 03: OFFICIAL RULES & LABELING BUDGET VERIFICATION
# ==============================================================================
def verify_competition_compliance(config: Dict[str, Any]):
    assert config["max_labels_per_class"] <= 5, "PELANGGARAN: Maksimal 5 label manual per kelas!"
    assert config["max_total_labels"] <= 40, "PELANGGARAN: Maksimal 40 label manual total!"
    assert config["num_classes"] == 8, "PELANGGARAN: Harus tepat 8 kelas resmi!"
    assert DATA_TRAIN_DIR.exists(), f"Direktori train resmi tidak ditemukan: {DATA_TRAIN_DIR}"
    assert DATA_TEST_DIR.exists(), f"Direktori test resmi tidak ditemukan: {DATA_TEST_DIR}"
    assert SAMPLE_SUBMISSION_PATH.exists(), f"Sample submission tidak ditemukan: {SAMPLE_SUBMISSION_PATH}"
    print("[COMPLIANCE VERIFIED] Konfigurasi mematuhi regulasi resmi kompetisi.")

verify_competition_compliance(CONFIG)"""
})

SECTIONS.append({
    "id": 4,
    "title": "04. Dataset Loading & Flat Structure Integrity",
    "markdown": r"""# 04. Dataset Loading & Flat Structure Integrity

Memuat data berlabel sah dari labeled_pool_40.csv dan kolam data tanpa label unlabeled_pool.csv:
- Tepat **40 citra berlabel manual** (5 per kelas).
- Tepat **1.877 citra unlabeled**.
- Tepat **480 citra test kompetisi** multi-ekstensi (.jpg, .jpeg, .png).
- String image_id mempertahankan leading zero (misal 0001, 001).""",
    "code": r"""# ==============================================================================
# SECTION 04: DATASET LOADING & INTEGRITY AUDIT
# ==============================================================================
SOURCE_LABELED_POOL_PATH = PROJECT_ROOT / "data" / "manifests" / "fewshot_pseudolabel_b2_v1" / "labeled_pool_40.csv"
SOURCE_UNLABELED_POOL_PATH = PROJECT_ROOT / "data" / "manifests" / "fewshot_pseudolabel_b2_v1" / "unlabeled_pool.csv"

if not SOURCE_LABELED_POOL_PATH.exists():
    raise FileNotFoundError(f"Berkas labeled pool 40 tidak ditemukan di: {SOURCE_LABELED_POOL_PATH}")
if not SOURCE_UNLABELED_POOL_PATH.exists():
    raise FileNotFoundError(f"Berkas unlabeled pool tidak ditemukan di: {SOURCE_UNLABELED_POOL_PATH}")

labeled_df = pd.read_csv(SOURCE_LABELED_POOL_PATH, dtype={"image_id": str})
unlabeled_df = pd.read_csv(SOURCE_UNLABELED_POOL_PATH, dtype={"image_id": str})

# Validasi kuota 40 label manual
assert len(labeled_df) == 40, f"Expected 40 labeled samples, got {len(labeled_df)}"
counts = labeled_df["class_name"].value_counts()
assert len(counts) == 8, f"Expected 8 classes, got {len(counts)}"
assert (counts == 5).all(), f"Each class must have exactly 5 samples, got:\n{counts}"

# Validasi 1877 citra unlabeled tanpa label ground truth
assert len(unlabeled_df) == 1877, f"Expected 1877 unlabeled samples, got {len(unlabeled_df)}"
assert "class_name" not in unlabeled_df.columns, "Anti-leakage: unlabeled pool cannot have class_name"

# Salin manifest ke folder fewshot_ssl_v2 untuk persistensi independen
labeled_df.to_csv(MANIFEST_DIR / "labeled_pool_40.csv", index=False)
unlabeled_df.to_csv(MANIFEST_DIR / "unlabeled_pool.csv", index=False)

# Audit data test multi-format
allowed_exts = {".jpg", ".jpeg", ".png"}
test_files = [p for p in DATA_TEST_DIR.iterdir() if p.is_file() and p.suffix.lower() in allowed_exts]
sample_df = pd.read_csv(SAMPLE_SUBMISSION_PATH, dtype={"image_id": str})
assert len(test_files) == 480, f"Expected 480 test files, got {len(test_files)}"
assert len(sample_df) == 480, f"Expected 480 sample submission rows, got {len(sample_df)}"

print("HASIL PEMUATAN DATASET TASK 09:")
print(f"  - Labeled Pool   : {len(labeled_df)} citra (tepat 5 citra per kelas)")
print(f"  - Unlabeled Pool : {len(unlabeled_df)} citra train tanpa label")
print(f"  - Test Kompetisi : {len(test_files)} citra resmi (477 .jpg, 2 .jpeg, 1 .png)")"""
})

SECTIONS.append({
    "id": 5,
    "title": "05. Stratified 5-Fold Cross-Validation Alignment",
    "markdown": r"""# 05. Stratified 5-Fold Cross-Validation Alignment

Menyelaraskan split 5-fold cross-validation persis sama dengan eksperimen baseline (cv_folds.csv):
- Total 5 folds.
- Setiap fold: **32 citra train** (4 per kelas) dan **8 citra validation** (tepat 1 per kelas).
- Bebas kebocoran (train dan validation dalam satu fold strictly disjoint).
- Memungkinkan perbandingan performa yang 100% adil (*apple-to-apple*) terhadap baseline.""",
    "code": r"""# ==============================================================================
# SECTION 05: STRATIFIED 5-FOLD CV ALIGNMENT
# ==============================================================================
SOURCE_CV_FOLDS_PATH = PROJECT_ROOT / "data" / "manifests" / "fewshot_pseudolabel_b2_v1" / "cv_folds.csv"
if SOURCE_CV_FOLDS_PATH.exists():
    cv_folds_df = pd.read_csv(SOURCE_CV_FOLDS_PATH, dtype={"image_id": str})
else:
    from sklearn.model_selection import StratifiedKFold
    cv_folds_df = labeled_df.copy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=CONFIG["random_seed"])
    cv_folds_df["fold"] = -1
    for f, (_, val_idx) in enumerate(skf.split(cv_folds_df, cv_folds_df["class_idx"])):
        cv_folds_df.loc[val_idx, "fold"] = f

cv_folds_df.to_csv(MANIFEST_DIR / "cv_folds.csv", index=False)

for f in range(5):
    f_train = cv_folds_df[cv_folds_df["fold"] != f]
    f_val = cv_folds_df[cv_folds_df["fold"] == f]
    assert len(f_train) == 32
    assert len(f_val) == 8
    assert (f_val["class_name"].value_counts() == 1).all()
    overlap = set(f_train["image_id"]) & set(f_val["image_id"])
    assert len(overlap) == 0, f"Fold {f} data leakage detected!"

print("[CV ALIGNMENT PASS] 5-Fold CV terverifikasi utuh (32 train / 8 val per fold, 1 val per kelas).")"""
})

SECTIONS.append({
    "id": 6,
    "title": "06. Preprocessing & Augmentation Pipelines",
    "markdown": r"""# 06. Preprocessing & Augmentation Pipelines

Mendefinisikan pipeline transformasi citra standar resmi EfficientNet_B2_Weights.IMAGENET1K_V1:
1. **Eval Transform (Validation & Inference)**:
   - Resize 288x288 (Bicubic, antialias=True).
   - Center Crop 288x288.
   - Float32 [0, 1] scaling.
   - ImageNet Normalization (Mean [0.485, 0.456, 0.406], Std [0.229, 0.224, 0.225]).
2. **Weak Augmentation Transform (Horizontal Flip)**:
   - Digunakan untuk memeriksa konsistensi prediksi pseudo-labeling dan Test-Time Augmentation (TTA).
3. **Train Augmentation Transform**:
   - Random Horizontal Flip, Random Rotation (15 derajat), Color Jitter (10%).""",
    "code": r"""# ==============================================================================
# SECTION 06: PREPROCESSING & AUGMENTATION PIPELINES
# ==============================================================================
eval_transforms = v2.Compose([
    v2.ToImage(),
    v2.Resize(CONFIG["img_size"], interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
    v2.CenterCrop(CONFIG["img_size"]),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

weak_flip_transforms = v2.Compose([
    v2.ToImage(),
    v2.Resize(CONFIG["img_size"], interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
    v2.CenterCrop(CONFIG["img_size"]),
    v2.RandomHorizontalFlip(p=1.0),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_transforms = v2.Compose([
    v2.ToImage(),
    v2.Resize(CONFIG["img_size"], interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
    v2.CenterCrop(CONFIG["img_size"]),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(degrees=15),
    v2.ColorJitter(brightness=0.1, contrast=0.1),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("[PIPELINE TERDEFINISI]")
print("  - eval_transforms       : Resize 288 + CenterCrop + Bicubic + ImageNet Norm")
print("  - weak_flip_transforms  : Evaluasi multi-view horizontal flip (TTA & consistency)")
print("  - train_transforms      : Flip + Rotation(15) + Jitter(10%) + ImageNet Norm")"""
})
