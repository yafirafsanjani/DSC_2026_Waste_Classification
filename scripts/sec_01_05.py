# scripts/sec_01_05.py
import json
SECTIONS = []

SECTIONS.append({
    "id": 1,
    "title": "01. Project Overview",
    "markdown": r"""# 01. Project Overview ? Few-Shot EfficientNet-B2 & Pseudo-Labeling

**Kompetisi**: DSC 2026 Waste Classification  
**Task**: TASK 08 ? Unified Pipeline: Few-Shot Learning + Pseudo-Labeling + Kaggle Submission  
**Backbone Arsitektur**: `EfficientNet-B2` (pretrained `ImageNet-1K_V1`)  
**Strategi Semi-Supervised**: Teacher-Student Pseudo-Labeling  

### Latar Belakang & Batasan Regulasi
Dalam skenario kompetisi resmi ini, kita menghadapi tantangan klasifikasi citra dengan **anggaran pelabelan manual yang sangat terbatas (Few-Shot)**:
- **Sumber Data Train Resmi**: `data/competition/train/` berisi **1.920 citra sampah tanpa label**.
- **Sumber Data Test Resmi**: `data/competition/test/` berisi **480 citra test tanpa label**.
- **Maksimal 5 citra per kelas** atau total **maksimal 40 citra manual** untuk 8 kelas target.
- Sisa dataset training (~1.880 citra) diperlakukan sebagai **Unlabeled Pool** tanpa akses ke label.
- Data test resmi kompetisi (480 citra) hanya digunakan untuk inferensi akhir (submission), **tanpa kebocoran data (zero leakage)**.

### Alur Eksekusi Eksperimen (Unified Single Notebook):
1. **Bagian A ? Persiapan (01-11)**: Environment, audit regulasi, audit dataset train resmi (1.920) & test (480), partisi labeled pool (40) & unlabeled pool, audit anti-kebocoran, dan protokol 5-Fold Stratified Cross-Validation.
2. **Bagian B ? Preprocessing (12-17)**: Preprocessing resmi EfficientNet-B2 (288x288, Bicubic, ImageNet Normalization) dan augmentasi data terkendali.
3. **Bagian C ? Few-Shot Teacher (18-26)**: Pelatihan model Teacher pada 40 citra berlabel sah, evaluasi Out-Of-Fold CV, dan checkpointing.
4. **Bagian D ? Pseudo-Labeling (27-33)**: Inferensi unlabeled pool menggunakan Teacher, analisis confidence scores, penyaringan pseudo-label (threshold 0.95), dan analisis keseimbangan kelas.
5. **Bagian E ? Student Training (34-41)**: Pelatihan model Student menggunakan data gabungan (40 labeled + pseudo-labels) dengan pembobotan loss semi-supervised.
6. **Bagian F ? Evaluasi (42-47)**: Perbandingan performa Teacher vs Student, metrik validasi, laporan klasifikasi, matriks konfusi, dan analisis overfitting/underfitting.
7. **Bagian G ? Kaggle Submission (48-54)**: Audit data test kompetisi, setup inferensi mandiri (standalone), prediksi 480 citra test, pembuatan CSV submission, dan verifikasi integritas CSV.

*Seluruh proses dikendalikan melalui flag eksekusi manual (default False). Pengguna menjalankan eksekusi secara mandiri.*""",
    "code": r"""# ==============================================================================
# SECTION 01: PROJECT METADATA & ARCHITECTURE OVERVIEW
# ==============================================================================
PIPELINE_METADATA = {
    "project": "DSC_2026_Waste_Classification",
    "task": "TASK_08_FewShot_PseudoLabeling_EfficientNet_B2",
    "model_backbone": "EfficientNet-B2 (torchvision IMAGENET1K_V1)",
    "num_classes": 8,
    "classes": ["cardboard", "electronic", "foodwaste", "glass", "metal", "paper", "plastic", "textile"],
    "train_source_dir": "data/competition/train",
    "test_source_dir": "data/competition/test",
    "max_labels_per_class": 5,
    "max_total_labels": 40,
    "cv_folds": 5,
    "confidence_threshold_default": 0.95,
    "pseudo_loss_weight": 0.5,
    "input_resolution": (288, 288),
    "random_seed": 42
}

print("=" * 75)
print(f"PROYEK          : {PIPELINE_METADATA['project']}")
print(f"TASK            : {PIPELINE_METADATA['task']}")
print(f"BACKBONE        : {PIPELINE_METADATA['model_backbone']}")
print(f"DATA TRAIN      : {PIPELINE_METADATA['train_source_dir']} (1.920 citra resmi tanpa label)")
print(f"DATA TEST       : {PIPELINE_METADATA['test_source_dir']} (480 citra resmi tanpa label)")
print(f"RESOLUSI INPUT  : {PIPELINE_METADATA['input_resolution']}")
print(f"ANGGARAN LABEL  : Maksimal {PIPELINE_METADATA['max_labels_per_class']} citra/kelas ({PIPELINE_METADATA['max_total_labels']} citra total)")
print(f"KELAS TARGET    : {', '.join(PIPELINE_METADATA['classes'])}")
print(f"RANDOM SEED     : {PIPELINE_METADATA['random_seed']}")
print("=" * 75)"""
})

SECTIONS.append({
    "id": 2,
    "title": "02. Environment Setup",
    "markdown": r"""# 02. Environment Setup & Manual Execution Flags

Bagian ini mengimpor pustaka standar dan saintifik, menetapkan random seed deterministik (42), mendeteksi perangkat keras akselerasi GPU (CUDA) dengan fallback CPU, dan mendeklarasikan seluruh **Manual Execution Flags**.

### Aturan Eksekusi Manual
Seluruh flag berat bernilai `False` secara default. Pengguna dapat mengaktifkan flag secara bertahap saat ingin menjalankan tahapan tertentu di lingkungan lokal:
- `RULES_VERIFIED`: Konfirmasi manual atas buku aturan lomba.
- `RUN_FEWSHOT_PREPARATION`: Menyiapkan partisi labeled pool (40) dan unlabeled pool dari `data/competition/train/`.
- `RUN_CROSS_VALIDATION`: Menjalankan 5-Fold Cross-Validation teacher.
- `RUN_TEACHER_TRAINING`: Melatih Final Teacher pada 40 citra berlabel.
- `RUN_PSEUDO_LABEL_GENERATION`: Menghasilkan raw pseudo-label pada data unlabeled.
- `RUN_PSEUDO_LABEL_SELECTION`: Menyaring pseudo-label dengan threshold confidence.
- `RUN_STUDENT_TRAINING`: Melatih model Student pada data gabungan.
- `RUN_TEACHER_STUDENT_EVALUATION`: Menjalankan evaluasi komparatif Teacher vs Student.
- `RUN_TEST_INFERENCE`: Memprediksi 480 citra test kompetisi.
- `RUN_SUBMISSION_GENERATION`: Membuat file submission CSV resmi.""",
    "code": r"""# ==============================================================================
# SECTION 02: ENVIRONMENT SETUP, DEPENDENCIES & MANUAL FLAGS
# ==============================================================================
import os
import sys
import json
import time
import math
import random
import shutil
import hashlib
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Sequence, Set
from dataclasses import dataclass

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import sklearn.metrics as metrics
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
import torchvision.transforms.functional as TF

try:
    from torchvision.transforms import v2
    USE_TRANSFORMS_V2 = True
except ImportError:
    import torchvision.transforms as v2
    USE_TRANSFORMS_V2 = False

from tqdm.auto import tqdm

# --- REPRODUCIBILITY SEED SETUP ---
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

# --- HARDWARE ACCELERATION DETECTION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()

print("STATUS ENVIRONMENT:")
print(f"  Python Version      : {sys.version.split()[0]}")
print(f"  PyTorch Version     : {torch.__version__}")
print(f"  Torchvision Version : {torchvision.__version__}")
print(f"  Device              : {DEVICE}")
if torch.cuda.is_available():
    print(f"  GPU Device Name     : {torch.cuda.get_device_name(0)}")
    print(f"  GPU VRAM Total      : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
print(f"  Mixed Precision     : {'Enabled (AMP)' if USE_AMP else 'Disabled (FP32)'}")
print(f"  Transforms Engine   : {'Torchvision v2' if USE_TRANSFORMS_V2 else 'Torchvision v1'}")

# ==============================================================================
# MANUAL EXECUTION FLAGS (ALL DEFAULT FALSE)
# Ubah flag ke True secara manual sebelum menjalankan cell tahapan terkait
# ==============================================================================
RULES_VERIFIED = False
RUN_FEWSHOT_PREPARATION = False
RUN_CROSS_VALIDATION = False
RUN_TEACHER_TRAINING = False
RUN_PSEUDO_LABEL_GENERATION = False
RUN_PSEUDO_LABEL_SELECTION = False
RUN_STUDENT_TRAINING = False
RUN_TEACHER_STUDENT_EVALUATION = False
RUN_TEST_INFERENCE = False
RUN_SUBMISSION_GENERATION = False

print()
print("STATUS MANUAL EXECUTION FLAGS:")
print(f"  RULES_VERIFIED                 = {RULES_VERIFIED}")
print(f"  RUN_FEWSHOT_PREPARATION        = {RUN_FEWSHOT_PREPARATION}")
print(f"  RUN_CROSS_VALIDATION           = {RUN_CROSS_VALIDATION}")
print(f"  RUN_TEACHER_TRAINING           = {RUN_TEACHER_TRAINING}")
print(f"  RUN_PSEUDO_LABEL_GENERATION    = {RUN_PSEUDO_LABEL_GENERATION}")
print(f"  RUN_PSEUDO_LABEL_SELECTION     = {RUN_PSEUDO_LABEL_SELECTION}")
print(f"  RUN_STUDENT_TRAINING           = {RUN_STUDENT_TRAINING}")
print(f"  RUN_TEACHER_STUDENT_EVALUATION = {RUN_TEACHER_STUDENT_EVALUATION}")
print(f"  RUN_TEST_INFERENCE             = {RUN_TEST_INFERENCE}")
print(f"  RUN_SUBMISSION_GENERATION      = {RUN_SUBMISSION_GENERATION}")"""
})

SECTIONS.append({
    "id": 3,
    "title": "03. Official Rules Verification",
    "markdown": r"""# 03. Official Rules Verification & Approval Gate

Sebelum memulai pelatihan, integritas kepatuhan regulasi lomba wajib diaudit secara teliti:
1. **Identifikasi Dokumen Resmi**:
   Handbook resmi kompetisi telah tersedia di direktori `docs/` (`HANDBOOK PRELIMINARY ROUND DSC GELAR RASA 2026  (1).pdf`).
2. **Ringkasan Ketentuan Regulasi Resmi (Berdasarkan Halaman 7?8 Handbook)**:
   - **Batas Hand Labeling Data Train**: Peserta tidak disarankan melakukan hand labeling pada data train. Jika melakukan hand labeling, dibatasi **maksimal 5 gambar per kelas** (total maksimal 40 gambar untuk 8 kelas target dari total 1.920 data train).
   - **Integritas Data Test**: Peserta **dilarang keras** melakukan hand labeling terhadap data test (480 citra) dan **dilarang menggunakan data test** dalam proses pelatihan model.
   - **Pretrained Model**: Peserta **diperbolehkan** menggunakan pretrained model selama model tersebut dijalankan oleh peserta dalam lingkungan pengerjaan lokal/kompetisi.
   - **Larangan Dataset Eksternal**: Peserta **hanya diperbolehkan** menggunakan dataset yang disediakan panitia dan **dilarang menggunakan dataset eksternal** dalam pengerjaan proyek.
   - **Larangan AutoML, LLM, VLM, Generative AI**: Peserta **dilarang keras** menggunakan AutoML/LLM, VLM, generative AI, atau layanan AI pihak ketiga untuk membuat model, melakukan klasifikasi, atau menghasilkan prediksi terhadap data kompetisi. Library/framework pendukung yang tidak melakukan inference terhadap data kompetisi tetap diperbolehkan.
   - **Status Pseudo-Labeling**: Teknik *pseudo-labeling* atau *semi-supervised learning* **tidak disebutkan secara eksplisit** dalam handbook. Namun, karena model dilatih secara mandiri pada data kompetisi internal tanpa dataset eksternal dan tetap menghormati kuota 5 hand label per kelas, teknik ini digunakan sebagai strategi teknis internal.
   - **Kewajiban Satu Notebook Reproducible**: Peserta diwajibkan mengumpulkan **hanya satu file notebook (.ipynb)** yang mencakup seluruh tahapan, bersifat *reproducible* sehingga dapat dijalankan ulang oleh panitia dan menghasilkan skor yang sama dengan file submission CSV.
3. **Approval Gate Independen**:
   Meskipun dokumen handbook resmi telah berhasil dideteksi dan diaudit, flag `RULES_VERIFIED` **tetap bernilai `False` secara default**. Pengguna sendiri yang harus menyetujui dan mengubah flag menjadi `True` secara manual di Section 02 setelah memvalidasi interpretasi aturan tersebut.""",
    "code": r"""# ==============================================================================
# SECTION 03: OFFICIAL RULES VERIFICATION & APPROVAL GATE
# ==============================================================================
def resolve_project_root() -> Path:
    candidates = [
        Path.cwd(),
        Path.cwd().parent,
        Path.cwd().parent.parent
    ]
    for candidate in candidates:
        if (candidate / "data" / "competition").exists():
            return candidate.resolve()
    return Path.cwd().resolve()

if "PROJECT_ROOT" not in globals():
    PROJECT_ROOT = resolve_project_root()

def detect_competition_handbook(docs_dir: Path) -> List[Dict[str, Any]]:
    # Mencari berkas PDF handbook/regulasi secara case-insensitive pada direktori docs
    matched_files = []
    if not docs_dir.exists():
        return matched_files
        
    keywords = ["handbook", "preliminary", "dsc", "gelar", "rasa", "rule", "regulasi", "panduan", "guide", "ketentuan", "pedoman"]
    
    for item in sorted(docs_dir.iterdir(), key=lambda p: p.name.lower()):
        if item.is_file() and item.suffix.lower() == ".pdf":
            name_lower = item.name.lower()
            if any(kw in name_lower for kw in keywords):
                matched_files.append({
                    "name": item.name,
                    "path": item.resolve(),
                    "relative_path": item.relative_to(PROJECT_ROOT).as_posix(),
                    "size_bytes": item.stat().st_size
                })
    return matched_files

DOCS_DIR = PROJECT_ROOT / "docs"
detected_handbooks = detect_competition_handbook(DOCS_DIR)

print("=" * 75)
print("HASIL AUDIT DETEKSI DOKUMEN REGULASI RESMI:")
print(f"Direktori Pencarian   : {DOCS_DIR.resolve()}")
print(f"Dokumen Ditemukan     : {len(detected_handbooks) > 0}")

if detected_handbooks:
    for idx, doc_info in enumerate(detected_handbooks, 1):
        print(f"
[DOKUMEN {idx}]")
        print(f"  Nama Berkas   : {doc_info['name']}")
        print(f"  Path Lengkap  : {doc_info['path']}")
        print(f"  Ukuran Berkas : {doc_info['size_bytes'] / (1024*1024):.2f} MB")
else:
    print("  [PERINGATAN] Berkas PDF regulasi/handbook tidak ditemukan di docs/.")

# Ringkasan Eksekutif Aturan Resmi Berdasarkan Dokumen (Handbook Halaman 7-8)
OFFICIAL_RULES_SUMMARY = {
    "hand_labeling_train_limit": "Maksimal 5 gambar per kelas (total 40 gambar manual dari 1920 data train).",
    "test_data_restriction": "Dilarang hand labeling pada data test dan dilarang menggunakan data test untuk training.",
    "pretrained_model_allowed": "Pretrained model diperbolehkan selama dijalankan dalam lingkungan lokal peserta.",
    "external_dataset_prohibited": "Dilarang menggunakan dataset eksternal; hanya boleh memakai dataset panitia.",
    "automl_llm_genai_prohibited": "Dilarang memakai AutoML/LLM, VLM, Generative AI, atau API pihak ketiga untuk model/prediksi.",
    "pseudo_labeling_status": "Tidak dijelaskan secara eksplisit di handbook (digunakan sebagai strategi teknis mandiri).",
    "single_notebook_reproducible": "Wajib 1 file notebook (.ipynb) yang reproducible dan menghasilkan skor sama dengan CSV submission.",
    "max_submissions_per_day": 3,
    "submission_deadline": "14 November 2026 pukul 23.59 WIB"
}

print("
" + "=" * 75)
print("RINGKASAN ATURAN KOMPETISI RESMI (HANDBOOK HALAMAN 7-8):")
print(f"1. Kuota Hand Labeling Data Train : {OFFICIAL_RULES_SUMMARY['hand_labeling_train_limit']}")
print(f"2. Larangan Data Test             : {OFFICIAL_RULES_SUMMARY['test_data_restriction']}")
print(f"3. Izin Pretrained Model          : {OFFICIAL_RULES_SUMMARY['pretrained_model_allowed']}")
print(f"4. Larangan Dataset Eksternal     : {OFFICIAL_RULES_SUMMARY['external_dataset_prohibited']}")
print(f"5. Larangan AutoML/LLM/GenAI      : {OFFICIAL_RULES_SUMMARY['automl_llm_genai_prohibited']}")
print(f"6. Status Pseudo-Labeling         : {OFFICIAL_RULES_SUMMARY['pseudo_labeling_status']}")
print(f"7. Kewajiban Single Notebook      : {OFFICIAL_RULES_SUMMARY['single_notebook_reproducible']}")
print(f"8. Batas Pengumpulan Submission   : {OFFICIAL_RULES_SUMMARY['submission_deadline']}")
print("=" * 75)

# Status Manual Approval Gate (Tidak diubah otomatis menjadi True)
print(f"
Status Approval Gate: RULES_VERIFIED = {RULES_VERIFIED}")
if not RULES_VERIFIED:
    print("[APPROVAL GATE: PENDING] Dokumen regulasi telah dideteksi dan diringkas,")
    print("namun status verifikasi tetap 'False' sesuai prinsip manual execution control.")
    print("Silakan tinjau ringkasan di atas, lalu ubah flag RULES_VERIFIED = True")
    print("pada Section 02 secara manual untuk menyetujui kepatuhan aturan.")
else:
    print("[APPROVAL GATE: VERIFIED] Status kepatuhan telah disetujui pengguna.")"""
})

SECTIONS.append({
    "id": 4,
    "title": "04. Dataset Path Configuration",
    "markdown": r"""# 04. Dataset Path Configuration (Official Competition Data)

Notebook ini dikonfigurasi untuk membaca **Dataset Resmi Asli dari Panitia**:
- **Train Resmi**: `data/competition/train/` (1.920 citra tanpa label).
- **Test Resmi**: `data/competition/test/` (480 citra tanpa label).
- **Template Submission**: `data/competition/SampleSubmission_DSC.csv`.

Semua path bersifat relatif dan portabel tanpa hardcoding path drive Windows.""",
    "code": r"""# ==============================================================================
# SECTION 04: DATASET PATH CONFIGURATION
# ==============================================================================
def resolve_project_root() -> Path:
    candidates = [
        Path.cwd(),
        Path.cwd().parent,
        Path.cwd().parent.parent
    ]
    for candidate in candidates:
        if (candidate / "data" / "competition").exists():
            return candidate.resolve()
    return Path.cwd().resolve()

PROJECT_ROOT = resolve_project_root()

# Direktori Sumber Data Resmi Kompetisi
DATA_TRAIN_DIR = PROJECT_ROOT / "data" / "competition" / "train"
DATA_TEST_DIR = PROJECT_ROOT / "data" / "competition" / "test"
SAMPLE_SUBMISSION_PATH = PROJECT_ROOT / "data" / "competition" / "SampleSubmission_DSC.csv"

# Direktori Manifest & Artefak TASK 08
MANIFEST_DIR = PROJECT_ROOT / "data" / "manifests" / "fewshot_pseudolabel_b2_v1"
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints" / "fewshot_pseudolabel_b2_v1"
CV_CHECKPOINT_DIR = CHECKPOINT_DIR / "cv"
TEACHER_CHECKPOINT_DIR = CHECKPOINT_DIR / "teacher"
STUDENT_CHECKPOINT_DIR = CHECKPOINT_DIR / "student"

OUTPUT_METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
OUTPUT_LOGS_DIR = PROJECT_ROOT / "outputs" / "logs"
OUTPUT_FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
SUBMISSION_DIR = PROJECT_ROOT / "Final Dataset for Submission to Kaggle"
SUBMISSION_CSV_PATH = SUBMISSION_DIR / "submission_fewshot_pseudolabel_b2.csv"

# Buat seluruh direktori artefak jika belum ada
for folder in [
    MANIFEST_DIR, CV_CHECKPOINT_DIR, TEACHER_CHECKPOINT_DIR, STUDENT_CHECKPOINT_DIR,
    OUTPUT_METRICS_DIR, OUTPUT_LOGS_DIR, OUTPUT_FIGURES_DIR, SUBMISSION_DIR
]:
    folder.mkdir(parents=True, exist_ok=True)

print(f"PROJECT_ROOT           : {PROJECT_ROOT}")
print(f"DATA_TRAIN_DIR         : {DATA_TRAIN_DIR} (Exists: {DATA_TRAIN_DIR.exists()})")
print(f"DATA_TEST_DIR          : {DATA_TEST_DIR} (Exists: {DATA_TEST_DIR.exists()})")
print(f"SAMPLE_SUBMISSION_PATH : {SAMPLE_SUBMISSION_PATH} (Exists: {SAMPLE_SUBMISSION_PATH.exists()})")
print(f"MANIFEST_DIR           : {MANIFEST_DIR}")
print(f"CHECKPOINT_DIR         : {CHECKPOINT_DIR}")
print(f"SUBMISSION_CSV_PATH    : {SUBMISSION_CSV_PATH}")"""
})

SECTIONS.append({
    "id": 5,
    "title": "05. Dataset Integrity Audit",
    "markdown": r"""# 05. Dataset Integrity Audit (Official Train & Test)

Audit fisik terhadap seluruh berkas citra pada dataset resmi kompetisi:
1. Memverifikasi seluruh 1.920 citra train dapat dibaca tanpa kerusakan.
2. Memverifikasi seluruh 480 citra test dapat dibaca tanpa kerusakan.
3. Memeriksa mode warna (RGB) dan sebaran format ekstensi (`.jpg`, `.jpeg`, `.png`).
4. Memastikan struktur `data/competition/train` adalah direktori datar (flat) tanpa subfolder kelas.""",
    "code": r"""# ==============================================================================
# SECTION 05: DATASET INTEGRITY AUDIT
# ==============================================================================
def audit_image_files(directory: Path, allowed_exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png")) -> Dict[str, Any]:
    if not directory.exists():
        return {"exists": False, "total_files": 0, "readable": 0, "corrupted": 0, "issues": []}
    
    image_paths = [p for p in directory.glob("*") if p.is_file() and p.suffix.lower() in allowed_exts]
    subdirs = [p for p in directory.iterdir() if p.is_dir()]
    readable_count = 0
    corrupted_count = 0
    issues = []
    
    for path in image_paths:
        try:
            with Image.open(path) as img:
                img.verify()
            readable_count += 1
        except Exception as err:
            corrupted_count += 1
            issues.append((str(path), str(err)))
            
    return {
        "exists": True,
        "directory": str(directory),
        "subdirectories_count": len(subdirs),
        "total_files": len(image_paths),
        "readable": readable_count,
        "corrupted": corrupted_count,
        "issues": issues
    }

print("Menjalankan audit integritas berkas citra resmi kompetisi...")
train_audit = audit_image_files(DATA_TRAIN_DIR)
test_audit = audit_image_files(DATA_TEST_DIR)

print(f"COMPETITION TRAIN DIR : {train_audit['total_files']} files | {train_audit['readable']} readable | {train_audit['corrupted']} corrupted | Subdirs: {train_audit['subdirectories_count']}")
print(f"COMPETITION TEST DIR  : {test_audit['total_files']} files | {test_audit['readable']} readable | {test_audit['corrupted']} corrupted | Subdirs: {test_audit['subdirectories_count']}")

assert train_audit['total_files'] == 1920, f"Ekspektasi 1920 citra train, ditemukan {train_audit['total_files']}"
assert test_audit['total_files'] == 480, f"Ekspektasi 480 citra test, ditemukan {test_audit['total_files']}"
assert train_audit['corrupted'] == 0 and test_audit['corrupted'] == 0, "Ditemukan berkas citra korup!"
print("[INTEGRITAS VALID] Seluruh 1.920 citra train dan 480 citra test resmi utuh dan dapat dibaca.")"""
})
