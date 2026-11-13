# scripts/sec_48_54.py
SECTIONS = []

SECTIONS.append({
    "id": 48,
    "title": "48. Competition Test Audit",
    "markdown": r"""# 48. Competition Test Audit (480 Citra)

Audit menyeluruh terhadap direktori data test kompetisi (data/competition/test/) dan template SampleSubmission_DSC.csv:
- Tepat **480 citra test** resmi berekstensi multi-format: **477 berkas .jpg**, **2 berkas .jpeg** (142.jpeg, 166.jpeg), dan **1 berkas .png** (039.png).
- Pola penamaan ID: 001 sampai 480 (mempertahankan leading zero).
- Sample submission memiliki tepat 480 baris data dengan header image_id,waste_category.
- **Anti-Leakage**: Nilai kategori pada sample submission adalah dummy dan dilarang digunakan sebagai target.
- Verifikasi integritas: Seluruh 480 citra dapat dibuka dan didekode oleh Pillow tanpa galat.""",
    "code": r"""# ==============================================================================
# SECTION 48: COMPETITION TEST AUDIT
# ==============================================================================
def audit_competition_test_data(test_dir: Path, sample_sub_path: Path):
    if not test_dir.exists():
        raise FileNotFoundError(f"Direktori test tidak ditemukan: {test_dir}")
    if not sample_sub_path.exists():
        raise FileNotFoundError(f"Sample submission tidak ditemukan: {sample_sub_path}")
        
    allowed_exts = {".jpg", ".jpeg", ".png"}
    test_files = [p for p in test_dir.iterdir() if p.is_file() and p.suffix.lower() in allowed_exts]
    test_files = sorted(test_files, key=lambda p: p.name)
    sample_df = pd.read_csv(sample_sub_path, dtype={"image_id": str})
    
    # Audit sebaran ekstensi resmi (.jpg, .jpeg, .png)
    ext_counts = {}
    for p in test_files:
        ext = p.suffix.lower()
        ext_counts[ext] = ext_counts.get(ext, 0) + 1
        
    test_stem_map = {p.stem: p for p in test_files}
    expected_ids = sample_df["image_id"].astype(str).tolist()
    missing_ids = [img_id for img_id in expected_ids if img_id not in test_stem_map]
    extra_ids = [stem for stem in test_stem_map if stem not in set(expected_ids)]
    
    # Verifikasi seluruh citra dapat dibuka dan didekode oleh Pillow
    corrupted_images = []
    for p in test_files:
        try:
            with Image.open(p) as img:
                img.verify()
        except Exception as err:
            corrupted_images.append((p.name, str(err)))
            
    print("HASIL AUDIT DATA TEST KOMPETISI:")
    print(f"  Jumlah Citra Test   : {len(test_files)} (Diharapkan tepat 480)")
    print(f"  Sebaran Ekstensi    : {ext_counts}")
    print(f"  Jumlah Baris Sample : {len(sample_df)} (Diharapkan tepat 480)")
    print(f"  Kolom Sample        : {list(sample_df.columns)}")
    print(f"  Contoh ID Citra     : {[p.stem for p in test_files[:5]]}")
    print(f"  ID Tidak Ditemukan  : {len(missing_ids)}")
    print(f"  ID Berlebih         : {len(extra_ids)}")
    print(f"  Citra Rusak         : {len(corrupted_images)}")
    
    assert len(test_files) == 480, f"Jumlah citra test ({len(test_files)}) tidak sesuai dengan 480!"
    assert len(sample_df) == 480, f"Jumlah baris sample submission ({len(sample_df)}) tidak sesuai dengan 480!"
    assert len(missing_ids) == 0, f"Ditemukan ID sample submission yang tidak ada di test: {missing_ids}"
    assert len(extra_ids) == 0, f"Ditemukan berkas test di luar sample submission: {extra_ids}"
    assert len(corrupted_images) == 0, f"Ditemukan citra test yang rusak: {corrupted_images}"
    print("[TEST AUDIT PASS] Data test kompetisi terverifikasi utuh, lengkap (480 citra multi-ekstensi), dan valid.")

audit_competition_test_data(DATA_TEST_DIR, SAMPLE_SUBMISSION_PATH)"""
})

SECTIONS.append({
    "id": 49,
    "title": "49. Standalone Inference Configuration",
    "markdown": r"""# 49. Standalone Inference Configuration (Fresh-Kernel Compatible)

**KRUSIAL**: Bagian ini didesain agar dapat dijalankan langsung dari **kernel Jupyter yang baru/bersih** tanpa perlu menjalankan cell training sebelumnya:
1. Mengimpor dependensi secara mandiri.
2. Mendeteksi PROJECT_ROOT secara dinamis.
3. Mendefinisikan arsitektur EfficientNet-B2 (weights=None) dan memuat state_dict langsung dari checkpoint disk.
4. Menyiapkan transform inferensi standar (288x288, ImageNet normalization).
5. Membaca template SampleSubmission_DSC.csv untuk menjaga urutan dan format ID resmi.""",
    "code": r"""# ==============================================================================
# SECTION 49: STANDALONE INFERENCE CONFIGURATION
# ==============================================================================
# Standalone imports
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import efficientnet_b2
from torchvision.transforms import v2
from tqdm.auto import tqdm

def standalone_setup():
    candidates = [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]
    root = Path.cwd()
    for c in candidates:
        if (c / "data" / "competition").exists() or (c / "data" / "extracted").exists():
            root = c.resolve()
            break
            
    classes = ["cardboard", "electronic", "foodwaste", "glass", "metal", "paper", "plastic", "textile"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = v2.Compose([
        v2.ToImage(),
        v2.Resize(288, interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
        v2.CenterCrop(288),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return root, classes, dev, transform

SA_ROOT, SA_CLASSES, SA_DEVICE, SA_TRANSFORM = standalone_setup()

# Fallback assignments jika dijalankan langsung dari kernel bersih
if "PROJECT_ROOT" not in globals():
    PROJECT_ROOT = SA_ROOT
if "DEVICE" not in globals():
    DEVICE = SA_DEVICE
if "CLASS_NAMES" not in globals():
    CLASS_NAMES = list(SA_CLASSES)
if "NUM_CLASSES" not in globals():
    NUM_CLASSES = len(CLASS_NAMES)
if "IDX_TO_CLASS" not in globals():
    IDX_TO_CLASS = {i: c for i, c in enumerate(CLASS_NAMES)}
if "CLASS_TO_IDX" not in globals():
    CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
if "DATA_TEST_DIR" not in globals():
    DATA_TEST_DIR = PROJECT_ROOT / "data" / "competition" / "test"
if "SAMPLE_SUBMISSION_PATH" not in globals():
    SAMPLE_SUBMISSION_PATH = PROJECT_ROOT / "data" / "competition" / "SampleSubmission_DSC.csv"
if "SUBMISSION_CSV_PATH" not in globals():
    SUBMISSION_CSV_PATH = PROJECT_ROOT / "Final Dataset for Submission to Kaggle" / "submission_fewshot_pseudolabel_b2.csv"
if "eval_transforms" not in globals():
    eval_transforms = SA_TRANSFORM

def build_standalone_inference_model(num_classes: int = 8) -> nn.Module:
    m = efficientnet_b2(weights=None)
    in_features = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes)
    )
    return m

print(f"[STANDALONE READY] Root: {SA_ROOT} | Device: {SA_DEVICE} | Classes: {len(SA_CLASSES)}")"""
})

SECTIONS.append({
    "id": 50,
    "title": "50. Best Model Selection",
    "markdown": r"""# 50. Best Model Selection for Test Inference

Logika pemilihan checkpoint terbaik untuk inferensi test kompetisi:
- **Prioritas 1**: Model **Student** (student/best.pt) jika telah dilatih menggunakan semi-supervised learning.
- **Prioritas 2**: Model **Teacher** (	eacher/best.pt) jika Student belum dilatih.
- Jika keduanya belum tersedia, fungsi akan memberikan pesan informatif bahwa training perlu dijalankan terlebih dahulu.""",
    "code": r"""# ==============================================================================
# SECTION 50: BEST MODEL SELECTION FOR TEST INFERENCE
# ==============================================================================
def select_best_inference_checkpoint(project_root: Path) -> Tuple[Path, str]:
    student_ckpt = project_root / "models" / "checkpoints" / "fewshot_pseudolabel_b2_v1" / "student" / "best.pt"
    teacher_ckpt = project_root / "models" / "checkpoints" / "fewshot_pseudolabel_b2_v1" / "teacher" / "best.pt"
    
    if student_ckpt.exists():
        return student_ckpt, "Student (Few-Shot + Pseudo-Labeling)"
    elif teacher_ckpt.exists():
        return teacher_ckpt, "Teacher (Few-Shot Supervised Only)"
    else:
        raise FileNotFoundError(
            f"Tidak ditemukan checkpoint model di {student_ckpt.parent} atau {teacher_ckpt.parent}. "
            "Jalankan Section 24 (Teacher) atau Section 38 (Student) terlebih dahulu!"
        )

try:
    SELECTED_CKPT_PATH, SELECTED_MODEL_TYPE = select_best_inference_checkpoint(PROJECT_ROOT)
    print(f"CHECKPOINT TERPILIH : {SELECTED_CKPT_PATH}")
    print(f"TIPE MODEL          : {SELECTED_MODEL_TYPE}")
except FileNotFoundError as err:
    print(f"[WAITING] {err}")
    SELECTED_CKPT_PATH, SELECTED_MODEL_TYPE = None, None"""
})

SECTIONS.append({
    "id": 51,
    "title": "51. Competition Test Inference",
    "markdown": r"""# 51. Competition Test Inference (480 Test Samples)

Menjalankan inferensi pada 480 citra test kompetisi asli tanpa label (data/competition/test/):
- Memuat model terbaik yang terpilih di Section 50 (Student jika tersedia, atau Teacher).
- Menggunakan 	orch.inference_mode() dan preprocessing resmi (288x288, ImageNet normalisasi).
- Mendukung berkas dengan ekstensi .jpg, .jpeg, dan .png secara dinamis case-insensitive.
- Menghasilkan prediksi kelas resmi untuk seluruh 480 citra.""",
    "code": r"""# ==============================================================================
# SECTION 51: COMPETITION TEST INFERENCE
# ==============================================================================
class StandaloneCompetitionTestDataset(Dataset):
    def __init__(self, image_paths: List[Path], transform: Optional[Callable] = None):
        self.image_paths = list(image_paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        path = self.image_paths[idx]
        with Image.open(path) as img:
            image = img.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, path.stem

def predict_competition_test(
    ckpt_path: Path,
    test_dir: Path,
    sample_sub_path: Path,
    device: torch.device
) -> Dict[str, str]:
    sample_df = pd.read_csv(sample_sub_path, dtype={"image_id": str})
    expected_ids = sample_df["image_id"].astype(str).tolist()
    
    # Inisialisasi model inferensi
    if "build_teacher_model" in globals():
        model = build_teacher_model(num_classes=NUM_CLASSES, pretrained=False, freeze_backbone=True)
    else:
        model = build_standalone_inference_model(num_classes=NUM_CLASSES)
        
    payload = torch.load(ckpt_path, map_location=device)
    state_dict = payload["model_state_dict"] if "model_state_dict" in payload else payload
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Resolusi ekstensi file (.jpg, .jpeg, .png) secara dinamis case-insensitive
    allowed_exts = {".jpg", ".jpeg", ".png"}
    test_file_map = {
        p.stem: p for p in test_dir.iterdir()
        if p.is_file() and p.suffix.lower() in allowed_exts
    }
    
    missing = [img_id for img_id in expected_ids if str(img_id) not in test_file_map]
    if missing:
        raise FileNotFoundError(f"Citra test berikut tidak ditemukan di {test_dir}: {missing}")
        
    # Urutkan berkas citra persis sesuai urutan sample submission template
    test_files = [test_file_map[str(img_id)] for img_id in expected_ids]
    
    test_ds = StandaloneCompetitionTestDataset(test_files, transform=eval_transforms)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)
    
    predictions = {}
    print(f"Menjalankan inferensi pada {len(test_ds)} citra test...")
    with torch.inference_mode():
        for images, img_ids in tqdm(test_loader, desc="Test Inference"):
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            for img_id, pred_idx in zip(img_ids, preds):
                predictions[str(img_id)] = IDX_TO_CLASS[int(pred_idx)]
                
    return predictions

if SELECTED_CKPT_PATH is not None and SELECTED_CKPT_PATH.exists():
    print(f"[RUN-ALL] Menjalankan inferensi test kompetisi menggunakan {SELECTED_MODEL_TYPE}...")
    test_predictions = predict_competition_test(SELECTED_CKPT_PATH, DATA_TEST_DIR, SAMPLE_SUBMISSION_PATH, DEVICE)
    print(f"[COMPLETED] Inferensi selesai. {len(test_predictions)} prediksi berhasil dibuat.")
else:
    print("[STATUS] Checkpoint model untuk inferensi belum tersedia.")
    test_predictions = {}"""
})

SECTIONS.append({
    "id": 52,
    "title": "52. Submission CSV Generation",
    "markdown": r"""# 52. Submission CSV Generation

Menyimpan hasil prediksi 480 citra test ke berkas CSV submission resmi:
- Lokasi: Final Dataset for Submission to Kaggle/submission_fewshot_pseudolabel_b2.csv
- Format: image_id,waste_category dengan urutan persis sesuai SampleSubmission_DSC.csv.
- Leading zero pada image_id dipertahankan secara utuh.""",
    "code": r"""# ==============================================================================
# SECTION 52: SUBMISSION CSV GENERATION
# ==============================================================================
def generate_submission_csv(
    predictions: Dict[str, str],
    sample_sub_path: Path,
    output_path: Path
) -> pd.DataFrame:
    sample_df = pd.read_csv(sample_sub_path, dtype={"image_id": str})
    
    records = []
    for img_id in sample_df["image_id"]:
        img_id_str = str(img_id)
        if img_id_str not in predictions:
            raise KeyError(f"Prediksi untuk image_id '{img_id_str}' tidak ditemukan!")
        records.append({
            "image_id": img_id_str,
            "waste_category": str(predictions[img_id_str])
        })
        
    sub_df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sub_df.to_csv(output_path, index=False)
    return sub_df

if len(test_predictions) == 480:
    print("[RUN-ALL] Menghasilkan berkas CSV submission resmi Kaggle...")
    submission_df = generate_submission_csv(test_predictions, SAMPLE_SUBMISSION_PATH, SUBMISSION_CSV_PATH)
    print(f"[SUCCESS] Submission CSV disimpan ke: {SUBMISSION_CSV_PATH}")
elif SUBMISSION_CSV_PATH.exists():
    submission_df = pd.read_csv(SUBMISSION_CSV_PATH, dtype={"image_id": str})
    print(f"[LOADED] Berkas submission CSV telah tersedia di: {SUBMISSION_CSV_PATH}")
else:
    print("[STATUS] Prediksi test belum lengkap (480 citra).")
    submission_df = pd.DataFrame()"""
})

SECTIONS.append({
    "id": 53,
    "title": "53. Submission CSV Verification",
    "markdown": r"""# 53. Submission CSV Verification

Verifikasi komprehensif terhadap integritas berkas submission CSV sebelum diunggah ke Kaggle:
- Tepat 480 baris.
- Dua kolom resmi: image_id,waste_category.
- Urutan dan format ID identik dengan template sample submission.
- Seluruh kategori valid dalam 8 kelas resmi.""",
    "code": r"""# ==============================================================================
# SECTION 53: SUBMISSION CSV VERIFICATION
# ==============================================================================
def verify_submission_csv(csv_path: Path, sample_sub_path: Path, valid_classes: Optional[List[str]] = None) -> bool:
    if not csv_path.exists():
        print(f"[PERINGATAN] Berkas submission {csv_path.name} belum ada.")
        return False
        
    if valid_classes is None:
        valid_classes = CLASS_NAMES if "CLASS_NAMES" in globals() else ["cardboard", "electronic", "foodwaste", "glass", "metal", "paper", "plastic", "textile"]
        
    sample_df = pd.read_csv(sample_sub_path, dtype={"image_id": str})
    sub_df = pd.read_csv(csv_path, dtype={"image_id": str})
    
    assert len(sub_df) == 480, f"Jumlah baris ({len(sub_df)}) tidak sama dengan 480!"
    assert list(sub_df.columns) == ["image_id", "waste_category"], f"Kolom tidak valid: {list(sub_df.columns)}"
    assert (sub_df["image_id"] == sample_df["image_id"]).all(), "Urutan image_id tidak cocok dengan template!"
    assert not sub_df["image_id"].duplicated().any(), "Ditemukan duplikasi image_id!"
    assert sub_df["waste_category"].notna().all(), "Ditemukan nilai waste_category yang kosong!"
    
    invalid_cats = set(sub_df["waste_category"]) - set(valid_classes)
    assert not invalid_cats, f"Ditemukan kategori di luar 8 kelas resmi: {invalid_cats}"
    
    print("=" * 65)
    print(f"[VERIFIKASI LOLOS] Berkas: {csv_path.name}")
    print(f"Total Baris         : {len(sub_df)} (Tepat 480)")
    print("Integritas Format   : 100% Valid (2 kolom resmi, urutan identik, tanpa duplikasi)")
    print("=" * 65)
    print("\nPreview 5 Baris Pertama Submission:")
    print(sub_df.head())
    print("\nDistribusi Prediksi Tiap Kelas:")
    print(sub_df["waste_category"].value_counts())
    return True

if SUBMISSION_CSV_PATH.exists():
    verify_submission_csv(SUBMISSION_CSV_PATH, SAMPLE_SUBMISSION_PATH, CLASS_NAMES if "CLASS_NAMES" in globals() else None)
else:
    print(f"[STATUS] Berkas submission {SUBMISSION_CSV_PATH.name} belum tersedia untuk diverifikasi.")"""
})

SECTIONS.append({
    "id": 54,
    "title": "54. Final Submission Summary",
    "markdown": r"""# 54. Final Submission Summary & Kaggle Upload Guide

Rangkuman akhir dan panduan pengunggahan hasil ke platform kompetisi Kaggle:
- File Submission: Final Dataset for Submission to Kaggle/submission_fewshot_pseudolabel_b2.csv.
- Sesuai regulasi kompetisi, **pengunggahan ke Kaggle dilakukan secara manual oleh pengguna**.
- Pipeline telah mematuhi seluruh batasan teknis: maksimal 40 label manual (5 per kelas), mitigasi kebocoran data, dan penyaringan pseudo-label berkualitas tinggi.""",
    "code": r"""# ==============================================================================
# SECTION 54: FINAL SUBMISSION SUMMARY & KAGGLE UPLOAD GUIDE
# ==============================================================================
def print_final_submission_guide():
    print("=" * 75)
    print("PANDUAN AKHIR PENGUNGGAHAN SUBMISSION KAGGLE:")
    print("1. Berkas submission resmi siap di:")
    print(f"   {SUBMISSION_CSV_PATH}")
    print("2. Buka halaman kompetisi di platform Kaggle.")
    print("3. Unggah file CSV tersebut ke menu 'Submit Predictions'.")
    print("4. Berikan deskripsi model:")
    print(f"   'Few-Shot EfficientNet-B2 (40 manual labels) + Pseudo-Labeling (th>=0.95)'")
    print("5. Periksa skor Public Leaderboard setelah proses evaluasi Kaggle selesai.")
    print("=" * 75)

print_final_submission_guide()"""
})
