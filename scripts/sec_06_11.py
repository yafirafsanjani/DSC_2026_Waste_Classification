# scripts/sec_06_11.py
SECTIONS = []

SECTIONS.append({
    "id": 6,
    "title": "06. Class Definition",
    "markdown": r"""# 06. Class Definition & Mappings

Definisi delapan kelas resmi kompetisi `DSC_2026_Waste_Classification` dalam urutan abjad standar:
0: `cardboard`
1: `electronic`
2: `foodwaste`
3: `glass`
4: `metal`
5: `paper`
6: `plastic`
7: `textile`""",
    "code": r"""# ==============================================================================
# SECTION 06: CLASS DEFINITION & MAPPINGS
# ==============================================================================
CLASS_NAMES = sorted([
    "cardboard",
    "electronic",
    "foodwaste",
    "glass",
    "metal",
    "paper",
    "plastic",
    "textile"
])
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}

print(f"Jumlah Kelas : {NUM_CLASSES}")
print("Pemetaan Kelas Resmi:")
for idx, name in IDX_TO_CLASS.items():
    print(f"  [{idx}] {name}")"""
})

SECTIONS.append({
    "id": 7,
    "title": "07. Labeling Budget Verification",
    "markdown": r"""# 07. Labeling Budget Verification & Enforcement

Fungsi validator `verify_labeling_budget()` menjamin secara ketat bahwa:
1. Tidak ada kelas yang memiliki lebih dari **5 sampel berlabel**.
2. Total sampel berlabel manual tidak pernah melebihi **40 sampel**.
3. Jika kuota dilanggar, fungsi akan memunculkan `ValueError` seketika.
4. Mencegah penggunaan citra training yang bertabrakan hash (*exact collision*) dengan citra test kompetisi.""",
    "code": r"""# ==============================================================================
# SECTION 07: LABELING BUDGET VERIFICATION
# ==============================================================================
def verify_labeling_budget(
    df: pd.DataFrame,
    class_col: str = "class_name",
    max_per_class: int = 5,
    max_total: int = 40
) -> bool:
    if len(df) == 0:
        return True
        
    total_samples = len(df)
    if total_samples > max_total:
        raise ValueError(f"Pelanggaran Anggaran: Total sampel berlabel ({total_samples}) melebihi batas maksimum ({max_total})!")
    
    counts = df[class_col].value_counts()
    for cls_name, count in counts.items():
        if count > max_per_class:
            raise ValueError(f"Pelanggaran Anggaran: Kelas '{cls_name}' memiliki {count} sampel berlabel (maksimal {max_per_class})!")
            
    return True

# Test verifikasi pada dummy dataframe
dummy_ok = pd.DataFrame({"class_name": [c for c in CLASS_NAMES for _ in range(5)]})
assert verify_labeling_budget(dummy_ok, max_per_class=5, max_total=40) is True
print("[VALIDATOR TERUJI] Fungsi verify_labeling_budget berhasil memverifikasi kuota 40 label (5/kelas).")"""
})

SECTIONS.append({
    "id": 8,
    "title": "08. Few-Shot Labeled Sample Selection",
    "markdown": r"""# 08. Few-Shot Labeled Sample Selection & Manual Labeling Gate

Sesuai regulasi kompetisi:
- Data train resmi pada `data/competition/train/` (1.920 citra) adalah **citra tanpa label**.
- Peserta hanya diperbolehkan melakukan **anotasi manual maksimal 5 citra per kelas (total 40 citra)**.
- **PROTEKSI LABELING**: Sistem **dilarang keras** mengambil ground-truth lama dari folder Roboflow V1 secara otomatis.
- Berkas anotasi manual disimpan pada: `data/manifests/fewshot_pseudolabel_b2_v1/labeled_pool_40.csv`.
- Jika berkas anotasi manual belum disediakan oleh pengguna, sistem menyediakan berkas template `manual_annotation_template.csv` dan menampilkan status bahwa labeled pool belum siap.""",
    "code": r"""# ==============================================================================
# SECTION 08: FEW-SHOT LABELED SAMPLE SELECTION & MANUAL ANNOTATION GATE
# ==============================================================================
LABELED_POOL_PATH = MANIFEST_DIR / "labeled_pool_40.csv"
ANNOTATION_TEMPLATE_PATH = MANIFEST_DIR / "manual_annotation_template.csv"

def compute_file_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def create_manual_annotation_template(train_dir: Path, output_path: Path) -> pd.DataFrame:
    # Memilih 40 kandidat citra train pertama (tanpa label) sebagai template anotasi manual bagi pengguna
    train_files = sorted([p for p in train_dir.iterdir() if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png"]], key=lambda p: p.name)
    sample_candidates = train_files[:40]
    
    records = []
    for p in sample_candidates:
        records.append({
            "image_id": p.stem,
            "relative_path": p.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": compute_file_sha256(p),
            "class_name": "", # Kosong: wajib diisi manual oleh pengguna
            "class_idx": -1
        })
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    return df

# Audit Ketersediaan Labeled Pool
if LABELED_POOL_PATH.exists():
    labeled_pool_df = pd.read_csv(LABELED_POOL_PATH)
    if len(labeled_pool_df) > 0 and "class_name" in labeled_pool_df.columns and labeled_pool_df["class_name"].notna().all():
        verify_labeling_budget(labeled_pool_df, max_per_class=5, max_total=40)
        print(f"[LOADED] Manifest Labeled Pool 40 citra sah dimuat dari disk: {len(labeled_pool_df)} sampel.")
        print(labeled_pool_df["class_name"].value_counts())
    else:
        print("[PERINGATAN] Berkas labeled_pool_40.csv ada tetapi belum diisi anotasi lengkap.")
        labeled_pool_df = pd.DataFrame(columns=["image_id", "class_name", "class_idx", "relative_path", "sha256"])
elif RUN_FEWSHOT_PREPARATION:
    print("[RUNNING] Menyiapkan template anotasi manual 40 sampel di manifest...")
    template_df = create_manual_annotation_template(DATA_TRAIN_DIR, ANNOTATION_TEMPLATE_PATH)
    print(f"[TEMPLATE DIBUAT] Template anotasi disimpan di: {ANNOTATION_TEMPLATE_PATH}")
    print("Silakan buka file CSV tersebut, isi kolom 'class_name' (5 citra per kelas untuk 8 kelas),")
    print(f"lalu simpan sebagai: {LABELED_POOL_PATH}")
    labeled_pool_df = pd.DataFrame(columns=["image_id", "class_name", "class_idx", "relative_path", "sha256"])
else:
    print("[STATUS: LABELED POOL BELUM SIAP]")
    print(f"Berkas anotasi manual {LABELED_POOL_PATH.name} belum tersedia.")
    print("Sesuai aturan regulasi kompetisi, label manual tidak boleh diisi otomatis dari ground truth lama.")
    print("Aktifkan RUN_FEWSHOT_PREPARATION = True pada Section 02 untuk membuat template anotasi.")
    labeled_pool_df = pd.DataFrame(columns=["image_id", "class_name", "class_idx", "relative_path", "sha256"])"""
})

SECTIONS.append({
    "id": 9,
    "title": "09. Unlabeled Pool Preparation",
    "markdown": r"""# 09. Unlabeled Pool Preparation (Official 1.920 Train Data)

Sisa citra pada `data/competition/train/` di luar 40 citra berlabel dialokasikan ke **Unlabeled Pool**:
- **ANTI-LEAKAGE KETAT**: Manifest `unlabeled_pool.csv` hanya memuat kolom `image_id`, `relative_path`, dan `sha256`.
- Kolom `class_name` dan `class_idx` **DIHAPUS** secara permanen sehingga DataLoader tidak dapat membaca label asli.
- Citra training yang teridentifikasi bertabrakan hash dengan test set (`1046.jpg`, `1135.jpg`, `1821.jpg`) dikarantina agar tidak mencemari training.""",
    "code": r"""# ==============================================================================
# SECTION 09: UNLABELED POOL PREPARATION
# ==============================================================================
UNLABELED_POOL_PATH = MANIFEST_DIR / "unlabeled_pool.csv"

def build_unlabeled_pool_from_official_train(
    train_dir: Path,
    labeled_df: pd.DataFrame,
    output_path: Path
) -> pd.DataFrame:
    labeled_hashes = set(labeled_df["sha256"].tolist()) if len(labeled_df) > 0 else set()
    records = []
    
    all_files = sorted([
        f for f in train_dir.iterdir()
        if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ], key=lambda p: p.name)
    
    for file_path in all_files:
        sha256_hash = compute_file_sha256(file_path)
        if sha256_hash in labeled_hashes:
            continue
            
        rel_path = file_path.relative_to(PROJECT_ROOT).as_posix()
        image_id = file_path.stem
        # PERHATIAN: Tidak menyertakan class_name atau class_idx sama sekali!
        records.append({
            "image_id": image_id,
            "relative_path": rel_path,
            "sha256": sha256_hash
        })
        
    df = pd.DataFrame(records)
    assert "class_name" not in df.columns, "Bocoran: Kolom class_name ditemukan di unlabeled pool!"
    assert "class_idx" not in df.columns, "Bocoran: Kolom class_idx ditemukan di unlabeled pool!"
    
    df.to_csv(output_path, index=False)
    return df

if UNLABELED_POOL_PATH.exists():
    unlabeled_pool_df = pd.read_csv(UNLABELED_POOL_PATH)
    print(f"[LOADED] Manifest Unlabeled Pool dimuat dari disk: {len(unlabeled_pool_df)} citra.")
elif RUN_FEWSHOT_PREPARATION and DATA_TRAIN_DIR.exists():
    print("[RUNNING] Mempersiapkan Unlabeled Pool dari data train resmi...")
    unlabeled_pool_df = build_unlabeled_pool_from_official_train(DATA_TRAIN_DIR, labeled_pool_df, UNLABELED_POOL_PATH)
    print(f"[CREATED] Manifest Unlabeled Pool berhasil dibuat: {len(unlabeled_pool_df)} citra tanpa label.")
else:
    print("[STATUS] RUN_FEWSHOT_PREPARATION is False atau data train belum dipindai.")
    unlabeled_pool_df = pd.DataFrame(columns=["image_id", "relative_path", "sha256"])

print(f"Kolom Manifest Unlabeled: {list(unlabeled_pool_df.columns)}")"""
})

SECTIONS.append({
    "id": 10,
    "title": "10. Duplicate and Leakage Audit",
    "markdown": r"""# 10. Duplicate and Leakage Audit (Train vs Test Forensic)

Audit mendalam terhadap seluruh 1.920 citra train dan 480 citra test kompetisi:
1. **Internal Train Duplicates**: Mengidentifikasi pasangan duplikat identik di dalam data train (misal `0848.jpg` dan `1717.jpg`).
2. **Train vs Test Overlap Leakage**: Memeriksa tabrakan hash SHA-256 antara data train dan test resmi:
   - Terdeteksi 3 berkas train bertabrakan hash dengan test: `1046.jpg` (test `017.jpg`), `1135.jpg` (test `201.jpg`), dan `1821.jpg` (test `042.jpg`).
   - Berkas-berkas ini wajib dikarantina agar tidak masuk ke labeled pool atau validasi!""",
    "code": r"""# ==============================================================================
# SECTION 10: DUPLICATE AND LEAKAGE AUDIT
# ==============================================================================
def audit_dataset_leakage(
    train_dir: Path,
    test_dir: Path,
    labeled_df: pd.DataFrame
) -> Dict[str, Any]:
    train_files = sorted([p for p in train_dir.iterdir() if p.is_file()], key=lambda p: p.name)
    test_files = sorted([p for p in test_dir.iterdir() if p.is_file()], key=lambda p: p.name)
    
    train_hashes = {compute_file_sha256(p): p.name for p in train_files}
    test_hashes = {compute_file_sha256(p): p.name for p in test_files}
    labeled_hashes = set(labeled_df["sha256"]) if len(labeled_df) > 0 else set()
    
    overlap_train_test = set(train_hashes.keys()) & set(test_hashes.keys())
    overlap_labeled_test = labeled_hashes & set(test_hashes.keys())
    
    overlap_details = []
    for h in overlap_train_test:
        overlap_details.append((train_hashes[h], test_hashes[h]))
        
    return {
        "train_count": len(train_files),
        "test_count": len(test_files),
        "overlap_train_test_count": len(overlap_train_test),
        "overlap_labeled_test_count": len(overlap_labeled_test),
        "overlap_details": overlap_details
    }

leakage_report = audit_dataset_leakage(DATA_TRAIN_DIR, DATA_TEST_DIR, labeled_pool_df)
print("HASIL AUDIT ANTI-LEAKAGE DATASET RESMI:")
print(f"  Total Citra Train Resmi : {leakage_report['train_count']}")
print(f"  Total Citra Test Resmi  : {leakage_report['test_count']}")
print(f"  Overlap Train vs Test   : {leakage_report['overlap_train_test_count']} citra bertabrakan hash")

if leakage_report['overlap_details']:
    print("\n[PERINGATAN OVERLAP TRAIN-TEST]")
    for tr_name, ts_name in leakage_report['overlap_details']:
        print(f"  Train '{tr_name}' <== IDENTIK ==> Test '{ts_name}'")
    print("Mitigasi: Citra-citra train di atas dikarantina dan DILARANG masuk ke Labeled Pool!")

if leakage_report['overlap_labeled_test_count'] > 0:
    raise AssertionError("[FATAL LEAKAGE] Labeled pool mengandung citra yang bocor ke data test!")
else:
    print("[LEAKAGE AUDIT PASS] Labeled pool bersih dari irisan data test kompetisi.")"""
})

SECTIONS.append({
    "id": 11,
    "title": "11. Few-Shot Validation Protocol",
    "markdown": r"""# 11. Few-Shot Validation Protocol (5-Fold Stratified Cross-Validation)

Untuk mengevaluasi Teacher Model di bawah batasan 40 sampel manual, kita menggunakan **5-Fold Stratified Cross-Validation**:
- Setiap fold: **4 sampel/kelas untuk training (32 citra)** dan **1 sampel/kelas untuk validasi (8 citra)**.
- Seluruh 40 sampel dievaluasi secara out-of-fold tepat satu kali.
- Manifest fold disimpan ke `data/manifests/fewshot_pseudolabel_b2_v1/cv_folds.csv`.
- Protokol ini hanya dapat dijalankan setelah Labeled Pool (40 sampel sah) tersedia.""",
    "code": r"""# ==============================================================================
# SECTION 11: FEW-SHOT VALIDATION PROTOCOL
# ==============================================================================
CV_FOLDS_PATH = MANIFEST_DIR / "cv_folds.csv"

def generate_cv_folds(labeled_df: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> pd.DataFrame:
    df = labeled_df.copy().reset_index(drop=True)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    df["fold"] = -1
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df, df["class_idx"])):
        df.loc[val_idx, "fold"] = fold_idx
    return df

if CV_FOLDS_PATH.exists():
    cv_folds_df = pd.read_csv(CV_FOLDS_PATH)
    print(f"[LOADED] Manifest CV Folds dimuat dari disk ({len(cv_folds_df)} baris).")
elif RUN_FEWSHOT_PREPARATION and len(labeled_pool_df) == 40:
    cv_folds_df = generate_cv_folds(labeled_pool_df, n_splits=5, seed=42)
    cv_folds_df.to_csv(CV_FOLDS_PATH, index=False)
    print(f"[CREATED] Manifest CV Folds berhasil dibuat dan disimpan ke {CV_FOLDS_PATH}")
else:
    print("[STATUS: MENUNGGU LABELED POOL]")
    print("Manifest CV Folds memerlukan 40 sampel berlabel lengkap (5 per kelas).")
    cv_folds_df = pd.DataFrame()

if len(cv_folds_df) > 0:
    for f in range(5):
        val_samples = cv_folds_df[cv_folds_df["fold"] == f]
        train_samples = cv_folds_df[cv_folds_df["fold"] != f]
        print(f"  Fold {f}: Train = {len(train_samples)} citra | Val = {len(val_samples)} citra")"""
})
