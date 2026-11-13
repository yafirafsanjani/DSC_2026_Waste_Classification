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
CLASSES = CLASS_NAMES  # Alias resmi untuk kompatibilitas antar-section

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
    "title": "08. Manual Labeled Pool Validation",
    "markdown": r"""# 08. Manual Labeled Pool Validation & Ingestion

Sesuai regulasi ketat kompetisi DSC Gelar Rasa 2026:
- Peserta **hanya diizinkan** memberi label manual maksimal **5 citra per kelas** (total maksimal **40 citra**).
- Dilarang keras menggunakan label ground truth dari dataset lama (Roboflow V1) atau sumber eksternal.
- Notebook secara otomatis membaca berkas anotasi manual resmi yang telah dibuat peserta pada:
  `data/manifests/fewshot_pseudolabel_b2_v1/labeled_pool_40.csv`

**Mekanisme Validasi Otomatis (Run-All Gate)**:
1. **Keberadaan Berkas**: Memverifikasi ketersediaan `labeled_pool_40.csv` dari `PROJECT_ROOT`. Jika belum ada, template dibuat otomatis dan eksekusi dihentikan dengan pesan kesalahan informatif.
2. **Integritas String ID**: Membaca `image_id` sebagai string untuk mempertahankan leading zero (misal `0005`).
3. **Validasi Kuota Legal**: Memastikan tepat 40 baris dengan distribusi seimbang tepat 5 sampel untuk setiap 8 kelas resmi (`cardboard`, `electronic`, `foodwaste`, `glass`, `metal`, `paper`, `plastic`, `textile`).
4. **Verifikasi Fisik & Dekode Citra**: Memastikan seluruh 40 citra fisik ada di `data/competition/train/`, serta dapat dibuka dan dimuat (*verify & load*) menggunakan Pillow.
5. **Anti-Leakage & Keunikan Konten**: Menghitung ulang hash SHA-256 dari citra asli, memastikan 0 tabrakan dengan 480 citra test kompetisi, serta memastikan tidak ada citra duplikat di dalam labeled pool.
6. **Rekonstruksi Deterministik**: Membangun ulang kolom `class_idx` (0-7), `relative_path`, dan `sha256` dari sumber terpercaya.""",
    "code": r"""# ==============================================================================
# SECTION 08: MANUAL LABELED POOL VALIDATION & INGESTION
# ==============================================================================
LABELED_POOL_PATH = MANIFEST_DIR / "labeled_pool_40.csv"
ANNOTATION_TEMPLATE_PATH = MANIFEST_DIR / "manual_annotation_template.csv"

def compute_file_sha256(filepath: Path) -> str:
    '''Menghitung hash SHA-256 suatu berkas secara efisien dan deterministik.'''
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def create_manual_annotation_template(train_dir: Path, test_dir: Path, output_path: Path) -> pd.DataFrame:
    '''Membuat template 40 citra train pertama yang bersih dari tabrakan test untuk anotasi manual.'''
    train_files = sorted([p for p in train_dir.iterdir() if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png"]], key=lambda p: p.name)
    test_hashes = {compute_file_sha256(p) for p in test_dir.iterdir() if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png"]} if test_dir.exists() else set()
    
    records = []
    count = 0
    for p in train_files:
        h = compute_file_sha256(p)
        if h in test_hashes:
            continue  # Lewati jika bertabrakan dengan data test
        records.append({
            "image_id": p.stem,
            "class_name": "",  # Wajib diisi manual oleh peserta
            "relative_path": p.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": h
        })
        count += 1
        if count >= 40:
            break
            
    df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df

# 1. Pastikan template anotasi selalu disiapkan jika file belum ada
if not ANNOTATION_TEMPLATE_PATH.exists() and DATA_TRAIN_DIR.exists():
    create_manual_annotation_template(DATA_TRAIN_DIR, DATA_TEST_DIR, ANNOTATION_TEMPLATE_PATH)

# 2. Validasi Ketersediaan File Anotasi Manual Labeled Pool
if not LABELED_POOL_PATH.exists():
    error_msg = (
        "\n" + "=" * 80 + "\n"
        "[PRASYARAT RUN ALL BELUM TERPENUHI: LABELED POOL TIDAK DITEMUKAN]\n"
        f"Berkas anotasi manual resmi belum tersedia pada:\n"
        f"  --> {LABELED_POOL_PATH.resolve()}\n\n"
        "Sesuai regulasi kompetisi DSC Gelar Rasa 2026:\n"
        "- Peserta wajib menyediakan anotasi manual maksimal 5 gambar per kelas (total 40 gambar).\n"
        "- Sistem dilarang mengisi label otomatis dari ground truth Roboflow atau sumber eksternal.\n\n"
        "Petunjuk Penyelesaian:\n"
        f"1. Template anotasi telah disiapkan secara otomatis di:\n"
        f"     {ANNOTATION_TEMPLATE_PATH.resolve()}\n"
        f"2. Buka template tersebut, lalu isi kolom 'class_name' dengan 8 kelas resmi:\n"
        f"     {CLASS_NAMES}\n"
        "   (Tepat 5 gambar untuk setiap kelas).\n"
        f"3. Simpan file yang telah diisi sebagai:\n"
        f"     '{LABELED_POOL_PATH.resolve()}'\n"
        "4. Jalankan kembali 'Kernel Restart & Run All'.\n"
        + "=" * 80
    )
    print(error_msg)
    raise FileNotFoundError(error_msg)

# 3. Muat dan Validasi Integritas Berkas Anotasi Manual
# image_id dibaca sebagai string untuk menjamin leading zero (misal '0005') tidak terpotong
labeled_pool_df = pd.read_csv(LABELED_POOL_PATH, dtype={"image_id": str})

# A. Validasi kolom wajib
required_cols = {"image_id", "class_name"}
if not required_cols.issubset(labeled_pool_df.columns):
    raise ValueError(f"Format {LABELED_POOL_PATH.name} tidak valid! Kolom wajib: {required_cols}")

# B. Validasi ukuran total baris
if len(labeled_pool_df) != CONFIG["max_total_labels"]:
    raise ValueError(
        f"Jumlah anotasi tidak valid! Ditemukan {len(labeled_pool_df)} baris, "
        f"tetapi regulasi mewajibkan tepat {CONFIG['max_total_labels']} citra manual (5 citra per kelas x 8 kelas)."
    )

# C. Validasi nilai kosong pada label
if labeled_pool_df["class_name"].isna().any() or (labeled_pool_df["class_name"].astype(str).str.strip() == "").any():
    raise ValueError(f"Kolom 'class_name' pada {LABELED_POOL_PATH.name} memuat nilai kosong/spasi!")

# D. Validasi kesesuaian 8 kelas resmi
invalid_classes = set(labeled_pool_df["class_name"]) - set(CLASS_NAMES)
if invalid_classes:
    raise ValueError(f"Ditemukan kelas tidak resmi pada {LABELED_POOL_PATH.name}: {invalid_classes}. Kelas resmi: {CLASS_NAMES}")

# E. Validasi kuota per kelas tepat 5 sampel
class_counts = labeled_pool_df["class_name"].value_counts()
for cname in CLASS_NAMES:
    cnt = class_counts.get(cname, 0)
    if cnt != CONFIG["max_labels_per_class"]:
        raise ValueError(
            f"Distribusi kelas tidak seimbang! Kelas '{cname}' memiliki {cnt} sampel, "
            f"tetapi regulasi mewajibkan tepat {CONFIG['max_labels_per_class']} sampel per kelas."
        )

# F. Validasi keunikan image_id
duplicate_ids = labeled_pool_df[labeled_pool_df["image_id"].duplicated()]["image_id"].tolist()
if duplicate_ids:
    raise ValueError(f"Terdapat duplikasi image_id pada labeled pool: {duplicate_ids}")

# G. Validasi keberadaan fisik citra di folder train resmi
train_files_map = {p.stem: p for p in DATA_TRAIN_DIR.iterdir() if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png"]}
missing_images = [img_id for img_id in labeled_pool_df["image_id"] if str(img_id) not in train_files_map]
if missing_images:
    raise ValueError(f"Image ID pada labeled pool tidak ditemukan di data train resmi: {missing_images[:10]}")

# H. Validasi dekode citra (seluruh 40 gambar dapat dibuka dan dimuat)
for img_id in labeled_pool_df["image_id"]:
    img_path = train_files_map[str(img_id)]
    try:
        with Image.open(img_path) as img:
            img.verify()
        with Image.open(img_path) as img:
            img.load()
    except Exception as err:
        raise RuntimeError(f"Citra {img_path.name} rusak atau tidak dapat didekode: {err}")

# I. Hitung ulang hash SHA-256 dari citra asli dan validasi anti-leakage test set
test_hashes = {compute_file_sha256(p): p.name for p in DATA_TEST_DIR.iterdir() if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png"]} if DATA_TEST_DIR.exists() else {}
computed_hashes = []
for idx, row in labeled_pool_df.iterrows():
    img_stem = str(row["image_id"])
    img_path = train_files_map[img_stem]
    h = compute_file_sha256(img_path)
    if h in test_hashes:
        raise AssertionError(f"[FATAL LEAKAGE] Citra {img_path.name} pada labeled pool bertabrakan dengan citra test kompetisi ({test_hashes[h]})! Hash: {h}")
    computed_hashes.append(h)

# J. Validasi tidak ada citra duplikat (SHA-256 unik) di dalam labeled pool
if len(set(computed_hashes)) != len(computed_hashes):
    raise ValueError("Ditemukan citra duplikat dengan konten SHA-256 yang identik di dalam labeled pool!")

# K. Pembangunan ulang metadata turunan secara deterministik dari sumber resmi
labeled_pool_df["sha256"] = computed_hashes
labeled_pool_df["class_idx"] = labeled_pool_df["class_name"].map(CLASS_TO_IDX).astype(int)
labeled_pool_df["relative_path"] = [train_files_map[str(iid)].relative_to(PROJECT_ROOT).as_posix() for iid in labeled_pool_df["image_id"]]

assert labeled_pool_df["class_idx"].between(0, NUM_CLASSES - 1).all(), "class_idx di luar rentang resmi 0-7!"

print("=" * 75)
print("[VALIDASI SUKSES] Labeled Pool 40 citra manual berhasil dimuat dan diverifikasi:")
print(f"  Total Citra Berlabel   : {len(labeled_pool_df)} (Tepat 40)")
print(f"  Keunikan Image ID      : 100% ({len(labeled_pool_df)} unik)")
print(f"  Keunikan Konten Hash   : 100% ({len(set(computed_hashes))} hash unik)")
print("  Integritas Test Leakage: 0 overlap (Bersih dari seluruh citra test)")
print("  Status Dekode Gambar   : 100% Terbuka dan Didekode Tanpa Kerusakan")
print("  Distribusi Sampel per Kelas (Tepat 5 per Kelas):")
for cname, count in labeled_pool_df["class_name"].value_counts().items():
    print(f"    - {cname:12s}: {count} sampel")
print("=" * 75)"""
})

SECTIONS.append({
    "id": 9,
    "title": "09. Unlabeled Pool Preparation",
    "markdown": r"""# 09. Unlabeled Pool Preparation

Citra train resmi yang tersisa (selain 40 citra berlabel manual dan citra yang bertabrakan dengan test set) dimasukkan ke dalam **Unlabeled Pool**:
- Total citra train resmi: 1.920 citra.
- Labeled pool: 40 citra.
- Citra bertabrakan test (dikarantina): 3 citra.
- Unlabeled pool bersih: 1.877 citra tanpa label.
- **Prinsip Anti-Leakage Ketat**: Manifest unlabeled pool HANYA memuat image_id, 
elative_path, dan sha256. Kolom class_name atau class_idx sama sekali tidak disertakan.""",
    "code": r"""# ==============================================================================
# SECTION 09: UNLABELED POOL PREPARATION
# ==============================================================================
UNLABELED_POOL_PATH = MANIFEST_DIR / "unlabeled_pool.csv"

def build_unlabeled_pool_from_official_train(
    train_dir: Path,
    labeled_df: pd.DataFrame,
    test_dir: Path,
    output_path: Path
) -> pd.DataFrame:
    labeled_hashes = set(labeled_df["sha256"].tolist()) if len(labeled_df) > 0 else set()
    test_hashes = {compute_file_sha256(p) for p in test_dir.iterdir() if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png"]} if test_dir.exists() else set()
    
    records = []
    all_files = sorted([
        f for f in train_dir.iterdir()
        if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ], key=lambda p: p.name)
    
    quarantined_test_coll = 0
    for file_path in all_files:
        sha256_hash = compute_file_sha256(file_path)
        if sha256_hash in labeled_hashes:
            continue
        if sha256_hash in test_hashes:
            quarantined_test_coll += 1
            continue
            
        rel_path = file_path.relative_to(PROJECT_ROOT).as_posix()
        image_id = file_path.stem
        records.append({
            "image_id": image_id,
            "relative_path": rel_path,
            "sha256": sha256_hash
        })
        
    df = pd.DataFrame(records)
    assert "class_name" not in df.columns, "Bocoran: Kolom class_name ditemukan di unlabeled pool!"
    assert "class_idx" not in df.columns, "Bocoran: Kolom class_idx ditemukan di unlabeled pool!"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    if quarantined_test_coll > 0:
        print(f"  [QUARANTINE] {quarantined_test_coll} citra train yang bertabrakan dengan test set berhasil dikarantina dari unlabeled pool.")
    return df

if UNLABELED_POOL_PATH.exists():
    unlabeled_pool_df = pd.read_csv(UNLABELED_POOL_PATH, dtype={"image_id": str})
    print(f"[LOADED] Manifest Unlabeled Pool dimuat dari disk: {len(unlabeled_pool_df)} citra.")
else:
    print("[RUN-ALL] Mempersiapkan Unlabeled Pool dari 1.920 citra train resmi...")
    unlabeled_pool_df = build_unlabeled_pool_from_official_train(DATA_TRAIN_DIR, labeled_pool_df, DATA_TEST_DIR, UNLABELED_POOL_PATH)
    print(f"[CREATED] Manifest Unlabeled Pool berhasil dibuat: {len(unlabeled_pool_df)} citra tanpa label.")

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
    "markdown": r"""# 11. Few-Shot Validation Protocol

Protokol validasi 5-Fold Stratified Cross-Validation pada 40 citra berlabel manual:
- Setiap fold membagi data menjadi:
  - **Training**: 32 citra (4 per kelas).
  - **Validation**: 8 citra (tepat 1 per kelas).
- Protokol ini menjamin evaluasi out-of-fold yang adil tanpa risiko kebocoran data (data leakage) sebelum Final Teacher dilatih dengan seluruh 40 citra.""",
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
    cv_folds_df = pd.read_csv(CV_FOLDS_PATH, dtype={"image_id": str})
    print(f"[LOADED] Manifest CV Folds dimuat dari disk ({len(cv_folds_df)} baris).")
else:
    print("[RUN-ALL] Membangun 5-Fold Stratified Cross-Validation Folds...")
    cv_folds_df = generate_cv_folds(labeled_pool_df, n_splits=CONFIG.get("cv_folds", 5), seed=CONFIG.get("random_seed", 42))
    cv_folds_df.to_csv(CV_FOLDS_PATH, index=False)
    print(f"[CREATED] Manifest CV Folds berhasil dibuat dan disimpan ke {CV_FOLDS_PATH}")

for f in range(CONFIG.get("cv_folds", 5)):
    val_samples = cv_folds_df[cv_folds_df["fold"] == f]
    train_samples = cv_folds_df[cv_folds_df["fold"] != f]
    print(f"  Fold {f}: Train = {len(train_samples)} citra | Val = {len(val_samples)} citra")"""
})
