# ==============================================================================
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
print("=" * 75)