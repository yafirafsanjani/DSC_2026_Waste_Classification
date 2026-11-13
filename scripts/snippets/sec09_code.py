# ==============================================================================
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

print(f"Kolom Manifest Unlabeled: {list(unlabeled_pool_df.columns)}")