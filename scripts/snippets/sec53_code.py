# ==============================================================================
# SECTION 53: SUBMISSION CSV VERIFICATION
# ==============================================================================
def verify_submission_csv(csv_path: Path, sample_sub_path: Path) -> bool:
    if not csv_path.exists():
        print(f"[PERINGATAN] Berkas submission {csv_path.name} belum ada.")
        return False
        
    sample_df = pd.read_csv(sample_sub_path, dtype={"image_id": str})
    sub_df = pd.read_csv(csv_path, dtype={"image_id": str})
    
    assert len(sub_df) == 480, f"Jumlah baris ({len(sub_df)}) tidak sama dengan 480!"
    assert list(sub_df.columns) == ["image_id", "waste_category"], f"Kolom tidak valid: {list(sub_df.columns)}"
    assert (sub_df["image_id"] == sample_df["image_id"]).all(), "Urutan image_id tidak cocok dengan template!"
    assert not sub_df["image_id"].duplicated().any(), "Ditemukan duplikasi image_id!"
    assert sub_df["waste_category"].notna().all(), "Ditemukan nilai waste_category yang kosong!"
    
    invalid_cats = set(sub_df["waste_category"]) - set(CLASS_NAMES)
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
    verify_submission_csv(SUBMISSION_CSV_PATH, SAMPLE_SUBMISSION_PATH)
else:
    print(f"[STATUS] Berkas submission {SUBMISSION_CSV_PATH.name} belum tersedia untuk diverifikasi.")