# ==============================================================================
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
        if img_id not in predictions:
            raise KeyError(f"Prediksi untuk image_id '{img_id}' tidak ditemukan!")
        records.append({
            "image_id": str(img_id),
            "waste_category": str(predictions[img_id])
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
    submission_df = pd.DataFrame()