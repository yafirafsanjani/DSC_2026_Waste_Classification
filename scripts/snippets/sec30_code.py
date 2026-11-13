# ==============================================================================
# SECTION 30: PSEUDO-LABEL FILTERING
# ==============================================================================
CONFIDENCE_THRESHOLD = CONFIG.get("pseudo_label_confidence_threshold", 0.95)
ACCEPTED_PSEUDO_PATH = MANIFEST_DIR / "accepted_pseudo_labels.csv"

def filter_pseudo_labels(
    raw_df: pd.DataFrame,
    threshold: float,
    output_path: Path
) -> pd.DataFrame:
    if raw_df.empty or "confidence" not in raw_df.columns:
        print("[PERINGATAN] Data pseudo-labels mentah kosong.")
        return pd.DataFrame()
        
    accepted = raw_df[raw_df["confidence"] >= threshold].copy()
    accepted["pseudo_weight"] = CONFIG.get("pseudo_label_loss_weight", 0.5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    accepted.to_csv(output_path, index=False)
    return accepted

if ACCEPTED_PSEUDO_PATH.exists():
    accepted_pseudo_df = pd.read_csv(ACCEPTED_PSEUDO_PATH, dtype={"image_id": str})
    print(f"[LOADED] Accepted pseudo-labels dimuat dari disk: {len(accepted_pseudo_df)} citra.")
elif len(raw_pseudo_df) > 0:
    print(f"[RUN-ALL] Menyaring pseudo-label dengan confidence threshold >= {CONFIDENCE_THRESHOLD}...")
    accepted_pseudo_df = filter_pseudo_labels(raw_pseudo_df, CONFIDENCE_THRESHOLD, ACCEPTED_PSEUDO_PATH)
    print(f"[SAVED] {len(accepted_pseudo_df)} / {len(raw_pseudo_df)} pseudo-labels diterima dan disimpan ke {ACCEPTED_PSEUDO_PATH}")
else:
    print("[STATUS] Raw pseudo-labels belum tersedia.")
    accepted_pseudo_df = pd.DataFrame()