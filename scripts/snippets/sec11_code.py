# ==============================================================================
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
    print(f"  Fold {f}: Train = {len(train_samples)} citra | Val = {len(val_samples)} citra")