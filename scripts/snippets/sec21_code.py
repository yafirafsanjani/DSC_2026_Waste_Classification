# ==============================================================================
# SECTION 21: CROSS-VALIDATION EXECUTION
# ==============================================================================
if CV_METRICS_PATH.exists():
    with open(CV_METRICS_PATH, "r", encoding="utf-8") as f:
        cv_results = json.load(f)
    print(f"[LOADED] Riwayat 5-Fold CV dimuat dari {CV_METRICS_PATH}.")
elif len(cv_folds_df) > 0:
    print("[RUN-ALL] Menjalankan 5-Fold Cross-Validation Teacher secara berurutan...")
    cv_results = []
    for f in range(CV_CONFIG["num_folds"]):
        res = run_cv_fold_training(f, cv_folds_df, CV_CONFIG)
        cv_results.append(res)
    with open(CV_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(cv_results, f, indent=2)
    print(f"[COMPLETED] 5-Fold CV selesai. Riwayat disimpan ke {CV_METRICS_PATH}")
else:
    print("[STATUS] Data CV Folds belum tersedia.")
    cv_results = []