# ==============================================================================
# SECTION 38: STUDENT TRAINING EXECUTION
# ==============================================================================
if STUDENT_BEST_CKPT.exists() and STUDENT_METRICS_PATH.exists():
    with open(STUDENT_METRICS_PATH, "r", encoding="utf-8") as f:
        student_results = json.load(f)
    print(f"[LOADED] Checkpoint Student tersedia: {STUDENT_BEST_CKPT}")
elif len(labeled_pool_df) > 0 and len(accepted_pseudo_df) > 0:
    print(f"[RUN-ALL] Melatih model Student semi-supervised pada dataset gabungan ({len(labeled_pool_df)} manual + {len(accepted_pseudo_df)} pseudo)...")
    student_results = train_student_model(labeled_pool_df, accepted_pseudo_df, STUDENT_CONFIG)
    with open(STUDENT_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(student_results, f, indent=2)
    print(f"[SAVED] Checkpoint Student berhasil disimpan ke {STUDENT_BEST_CKPT}")
else:
    print("[STATUS] Data labeled pool atau accepted pseudo-labels belum siap.")
    student_results = {}