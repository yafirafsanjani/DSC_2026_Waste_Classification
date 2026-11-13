# ==============================================================================
# SECTION 24: FINAL TEACHER TRAINING
# ==============================================================================
if TEACHER_BEST_CKPT.exists() and TEACHER_METRICS_PATH.exists():
    with open(TEACHER_METRICS_PATH, "r", encoding="utf-8") as f:
        teacher_results = json.load(f)
    print(f"[LOADED] Checkpoint Final Teacher tersedia: {TEACHER_BEST_CKPT}")
elif len(labeled_pool_df) > 0:
    print("[RUN-ALL] Melatih Final Teacher model menggunakan seluruh 40 citra berlabel...")
    teacher_results = train_final_teacher(labeled_pool_df, FINAL_TEACHER_CONFIG)
    with open(TEACHER_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(teacher_results, f, indent=2)
    print(f"[SAVED] Final Teacher Checkpoint disimpan ke {TEACHER_BEST_CKPT}")
else:
    print("[STATUS] Labeled pool belum tersedia.")
    teacher_results = {}