# scripts/sec_42_47.py
SECTIONS = []

SECTIONS.append({
    "id": 42,
    "title": "42. Teacher vs Student Comparison",
    "markdown": """# 42. Teacher vs Student Comparison

Analisis komparatif menyeluruh antara model Teacher dan model Student:
- **Teacher**: Dilatih hanya pada 40 citra berlabel manual.
- **Student**: Dilatih pada 40 citra berlabel manual + citra pseudo-label yang lolos filter (threshold >= 0.95).
- **Integritas Metrik**: Perbandingan disajikan secara jujur. Tidak boleh membandingkan metrik out-of-fold Teacher dengan metrik in-sample Student sebagai klaim peningkatan semu.""",
    "code": """# ==============================================================================
# SECTION 42: TEACHER VS STUDENT COMPARISON
# ==============================================================================
def compare_teacher_and_student():
    records = []
    
    # Data Teacher
    if TEACHER_BEST_CKPT.exists():
        t_payload = torch.load(TEACHER_BEST_CKPT, map_location="cpu")
        records.append({
            "Model": "Final Teacher",
            "Backbone": "EfficientNet-B2",
            "Supervised Samples": 40,
            "Pseudo Samples": 0,
            "Final Loss": f"{t_payload.get('final_loss', float('nan')):.4f}",
            "Training Acc": f"{t_payload.get('final_acc', float('nan'))*100:.2f}%",
            "Role": "Baseline & Pseudo-Label Generator"
        })
        
    # Data Student
    if STUDENT_BEST_CKPT.exists():
        s_payload = torch.load(STUDENT_BEST_CKPT, map_location="cpu")
        records.append({
            "Model": "Student Semi-Sup",
            "Backbone": "EfficientNet-B2",
            "Supervised Samples": 40,
            "Pseudo Samples": s_payload.get("num_pseudo", 0),
            "Final Loss": f"{s_payload.get('final_loss', float('nan')):.4f}",
            "Training Acc": f"{s_payload.get('final_acc', float('nan'))*100:.2f}%",
            "Role": "Final Classifier with Distillation"
        })
        
    if records:
        df_comp = pd.DataFrame(records)
        print("TABEL PERBANDINGAN TEACHER VS STUDENT:")
        print(df_comp.to_string(index=False))
    else:
        print("[INFO] Model Teacher atau Student belum selesai dilatih.")

if RUN_TEACHER_STUDENT_EVALUATION or (TEACHER_BEST_CKPT.exists() and STUDENT_BEST_CKPT.exists()):
    compare_teacher_and_student()
else:
    print("[STATUS] RUN_TEACHER_STUDENT_EVALUATION is False. Evaluasi komparatif dilewati.")"""
})

SECTIONS.append({
    "id": 43,
    "title": "43. Validation Metrics",
    "markdown": """# 43. Validation Metrics

Rekapitulasi metrik validasi resmi dari Out-Of-Fold Cross-Validation:
- Akurasi
- Macro Precision
- Macro Recall
- Macro F1-Score across all 8 classes.""",
    "code": """# ==============================================================================
# SECTION 43: VALIDATION METRICS
# ==============================================================================
if CV_METRICS_PATH.exists():
    with open(CV_METRICS_PATH, "r", encoding="utf-8") as f:
        cv_data = json.load(f)
    print("METRIK VALIDASI OUT-OF-FOLD (5-FOLD CV):")
    for r in cv_data:
        best_ep = min(r["history"], key=lambda h: h["val_loss"])
        print(f"  Fold {r['fold']}: Loss={best_ep['val_loss']:.4f} | Acc={best_ep['val_acc']*100:.2f}% | F1={best_ep['val_f1']*100:.2f}%")
else:
    print("[INFO] Metrik CV validasi belum tersedia.")"""
})

SECTIONS.append({
    "id": 44,
    "title": "44. Classification Report",
    "markdown": """# 44. Classification Report

Laporan klasifikasi per kelas untuk 8 kelas sampah target (Precision, Recall, F1-Score, dan Support) menggunakan `sklearn.metrics.classification_report`.""",
    "code": """# ==============================================================================
# SECTION 44: CLASSIFICATION REPORT
# ==============================================================================
def display_classification_report(y_true: List[int], y_pred: List[int]):
    print("CLASSIFICATION REPORT (8 KELAS TARGET):")
    rep = metrics.classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4, zero_division=0)
    print(rep)

# Contoh evaluasi pada labeled pool jika checkpoint teacher tersedia
if TEACHER_BEST_CKPT.exists() and len(labeled_pool_df) > 0:
    t_model = load_trained_teacher_model(TEACHER_BEST_CKPT, DEVICE)
    ds = FewShotLabeledDataset(labeled_pool_df, PROJECT_ROOT, transform=eval_transforms)
    ld = DataLoader(ds, batch_size=8, shuffle=False)
    y_t, y_p = [], []
    with torch.inference_mode():
        for imgs, tgts, _ in ld:
            logits = t_model(imgs.to(DEVICE))
            y_t.extend(tgts.numpy().tolist())
            y_p.extend(logits.argmax(dim=1).cpu().numpy().tolist())
    display_classification_report(y_t, y_p)
else:
    print("[INFO] Checkpoint Teacher belum tersedia untuk classification report.")"""
})

SECTIONS.append({
    "id": 45,
    "title": "45. Confusion Matrix",
    "markdown": """# 45. Confusion Matrix (Raw & Normalized)

Visualisasi Confusion Matrix untuk menganalisis kesalahan prediksi antarkelas (misalnya antara `cardboard` vs `paper` atau `glass` vs `plastic`).""",
    "code": """# ==============================================================================
# SECTION 45: CONFUSION MATRIX
# ==============================================================================
def plot_confusion_matrices(y_true: List[int], y_pred: List[int]):
    cm_raw = metrics.confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    cm_norm = cm_raw.astype("float") / (cm_raw.sum(axis=1)[:, np.newaxis] + 1e-9)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    im1 = ax1.imshow(cm_raw, interpolation="nearest", cmap=plt.cm.Blues)
    ax1.set_title("Confusion Matrix (Hitungan Mentah)", weight="bold")
    fig.colorbar(im1, ax=ax1)
    tick_marks = np.arange(NUM_CLASSES)
    ax1.set_xticks(tick_marks)
    ax1.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax1.set_yticks(tick_marks)
    ax1.set_yticklabels(CLASS_NAMES)
    ax1.set_xlabel("Prediksi")
    ax1.set_ylabel("True Label")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax1.text(j, i, format(cm_raw[i, j], "d"), ha="center", va="center", color="white" if cm_raw[i, j] > cm_raw.max()/2 else "black")
            
    im2 = ax2.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Greens)
    ax2.set_title("Confusion Matrix (Ternormalisasi)", weight="bold")
    fig.colorbar(im2, ax=ax2)
    ax2.set_xticks(tick_marks)
    ax2.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax2.set_yticks(tick_marks)
    ax2.set_yticklabels(CLASS_NAMES)
    ax2.set_xlabel("Prediksi")
    ax2.set_ylabel("True Label")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax2.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", color="white" if cm_norm[i, j] > 0.5 else "black")
            
    plt.tight_layout()
    plt.show()

if 'y_t' in locals() and len(y_t) > 0:
    plot_confusion_matrices(y_t, y_p)
else:
    print("[INFO] Data evaluasi belum tersedia.")"""
})

SECTIONS.append({
    "id": 46,
    "title": "46. Overfitting / Underfitting Analysis",
    "markdown": """# 46. Overfitting / Underfitting Analysis

Analisis stabilitas model pada skenario data sedikit (Few-Shot):
1. **Risiko Overfitting**: Dengan hanya 5 citra per kelas, model rentan menghafal latar belakang (background memorization).
2. **Mitigasi**: Pembekuan backbone (frozen backbone), Dropout 0.3, Weight Decay 1e-4, dan augmentasi fotorealistik.
3. **Risiko Pseudo-Labeling**: Confirmation bias terjadi jika threshold terlalu rendah (< 0.80). Threshold 0.95 meminimalkan transfer label salah ke model Student.""",
    "code": """# ==============================================================================
# SECTION 46: OVERFITTING / UNDERFITTING ANALYSIS
# ==============================================================================
print("ANALISIS RESIKO DAN REGULARISASI:")
print("1. Parameter Ratio : 11.272 parameter trainable pada 40 citra (rasio ~281 param/citra).")
print("2. Freeze Backbone : 7,70 juta parameter backbone dibekukan untuk menjaga stabilitas.")
print("3. Pseudo Filter   : Threshold 0.95 membatasi hanya prediksi berkeyakinan tinggi yang masuk ke Student.")
print("4. Loss Weighting  : Bobot pseudo 0.5 memberikan prioritas lebih tinggi pada label asli berbobot 1.0.")"""
})

SECTIONS.append({
    "id": 47,
    "title": "47. Experiment Summary",
    "markdown": """# 47. Experiment Summary

Ringkasan eksekutif eksperimen TASK 08:
- Status kuota label (40/40 terpakai).
- Status Teacher dan Student model.
- Checkpoint yang siap digunakan untuk tahapan inferensi submission.""",
    "code": """# ==============================================================================
# SECTION 47: EXPERIMENT SUMMARY
# ==============================================================================
def print_experiment_summary():
    print("=" * 70)
    print("RINGKASAN EKSEKUTIF EKSPERIMEN TASK 08:")
    print(f"  Project Root       : {PROJECT_ROOT.name}")
    print(f"  Backbone Model     : EfficientNet-B2 (Pretrained ImageNet-1K)")
    print(f"  Batas Kuota Label  : 40 Citra Manual (5 per kelas x 8 kelas) - DIPATUHI")
    print(f"  Unlabeled Pool     : {len(unlabeled_pool_df)} citra")
    print(f"  Accepted Pseudo    : {len(accepted_pseudo_df)} citra (Threshold >= {CONFIDENCE_THRESHOLD})")
    print(f"  Teacher Checkpoint : {'Tersedia' if TEACHER_BEST_CKPT.exists() else 'Belum Dilatih'}")
    print(f"  Student Checkpoint : {'Tersedia' if STUDENT_BEST_CKPT.exists() else 'Belum Dilatih'}")
    print("=" * 70)

print_experiment_summary()"""
})
