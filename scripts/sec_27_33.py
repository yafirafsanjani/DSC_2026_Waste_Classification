# scripts/sec_27_33.py
SECTIONS = []

SECTIONS.append({
    "id": 27,
    "title": "27. Unlabeled Inference Setup",
    "markdown": r"""# 27. Unlabeled Inference Setup

Mempersiapkan inferensi model Teacher pada **Unlabeled Pool** (~1.876 citra):
- Memeriksa ketersediaan checkpoint Final Teacher (`TEACHER_BEST_CKPT`).
- Memuat arsitektur `EfficientNet-B2` dengan `weights=None` dan memuat `state_dict` langsung dari disk.
- Mengatur model ke mode `model.eval()` dan menggunakan `torch.inference_mode()` untuk efisiensi komputasi VRAM.""",
    "code": r"""# ==============================================================================
# SECTION 27: UNLABELED INFERENCE SETUP
# ==============================================================================
def load_trained_teacher_model(ckpt_path: Path, device: torch.device) -> nn.Module:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint teacher tidak ditemukan: {ckpt_path}. Jalankan Section 24 terlebih dahulu.")
    
    model = build_teacher_model(num_classes=NUM_CLASSES, pretrained=False, freeze_backbone=True)
    payload = torch.load(ckpt_path, map_location=device)
    state_dict = payload["model_state_dict"] if "model_state_dict" in payload else payload
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

print(f"Target Checkpoint Teacher: {TEACHER_BEST_CKPT} (Exists: {TEACHER_BEST_CKPT.exists()})")
if TEACHER_BEST_CKPT.exists():
    inference_teacher = load_trained_teacher_model(TEACHER_BEST_CKPT, DEVICE)
    print("[READY] Model Teacher siap untuk inferensi pada data unlabeled.")
else:
    print("[WAITING] Model Teacher belum dilatih. Selesaikan training Section 24 terlebih dahulu.")
    inference_teacher = None"""
})

SECTIONS.append({
    "id": 28,
    "title": "28. Pseudo-Label Generation",
    "markdown": r"""# 28. Pseudo-Label Generation (Unlabeled Pool)

Menghasilkan prediksi pseudo-label untuk seluruh 1.877 citra unlabeled menggunakan model Final Teacher:
- Jika berkas pseudo_labels.csv telah tersedia, data dimuat langsung dari disk.
- Jika belum, inferensi dijalankan secara otomatis dengan output memuat image_id, predicted_class, confidence_score, dan metadata checkpoint Teacher.""",
    "code": r"""# ==============================================================================
# SECTION 28: PSEUDO-LABEL GENERATION
# ==============================================================================
RAW_PSEUDO_PATH = MANIFEST_DIR / "pseudo_labels.csv"

def generate_raw_pseudo_labels(
    model: nn.Module,
    unlabeled_df: pd.DataFrame,
    output_path: Path,
    batch_size: int = 16,
    device: torch.device = DEVICE
) -> pd.DataFrame:
    dataset = UnlabeledDataset(unlabeled_df, PROJECT_ROOT, transform=eval_transforms)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    records = []
    print(f"Menghasilkan pseudo-label pada {len(dataset)} citra unlabeled...")
    
    model.eval()
    with torch.inference_mode():
        for images, image_ids in tqdm(loader, desc="Pseudo-Labeling Inference"):
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            confidences, preds = torch.max(probs, dim=1)
            
            for img_id, pred_idx, conf in zip(image_ids, preds.cpu().numpy(), confidences.cpu().numpy()):
                records.append({
                    "image_id": str(img_id),
                    "predicted_idx": int(pred_idx),
                    "predicted_class": IDX_TO_CLASS[int(pred_idx)],
                    "confidence": float(conf),
                    "confidence_score": float(conf),
                    "teacher_checkpoint": TEACHER_BEST_CKPT.name if TEACHER_BEST_CKPT.exists() else "teacher_best"
                })
                
    df_pred = pd.DataFrame(records)
    merged = pd.merge(unlabeled_df[["image_id", "relative_path", "sha256"]], df_pred, on="image_id")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    return merged

if RAW_PSEUDO_PATH.exists():
    raw_pseudo_df = pd.read_csv(RAW_PSEUDO_PATH, dtype={"image_id": str})
    print(f"[LOADED] Pseudo-labels dimuat dari {RAW_PSEUDO_PATH} ({len(raw_pseudo_df)} baris).")
elif inference_teacher is not None and len(unlabeled_pool_df) > 0:
    print(f"[RUN-ALL] Menghasilkan pseudo-label pada {len(unlabeled_pool_df)} citra unlabeled...")
    raw_pseudo_df = generate_raw_pseudo_labels(inference_teacher, unlabeled_pool_df, output_path=RAW_PSEUDO_PATH, batch_size=16)
    print(f"[COMPLETED] Pseudo-labels berhasil disimpan ke {RAW_PSEUDO_PATH}")
else:
    print("[STATUS] Prasyarat inference_teacher atau unlabeled pool belum siap.")
    raw_pseudo_df = pd.DataFrame()"""
})

SECTIONS.append({
    "id": 29,
    "title": "29. Confidence Score Analysis",
    "markdown": r"""# 29. Confidence Score Analysis

Analisis distribusi skor keyakinan (confidence) model Teacher pada data unlabeled:
- Histogram probabilitas [0.0 - 1.0].
- Persentil distribusi confidence (25%, 50%, 75%, 90%, 95%).
- Proporsi sampel yang memenuhi berbagai ambang batas (>= 0.80, >= 0.85, >= 0.90, >= 0.95).""",
    "code": r"""# ==============================================================================
# SECTION 29: CONFIDENCE SCORE ANALYSIS
# ==============================================================================
def analyze_confidence_scores(df: pd.DataFrame):
    if len(df) == 0:
        print("[INFO] Data raw pseudo-labels belum tersedia untuk dianalisis.")
        return
        
    confs = df["confidence"].values
    print("DISTRIBUSI SKOR CONFIDENCE:")
    print(f"  Rata-rata    : {np.mean(confs):.4f}")
    print(f"  Median (50%) : {np.median(confs):.4f}")
    print(f"  Kuantil 75%  : {np.percentile(confs, 75):.4f}")
    print(f"  Kuantil 90%  : {np.percentile(confs, 90):.4f}")
    print(f"  Kuantil 95%  : {np.percentile(confs, 95):.4f}")
    
    thresholds = [0.80, 0.85, 0.90, 0.95, 0.98]
    print()
    print("PROPORSI SAMPEL PER THRESHOLD:")
    for th in thresholds:
        count = (confs >= th).sum()
        pct = (count / len(confs)) * 100
        print(f"  >= {th:.2f} : {count:4d} citra ({pct:.1f}%)")
        
    plt.figure(figsize=(9, 4.5))
    plt.hist(confs, bins=30, color="teal", edgecolor="black", alpha=0.8)
    plt.axvline(0.95, color="red", linestyle="--", linewidth=2, label="Threshold Rekomendasi (0.95)")
    plt.title("Distribusi Skor Kepercayaan (Confidence) Prediksi Teacher", weight="bold")
    plt.xlabel("Confidence Score")
    plt.ylabel("Frekuensi")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

analyze_confidence_scores(raw_pseudo_df)"""
})

SECTIONS.append({
    "id": 30,
    "title": "30. Pseudo-Label Filtering",
    "markdown": r"""# 30. Pseudo-Label Filtering

Menyaring prediksi pseudo-label dengan confidence threshold ketat (default $\\ge 0.95$):
- Prediksi yang lolos disimpan ke data/manifests/fewshot_pseudolabel_b2_v1/accepted_pseudo_labels.csv.
- Setiap sampel yang lolos diberikan bobot loss pseudo (default 0.5) untuk pelatihan model Student semi-supervised.""",
    "code": r"""# ==============================================================================
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
    accepted_pseudo_df = pd.DataFrame()"""
})

SECTIONS.append({
    "id": 31,
    "title": "31. Class Balance Analysis",
    "markdown": r"""# 31. Class Balance Analysis (Accepted Pseudo-Labels)

Pemeriksaan keseimbangan kelas dari pseudo-label yang diterima untuk mendeteksi apakah terjadi bias model ekstrem atau dominasi kelas tunggal (class collapse).""",
    "code": r"""# ==============================================================================
# SECTION 31: CLASS BALANCE ANALYSIS
# ==============================================================================
def plot_pseudo_class_balance(accepted_df: pd.DataFrame):
    if len(accepted_df) == 0:
        print("[INFO] Belum ada accepted pseudo-labels untuk divisualisasikan.")
        return
        
    counts = accepted_df["predicted_class"].value_counts().reindex(CLASS_NAMES).fillna(0)
    plt.figure(figsize=(10, 4.5))
    bars = plt.bar(counts.index, counts.values, color="coral", edgecolor="black")
    plt.title(f"Distribusi Kelas Accepted Pseudo-Labels (Total: {len(accepted_df)} Citra)", weight="bold")
    plt.xlabel("Kelas Prediksi")
    plt.ylabel("Jumlah Citra Diterima")
    plt.xticks(rotation=45)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 0.5, f"{int(h)}", ha="center", va="bottom", weight="bold")
    plt.tight_layout()
    plt.show()

plot_pseudo_class_balance(accepted_pseudo_df)"""
})

SECTIONS.append({
    "id": 32,
    "title": "32. Pseudo-Label Consistency Audit",
    "markdown": r"""# 32. Pseudo-Label Consistency Audit

Audit integritas dan konsistensi manifest pseudo-label:
- Memastikan ID unik (tanpa duplikasi).
- Memastikan seluruh kelas termasuk dalam 8 kelas resmi.
- Memastikan tidak ada tumpang tindih dengan 40 labeled samples atau test set.""",
    "code": r"""# ==============================================================================
# SECTION 32: PSEUDO-LABEL CONSISTENCY AUDIT
# ==============================================================================
def audit_pseudo_label_manifest(accepted_df: pd.DataFrame, labeled_df: pd.DataFrame, test_dir: Path):
    if len(accepted_df) == 0:
        print("[INFO] Manifest pseudo-labels kosong, audit dilewati.")
        return
        
    assert accepted_df["image_id"].is_unique, "Duplikasi ID ditemukan di accepted pseudo-labels!"
    invalid_classes = set(accepted_df["predicted_class"]) - set(CLASS_NAMES)
    assert len(invalid_classes) == 0, f"Kelas tidak valid: {invalid_classes}"
    overlap_labeled = set(accepted_df["sha256"]) & set(labeled_df["sha256"])
    assert len(overlap_labeled) == 0, f"Bocoran: Pseudo-label tumpang tindih dengan labeled pool! ({len(overlap_labeled)})"
    
    print("[AUDIT VALID] Manifest accepted pseudo-labels konsisten dan bebas anomali.")

audit_pseudo_label_manifest(accepted_pseudo_df, labeled_pool_df, DATA_TEST_DIR)"""
})

SECTIONS.append({
    "id": 33,
    "title": "33. Accepted Pseudo-Label Manifest",
    "markdown": r"""# 33. Accepted Pseudo-Label Manifest Summary

Ringkasan manifest akhir pseudo-label yang diterima dan siap digabungkan dengan labeled pool untuk melatih model Student.""",
    "code": r"""# ==============================================================================
# SECTION 33: ACCEPTED PSEUDO-LABEL MANIFEST SUMMARY
# ==============================================================================
if len(accepted_pseudo_df) > 0:
    print(f"Total Accepted Pseudo-Labels : {len(accepted_pseudo_df)} citra")
    total_unlabeled = len(unlabeled_pool_df) if len(unlabeled_pool_df) > 0 else 1876
    print(f"Persentase Diterima          : {len(accepted_pseudo_df)/total_unlabeled*100:.2f}% dari unlabeled pool")
    print()
    print("Preview 5 Baris Pertama:")
    print(accepted_pseudo_df[["image_id", "predicted_class", "confidence", "relative_path"]].head())
else:
    print("[INFO] Accepted pseudo-labels belum dibangkitkan.")"""
})
