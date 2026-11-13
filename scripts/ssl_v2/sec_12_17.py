# scripts/ssl_v2/sec_12_17.py
SECTIONS = []

SECTIONS.append({
    "id": 12,
    "title": "12. Unlabeled Feature Extraction & Multi-View Augmentation",
    "markdown": r"""# 12. Unlabeled Feature Extraction & Multi-View Augmentation

Untuk mengevaluasi 1.877 citra unlabeled tanpa kebocoran dan tanpa membebani GPU:
1. Mengekstrak representasi fitur untuk **dua sudut pandang (*two-view*)**:
   - **View 1 (Original)**: Transformasi evaluasi standar.
   - **View 2 (Weak Flip)**: Transformasi dengan horizontal flip.
2. Meng-cache representasi kedua pandangan ke disk (`unlabeled_embeddings_v2.pt`).
3. Konsistensi prediksi antara kedua pandangan ini akan menjadi penyaring utama keabsahan pseudo-label.""",
    "code": r"""# ==============================================================================
# SECTION 12: UNLABELED FEATURE EXTRACTION (TWO-VIEW)
# ==============================================================================
class TwoViewUnlabeledDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root_dir: Path, transform_v1=None, transform_v2=None):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform_v1 = transform_v1
        self.transform_v2 = transform_v2
        
    def __len__(self) -> int:
        return len(self.df)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        row = self.df.iloc[idx]
        img_id = str(row["image_id"])
        img_path = self.root_dir / row["relative_path"]
        with Image.open(img_path) as img:
            img_rgb = img.convert("RGB")
            
        t1 = self.transform_v1(img_rgb) if self.transform_v1 else img_rgb
        t2 = self.transform_v2(img_rgb) if self.transform_v2 else img_rgb
        return t1, t2, img_id

UNLABELED_CACHE_PATH = MANIFEST_DIR / "unlabeled_embeddings_v2.pt"

ds_unlabeled_views = TwoViewUnlabeledDataset(
    unlabeled_df, PROJECT_ROOT,
    transform_v1=eval_transforms,
    transform_v2=weak_flip_transforms
)
loader_unlabeled_views = DataLoader(ds_unlabeled_views, batch_size=16, shuffle=False, num_workers=0)

if UNLABELED_CACHE_PATH.exists():
    cache_unlab = load_project_checkpoint(UNLABELED_CACHE_PATH, device="cpu")
    raw_v1 = cache_unlab["feats_v1"]
    unlabeled_feats_v1 = raw_v1.numpy() if isinstance(raw_v1, torch.Tensor) else np.array(raw_v1)
    raw_v2 = cache_unlab["feats_v2"]
    unlabeled_feats_v2 = raw_v2.numpy() if isinstance(raw_v2, torch.Tensor) else np.array(raw_v2)
    unlabeled_image_ids = list(cache_unlab["image_ids"])
    print(f"[LOADED CACHE] Fitur 1877 unlabeled termuat dari {UNLABELED_CACHE_PATH.name}")
else:
    print(f"Mengekstrak fitur multi-view pada {len(ds_unlabeled_views)} citra unlabeled...")
    base_teacher_model.eval()
    feats_v1_list, feats_v2_list, ids_list = [], [], []
    
    with torch.inference_mode():
        for img_v1, img_v2, ids in tqdm(loader_unlabeled_views, desc="Two-View Extraction"):
            img_v1, img_v2 = img_v1.to(DEVICE), img_v2.to(DEVICE)
            
            f1 = torch.flatten(base_teacher_model.avgpool(base_teacher_model.features(img_v1)), 1)
            f2 = torch.flatten(base_teacher_model.avgpool(base_teacher_model.features(img_v2)), 1)
            
            f1_norm = F.normalize(f1, p=2, dim=1).cpu().numpy()
            f2_norm = F.normalize(f2, p=2, dim=1).cpu().numpy()
            
            feats_v1_list.append(f1_norm)
            feats_v2_list.append(f2_norm)
            ids_list.extend([str(i) for i in ids])
            
    unlabeled_feats_v1 = np.vstack(feats_v1_list)
    unlabeled_feats_v2 = np.vstack(feats_v2_list)
    unlabeled_image_ids = ids_list
    
    torch.save({
        "feats_v1": torch.from_numpy(unlabeled_feats_v1).float(),
        "feats_v2": torch.from_numpy(unlabeled_feats_v2).float(),
        "image_ids": list(unlabeled_image_ids)
    }, UNLABELED_CACHE_PATH)
    print(f"[SAVED CACHE] Fitur multi-view 1877 citra disimpan ke {UNLABELED_CACHE_PATH.name}")

print(f"Fitur View 1: Shape {unlabeled_feats_v1.shape} | Fitur View 2: Shape {unlabeled_feats_v2.shape}")"""
})

SECTIONS.append({
    "id": 13,
    "title": "13. Pseudo-Label Generation & Confidence Distribution Analysis",
    "markdown": r"""# 13. Pseudo-Label Generation & Confidence Distribution Analysis

Menganalisis probabilitas prediksi Teacher pada 1.877 citra unlabeled:
1. Menghitung distribusi nilai confidence skor ($c = \max p_i$).
2. Mengevaluasi kuantil sebaran (persentil ke-50, 75, 90, 95).
3. Menyimpan seluruh prediksi mentah ke `data/manifests/fewshot_ssl_v2/pseudo_labels_v2.csv`.""",
    "code": r"""# ==============================================================================
# SECTION 13: PSEUDO-LABEL GENERATION & CONFIDENCE ANALYSIS
# ==============================================================================
def predict_features_with_head(head_model: nn.Module, feats_np: np.ndarray, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    head_model.eval()
    head_model.to(device)
    feats_tensor = torch.from_numpy(feats_np).to(device)
    with torch.no_grad():
        logits = head_model(feats_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
    return probs, preds

probs_v1, preds_v1 = predict_features_with_head(final_teacher_head, unlabeled_feats_v1, DEVICE)
probs_v2, preds_v2 = predict_features_with_head(final_teacher_head, unlabeled_feats_v2, DEVICE)

conf_v1 = np.max(probs_v1, axis=1)
conf_v2 = np.max(probs_v2, axis=1)
consistency_mask = (preds_v1 == preds_v2)

# Susun DataFrame pseudo-label mentah
raw_pseudo_records = []
for i, img_id in enumerate(unlabeled_image_ids):
    raw_pseudo_records.append({
        "image_id": str(img_id),
        "predicted_idx": int(preds_v1[i]),
        "predicted_class": IDX_TO_CLASS[int(preds_v1[i])],
        "confidence_v1": float(conf_v1[i]),
        "confidence_v2": float(conf_v2[i]),
        "is_consistent": bool(consistency_mask[i])
    })

raw_pseudo_df = pd.DataFrame(raw_pseudo_records)
merged_pseudo_df = pd.merge(unlabeled_df[["image_id", "relative_path", "sha256"]], raw_pseudo_df, on="image_id")
RAW_PSEUDO_PATH = MANIFEST_DIR / "pseudo_labels_v2.csv"
merged_pseudo_df.to_csv(RAW_PSEUDO_PATH, index=False)

print("ANALISIS SEBARAN CONFIDENCE PSEUDO-LABEL:")
print(f"  - Total Citra Unlabeled : {len(raw_pseudo_df)}")
print(f"  - Nilai Min Confidence  : {conf_v1.min():.4f}")
print(f"  - Nilai Max Confidence  : {conf_v1.max():.4f}")
print(f"  - Rata-Rata Confidence  : {conf_v1.mean():.4f}")
print(f"  - Persentil ke-50 (Med) : {np.percentile(conf_v1, 50):.4f}")
print(f"  - Persentil ke-75       : {np.percentile(conf_v1, 75):.4f}")
print(f"  - Persentil ke-90       : {np.percentile(conf_v1, 90):.4f}")
print(f"  - Persentil ke-95       : {np.percentile(conf_v1, 95):.4f}")
print(f"  - Konsistensi 2-View    : {consistency_mask.sum()} / {len(consistency_mask)} ({consistency_mask.mean()*100:.2f}%)")

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(conf_v1, bins=30, color="teal", edgecolor="black", alpha=0.8)
ax.axvline(CONFIG["pseudo_min_confidence"], color="red", linestyle="--", linewidth=2, label=f"Threshold Floor ({CONFIG['pseudo_min_confidence']})")
ax.set_title("Distribusi Confidence Skor Unlabeled Citra (View 1)", weight="bold")
ax.set_xlabel("Confidence Score")
ax.set_ylabel("Frekuensi")
ax.legend()
plt.tight_layout()
fig.savefig(OUTPUT_FIGURES_DIR / "fewshot_ssl_v2_confidence_dist.png", dpi=150)
plt.close(fig)"""
})

SECTIONS.append({
    "id": 14,
    "title": "14. Multi-View Consistency & Class-Balanced Top-K Filtering",
    "markdown": r"""# 14. Multi-View Consistency & Class-Balanced Top-K Filtering

Strategi penyaringan pseudo-label yang aman dan seimbang:
1. **Filter 1 (Augmentation Consistency)**: Hanya menerima citra di mana prediksi pada View 1 identik dengan View 2 (`pred_v1 == pred_v2`).
2. **Filter 2 (Confidence Floor)**: Nilai confidence harus $\ge 0.45$ (jauh melampaui tebakan acak 8 kelas = $0.125$).
3. **Filter 3 (Class-Balanced Top-K)**: Mengambil paling banyak $K=15$ citra dengan confidence tertinggi per kelas untuk mencegah ketimpangan distribusi kelas.
4. Hasil disimpan ke `data/manifests/fewshot_ssl_v2/accepted_pseudo_labels_v2.csv`.""",
    "code": r"""# ==============================================================================
# SECTION 14: MULTI-VIEW & BALANCED TOP-K FILTERING
# ==============================================================================
min_conf = CONFIG["pseudo_min_confidence"]
top_k = CONFIG["pseudo_top_k_per_class"]

# 1. Terapkan filter konsistensi dan batas bawah confidence
eligible_df = merged_pseudo_df[
    (merged_pseudo_df["is_consistent"]) &
    (merged_pseudo_df["confidence_v1"] >= min_conf)
].copy()

# 2. Ambil Top-K citra teratas per kelas
accepted_dfs = []
for c_name in CLASS_NAMES:
    cls_subset = eligible_df[eligible_df["predicted_class"] == c_name]
    top_subset = cls_subset.sort_values(by="confidence_v1", ascending=False).head(top_k)
    accepted_dfs.append(top_subset)

if accepted_dfs:
    accepted_pseudo_df = pd.concat(accepted_dfs, ignore_index=True)
else:
    accepted_pseudo_df = pd.DataFrame()

ACCEPTED_PSEUDO_PATH = MANIFEST_DIR / "accepted_pseudo_labels_v2.csv"
accepted_pseudo_df.to_csv(ACCEPTED_PSEUDO_PATH, index=False)

HAS_ACCEPTED_PSEUDO = len(accepted_pseudo_df) > 0 and len(accepted_pseudo_df["predicted_class"].unique()) >= 4

print("=" * 70)
print(f"HASIL PENYARINGAN PSEUDO-LABEL TASK 09:")
print(f"  - Total Lolos Filter Konsistensi & Conf >= {min_conf}: {len(eligible_df)}")
print(f"  - Total Diterima (Balanced Top-{top_k}/Kelas)       : {len(accepted_pseudo_df)}")
print(f"  - Status Student Training Diizinkan                : {HAS_ACCEPTED_PSEUDO}")
print("=" * 70)

if HAS_ACCEPTED_PSEUDO:
    print("\nSebaran Pseudo-Label Diterima Tiap Kelas:")
    print(accepted_pseudo_df["predicted_class"].value_counts())
else:
    print("[INFO] Tidak cukup pseudo-label berkualitas tinggi yang lolos filter.")
    print("       Pipeline akan mempertahankan model Teacher sebagai seleksi final.")"""
})

SECTIONS.append({
    "id": 15,
    "title": "15. Student Dataset Construction & SemiSupervisedLoss",
    "markdown": r"""# 15. Student Dataset Construction & SemiSupervisedLoss

Membangun dataset gabungan dan fungsi loss semi-supervised untuk pelatihan Student:
- **Sampel Labeled Manual**: 40 citra, bobot sampel $w_{\text{sup}} = 1.0$.
- **Sampel Pseudo-Labeled**: $M$ citra terverifikasi, bobot sampel $w_{\text{pseudo}} = \lambda = 0.35$.
- **Semi-Supervised Loss**:
  $$\mathcal{L} = \mathcal{L}_{\text{sup}} + \lambda_{\text{pseudo}} \mathcal{L}_{\text{pseudo}}$$""",
    "code": r"""# ==============================================================================
# SECTION 15: STUDENT DATASET & LOSS FORMULATION
# ==============================================================================
class CombinedStudentDataset(Dataset):
    def __init__(self, labeled_df: pd.DataFrame, pseudo_df: pd.DataFrame, root_dir: Path, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        # 1. Masukkan sampel berlabel manual
        for _, row in labeled_df.iterrows():
            self.samples.append({
                "relative_path": row["relative_path"],
                "target": int(row["class_idx"]),
                "is_pseudo": False,
                "weight": 1.0,
                "image_id": str(row["image_id"])
            })
            
        # 2. Masukkan sampel pseudo-labeled terverifikasi
        if len(pseudo_df) > 0:
            for _, row in pseudo_df.iterrows():
                self.samples.append({
                    "relative_path": row["relative_path"],
                    "target": int(row["predicted_idx"]),
                    "is_pseudo": True,
                    "weight": float(CONFIG["student_pseudo_weight"]),
                    "image_id": str(row["image_id"])
                })
                
    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, bool, float, str]:
        item = self.samples[idx]
        img_path = self.root_dir / item["relative_path"]
        with Image.open(img_path) as img:
            image = img.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, item["target"], item["is_pseudo"], item["weight"], item["image_id"]

class SemiSupervisedLoss(nn.Module):
    def __init__(self, pseudo_weight: float = 0.35):
        super().__init__()
        self.pseudo_weight = pseudo_weight
        self.ce = nn.CrossEntropyLoss(reduction="none")
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, is_pseudo: torch.Tensor, sample_weights: torch.Tensor):
        losses = self.ce(logits, targets)
        sup_mask = ~is_pseudo
        pseudo_mask = is_pseudo
        
        sup_loss = (losses[sup_mask] * sample_weights[sup_mask]).mean() if sup_mask.sum() > 0 else torch.tensor(0.0, device=logits.device)
        pseudo_loss = (losses[pseudo_mask] * sample_weights[pseudo_mask]).mean() if pseudo_mask.sum() > 0 else torch.tensor(0.0, device=logits.device)
        total_loss = sup_loss + self.pseudo_weight * pseudo_loss
        return total_loss, sup_loss, pseudo_loss

student_ds = CombinedStudentDataset(labeled_df, accepted_pseudo_df, PROJECT_ROOT, transform=train_transforms)
student_loader = DataLoader(student_ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=0)

print(f"[STUDENT DATASET SIAP] Total Sampel Pelatihan: {len(student_ds)}")
print(f"  - Labeled Manual: {len(labeled_df)}")
print(f"  - Pseudo-Labeled: {len(accepted_pseudo_df)}")"""
})

SECTIONS.append({
    "id": 16,
    "title": "16. Student Model Training with Proper BatchNorm Freezing",
    "markdown": r"""# 16. Student Model Training with Proper BatchNorm Freezing

Melatih model Student menggunakan bobot ImageNet-1K dan backbone dengan BatchNorm yang dibekukan:
- Mencegah pergeseran statistik running_mean / running_var pada batch kecil.
- Mengoptimalkan parameter head classifier dan blok representasi terakhir.
- Menyimpan model terbaik ke `models/checkpoints/fewshot_ssl_v2/student/best.pt`.""",
    "code": r"""# ==============================================================================
# SECTION 16: STUDENT MODEL TRAINING
# ==============================================================================
STUDENT_BEST_CKPT = STUDENT_CKPT_DIR / "best.pt"
STUDENT_METRICS_PATH = OUTPUT_METRICS_DIR / "fewshot_ssl_v2_student.json"

def train_student_network(loader: DataLoader, config: Dict[str, Any], device: torch.device):
    model = build_efficientnet_b2_backbone(num_classes=NUM_CLASSES, pretrained=True).to(device)
    freeze_backbone_with_eval_bn(model)
    
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["student_lr"],
        weight_decay=config["student_weight_decay"]
    )
    criterion = SemiSupervisedLoss(pseudo_weight=config["student_pseudo_weight"])
    scaler = torch.cuda.amp.GradScaler(enabled=config["use_amp"])
    
    history = []
    print(f"Melatih Student Model ({config['student_epochs']} epochs)...")
    
    for epoch in range(config["student_epochs"]):
        model.eval()
        model.classifier.train()
        
        tot_loss_acc, sup_loss_acc, p_loss_acc = 0.0, 0.0, 0.0
        correct, total = 0, 0
        
        for images, targets, is_pseudo, weights, _ in loader:
            images = images.to(device)
            targets = targets.to(device)
            is_pseudo = is_pseudo.to(device)
            weights = weights.to(device)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=config["use_amp"]):
                logits = model(images)
                loss, sup_l, p_l = criterion(logits, targets, is_pseudo, weights)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            bs = images.size(0)
            tot_loss_acc += loss.item() * bs
            sup_loss_acc += sup_l.item() * bs
            p_loss_acc += p_l.item() * bs
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += bs
            
        ep_loss = tot_loss_acc / total
        ep_acc = correct / total
        rec = {
            "epoch": epoch + 1, "total_loss": ep_loss,
            "sup_loss": sup_loss_acc / total,
            "pseudo_loss": p_loss_acc / total,
            "train_acc": ep_acc
        }
        history.append(rec)
        if (epoch + 1) % 5 == 0 or epoch == config["student_epochs"] - 1:
            print(f"  Student Ep {epoch+1:02d}/{config['student_epochs']} | Loss: {ep_loss:.4f} | Train Acc: {ep_acc*100:.2f}%")
            
    torch.save({
        "model_state_dict": model.state_dict(),
        "final_loss": ep_loss,
        "final_acc": ep_acc,
        "class_names": CLASS_NAMES,
        "config": config
    }, STUDENT_BEST_CKPT)
    
    return {"final_loss": ep_loss, "final_acc": ep_acc, "history": history}

if HAS_ACCEPTED_PSEUDO:
    student_results = train_student_network(student_loader, CONFIG, DEVICE)
    with open(STUDENT_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(student_results, f, indent=2)
    print(f"[STUDENT SUCCESS] Model tersimpan di {STUDENT_BEST_CKPT.name}")
else:
    print("[STUDENT SKIPPED] Pelatihan dilewati karena pseudo-label tidak memenuhi syarat kualitas.")
    student_results = {}"""
})

SECTIONS.append({
    "id": 17,
    "title": "17. Student vs Teacher Evaluation & Final Model Selection",
    "markdown": r"""# 17. Student vs Teacher Evaluation & Final Model Selection

Memilih model akhir yang paling optimal untuk inferensi test kompetisi:
- Jika model Student berhasil dilatih dan divalidasi, Student menjadi kandidat utama.
- Jika pseudo-label tidak memadai atau Student belum tersedia, **Final Teacher** (model terpilih pada Section 10 & 11) menjadi model inferensi definitif.
- Menjamin tidak ada pemilihan model acak atau dependensi palsu.""",
    "code": r"""# ==============================================================================
# SECTION 17: FINAL MODEL SELECTION FOR TEST INFERENCE
# ==============================================================================
def select_final_inference_model() -> Tuple[str, Path]:
    student_ckpt = STUDENT_CKPT_DIR / "best.pt" if "STUDENT_CKPT_DIR" in globals() else PROJECT_ROOT / "models" / "checkpoints" / "fewshot_ssl_v2" / "student" / "best.pt"
    teacher_ckpt = TEACHER_CKPT_DIR / "best.pt" if "TEACHER_CKPT_DIR" in globals() else PROJECT_ROOT / "models" / "checkpoints" / "fewshot_ssl_v2" / "teacher" / "best.pt"
    has_pseudo = globals().get("HAS_ACCEPTED_PSEUDO", False)
    
    if student_ckpt.exists() and has_pseudo:
        return "Student (Semi-Supervised Few-Shot)", student_ckpt
    elif teacher_ckpt.exists():
        try:
            t_payload = load_project_checkpoint(teacher_ckpt, device="cpu")
            model_name = t_payload.get("model_name", globals().get("BEST_SUPERVISED_MODEL_NAME", "Supervised"))
        except Exception:
            model_name = globals().get("BEST_SUPERVISED_MODEL_NAME", "Supervised")
        return f"Teacher ({model_name})", teacher_ckpt
    else:
        raise FileNotFoundError("Tidak ditemukan checkpoint Teacher maupun Student!")

SELECTED_FINAL_MODEL_TYPE, SELECTED_FINAL_MODEL_PATH = select_final_inference_model()
print("=" * 70)
print(f"MODEL FINAL TERPILIH UNTUK INFERENSI KAGGLE:")
print(f"  - Tipe Model : {SELECTED_FINAL_MODEL_TYPE}")
print(f"  - Checkpoint : {SELECTED_FINAL_MODEL_PATH}")
print("=" * 70)"""
})
