# scripts/sec_34_41.py
SECTIONS = []

SECTIONS.append({
    "id": 34,
    "title": "34. Student Model Initialization",
    "markdown": """# 34. Student Model Initialization

Inisialisasi model **Student**:
- Arsitektur: `EfficientNet-B2` pretrained ImageNet-1K (`weights=EfficientNet_B2_Weights.IMAGENET1K_V1`).
- Penggantian classifier head menjadi 8 kelas dengan mempertahankan Dropout (0.3).
- Model Student dilatih pada gabungan data: **40 citra berlabel manual** + **citra pseudo-label terpilih**.""",
    "code": """# ==============================================================================
# SECTION 34: STUDENT MODEL INITIALIZATION
# ==============================================================================
def build_student_model(
    num_classes: int = 8,
    pretrained: bool = True,
    freeze_backbone: bool = True
) -> nn.Module:
    weights = EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b2(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
            
    return model

student_model = build_student_model(num_classes=NUM_CLASSES, pretrained=True, freeze_backbone=True)
student_model = student_model.to(DEVICE)
print("[STUDENT INITIALIZED] Model Student EfficientNet-B2 siap dikonfigurasi.")"""
})

SECTIONS.append({
    "id": 35,
    "title": "35. Student Dataset Construction",
    "markdown": """# 35. Student Dataset Construction (Combined Semi-Supervised)

Konstruksi kelas Dataset gabungan untuk Student (`CombinedStudentDataset`):
- Memadukan sampel **Manual Supervised** (bobot sampel = 1.0) dan sampel **Accepted Pseudo-Label** (bobot sampel = $\lambda_{pseudo}$).
- Mengembalikan: `(image_tensor, target_class_idx, is_pseudo_flag, sample_weight, image_id)`.""",
    "code": """# ==============================================================================
# SECTION 35: STUDENT DATASET CONSTRUCTION
# ==============================================================================
class CombinedStudentDataset(Dataset):
    def __init__(
        self,
        labeled_df: pd.DataFrame,
        pseudo_df: pd.DataFrame,
        project_root: Path,
        transform: Optional[Callable] = None,
        pseudo_weight: float = 0.5
    ):
        self.project_root = project_root
        self.transform = transform
        self.records = []
        
        # 1. Masukkan sampel berlabel asli
        for _, row in labeled_df.iterrows():
            self.records.append({
                "relative_path": row["relative_path"],
                "target_idx": int(row["class_idx"]),
                "is_pseudo": False,
                "weight": 1.0,
                "image_id": str(row["image_id"])
            })
            
        # 2. Masukkan sampel pseudo-labeled
        for _, row in pseudo_df.iterrows():
            self.records.append({
                "relative_path": row["relative_path"],
                "target_idx": int(row["predicted_idx"]),
                "is_pseudo": True,
                "weight": float(pseudo_weight),
                "image_id": str(row["image_id"])
            })
            
    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, bool, float, str]:
        item = self.records[idx]
        img_path = self.project_root / item["relative_path"]
        with Image.open(img_path) as img:
            image = img.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
            
        return image, item["target_idx"], item["is_pseudo"], item["weight"], item["image_id"]

print("[DATASET STUDENT TERDEFINISI] CombinedStudentDataset siap memadukan data berlabel dan pseudo-label.")"""
})

SECTIONS.append({
    "id": 36,
    "title": "36. Supervised and Pseudo-Labeled Loss",
    "markdown": """# 36. Supervised and Pseudo-Labeled Joint Loss

Formulasi fungsi loss semi-supervised terbobot:
$$\mathcal{L}_{total} = \frac{1}{N_{sup}} \sum_{i \in sup} \text{CE}(y_i, \hat{y}_i) + \lambda_{pseudo} \cdot \frac{1}{N_{pseudo}} \sum_{j \in pseudo} \text{CE}(\tilde{y}_j, \hat{y}_j)$$
di mana $\lambda_{pseudo} = 0.5$ secara default untuk mencegah noise pseudo-label mendominasi gradient.""",
    "code": """# ==============================================================================
# SECTION 36: SUPERVISED AND PSEUDO-LABELED LOSS
# ==============================================================================
class SemiSupervisedLoss(nn.Module):
    def __init__(self, pseudo_weight: float = 0.5):
        super().__init__()
        self.pseudo_weight = pseudo_weight
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        is_pseudo: torch.Tensor,
        sample_weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        losses = self.ce(logits, targets)
        
        sup_mask = ~is_pseudo
        pseudo_mask = is_pseudo
        
        if sup_mask.sum() > 0:
            sup_loss = (losses[sup_mask] * sample_weights[sup_mask]).mean()
        else:
            sup_loss = torch.tensor(0.0, device=logits.device)
            
        if pseudo_mask.sum() > 0:
            pseudo_loss = (losses[pseudo_mask] * sample_weights[pseudo_mask]).mean()
        else:
            pseudo_loss = torch.tensor(0.0, device=logits.device)
            
        total_loss = sup_loss + (self.pseudo_weight * pseudo_loss)
        return total_loss, sup_loss, pseudo_loss

semi_sup_criterion = SemiSupervisedLoss(pseudo_weight=0.5)
print("[LOSS TERDEFINISI] SemiSupervisedLoss dengan bobot pseudo = 0.5 siap digunakan.")"""
})

SECTIONS.append({
    "id": 37,
    "title": "37. Student Training Configuration",
    "markdown": """# 37. Student Training Configuration

Konfigurasi pelatihan Student Model:
- Optimizer: `AdamW` (learning rate = 5e-4, weight decay = 1e-4)
- Batch Size: 8
- Maksimum Epoch: 25
- Pseudo Loss Weight: 0.5
- Augmentasi: Transformasi pelatihan terkendali (Section 13)""",
    "code": """# ==============================================================================
# SECTION 37: STUDENT TRAINING CONFIGURATION
# ==============================================================================
STUDENT_CONFIG = {
    "experiment_id": "fewshot_pseudolabel_b2_student",
    "batch_size": 8,
    "learning_rate": 5e-4,
    "weight_decay": 1e-4,
    "epochs": 25,
    "pseudo_loss_weight": 0.5,
    "optimizer": "AdamW"
}

print("KONFIGURASI STUDENT TRAINING:")
for k, v in STUDENT_CONFIG.items():
    print(f"  {k:22}: {v}")"""
})

SECTIONS.append({
    "id": 38,
    "title": "38. Manual Student Training",
    "markdown": """# 38. Manual Student Training Execution

Pelatihan model Student pada dataset semi-supervised gabungan:
- Dilindungi oleh flag `RUN_STUDENT_TRAINING`.
- Memantau running accuracy, supervised loss, dan pseudo loss per batch menggunakan `tqdm.auto`.
- Menyimpan checkpoint Student ke `models/checkpoints/fewshot_pseudolabel_b2_v1/student/best.pt`.
- Menyimpan riwayat metrik ke `outputs/metrics/fewshot_pseudolabel_b2_student.json`.""",
    "code": """# ==============================================================================
# SECTION 38: MANUAL STUDENT TRAINING EXECUTION
# ==============================================================================
STUDENT_BEST_CKPT = STUDENT_CHECKPOINT_DIR / "best.pt"
STUDENT_METRICS_PATH = OUTPUT_METRICS_DIR / "fewshot_pseudolabel_b2_student.json"

def train_student_model(
    labeled_df: pd.DataFrame,
    pseudo_df: pd.DataFrame,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    dataset = CombinedStudentDataset(labeled_df, pseudo_df, PROJECT_ROOT, transform=train_transforms, pseudo_weight=config["pseudo_loss_weight"])
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0)
    
    model = build_student_model(num_classes=NUM_CLASSES, pretrained=True, freeze_backbone=True).to(DEVICE)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    criterion = SemiSupervisedLoss(pseudo_weight=config["pseudo_loss_weight"])
    
    print(f"Melatih Model Student pada {len(dataset)} citra gabungan ({len(labeled_df)} manual + {len(pseudo_df)} pseudo)...")
    history = []
    
    for epoch in range(config["epochs"]):
        model.train()
        total_loss_accum = 0.0
        sup_loss_accum = 0.0
        pseudo_loss_accum = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(loader, desc=f"Student Ep {epoch+1}/{config['epochs']}", leave=False)
        for images, targets, is_pseudo, weights, _ in pbar:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)
            is_pseudo = is_pseudo.to(DEVICE)
            weights = weights.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(images)
            loss, sup_l, p_l = criterion(logits, targets, is_pseudo, weights)
            loss.backward()
            optimizer.step()
            
            bs = images.size(0)
            total_loss_accum += loss.item() * bs
            sup_loss_accum += sup_l.item() * bs
            pseudo_loss_accum += p_l.item() * bs
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += bs
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct/total:.3f}"})
            
        ep_loss = total_loss_accum / total
        ep_sup_loss = sup_loss_accum / total
        ep_p_loss = pseudo_loss_accum / total
        ep_acc = correct / total
        
        rec = {
            "epoch": epoch + 1, "total_loss": ep_loss, "sup_loss": ep_sup_loss,
            "pseudo_loss": ep_p_loss, "accuracy": ep_acc
        }
        history.append(rec)
        print(f"  Ep {epoch+1:02d} | Loss: {ep_loss:.4f} (Sup: {ep_sup_loss:.4f}, Pseudo: {ep_p_loss:.4f}) | Acc: {ep_acc*100:.2f}%")
        
    torch.save({
        "epoch": config["epochs"],
        "model_state_dict": model.state_dict(),
        "final_loss": ep_loss,
        "final_acc": ep_acc,
        "class_names": CLASS_NAMES,
        "config": config,
        "num_labeled": len(labeled_df),
        "num_pseudo": len(pseudo_df)
    }, STUDENT_BEST_CKPT)
    
    return {"final_loss": ep_loss, "final_acc": ep_acc, "history": history}

if STUDENT_BEST_CKPT.exists() and STUDENT_METRICS_PATH.exists():
    with open(STUDENT_METRICS_PATH, "r", encoding="utf-8") as f:
        student_results = json.load(f)
    print(f"[LOADED] Checkpoint Student tersedia: {STUDENT_BEST_CKPT}")
elif RUN_STUDENT_TRAINING and len(labeled_pool_df) > 0 and len(accepted_pseudo_df) > 0:
    student_results = train_student_model(labeled_pool_df, accepted_pseudo_df, STUDENT_CONFIG)
    with open(STUDENT_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(student_results, f, indent=2)
    print(f"[SAVED] Checkpoint Student berhasil disimpan ke {STUDENT_BEST_CKPT}")
else:
    print("[STATUS] RUN_STUDENT_TRAINING is False atau data pseudo belum siap.")
    student_results = {}"""
})

SECTIONS.append({
    "id": 39,
    "title": "39. Student Training History",
    "markdown": """# 39. Student Training History Summary

Tabel riwayat metrik pelatihan Student Model (Total Loss, Supervised Loss, Pseudo Loss, dan Running Accuracy) per epoch.""",
    "code": """# ==============================================================================
# SECTION 39: STUDENT TRAINING HISTORY SUMMARY
# ==============================================================================
if student_results and "history" in student_results:
    df_student_hist = pd.DataFrame(student_results["history"])
    print("RIWAYAT PELATIHAN STUDENT MODEL:")
    print(df_student_hist.to_string(index=False))
else:
    print("[INFO] Riwayat pelatihan Student belum tersedia.")"""
})

SECTIONS.append({
    "id": 40,
    "title": "40. Student Learning Curves",
    "markdown": """# 40. Student Learning Curves

Visualisasi kurva pembelajaran Student:
- Dekomposisi Total Loss, Supervised Loss, dan Pseudo Loss.
- Kurva akurasi pelatihan.
- Gambar disimpan ke `outputs/figures/fewshot_pseudolabel_b2_student_curves.png`.""",
    "code": """# ==============================================================================
# SECTION 40: STUDENT LEARNING CURVES
# ==============================================================================
STUDENT_CURVES_PATH = OUTPUT_FIGURES_DIR / "fewshot_pseudolabel_b2_student_curves.png"

def plot_student_curves(results: Dict[str, Any]):
    if not results or "history" not in results:
        print("[INFO] Kurva student belum tersedia.")
        return
        
    hist = results["history"]
    epochs = [h["epoch"] for h in hist]
    total_l = [h["total_loss"] for h in hist]
    sup_l = [h["sup_loss"] for h in hist]
    p_l = [h["pseudo_loss"] for h in hist]
    acc = [h["accuracy"] for h in hist]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(epochs, total_l, "o-", color="purple", label="Total Loss")
    ax1.plot(epochs, sup_l, "--", color="blue", label="Supervised Loss")
    ax1.plot(epochs, p_l, ":", color="orange", label="Pseudo Loss")
    ax1.set_title("Student Semi-Supervised Loss", weight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    ax2.plot(epochs, acc, "s-", color="forestgreen", label="Training Accuracy")
    ax2.set_title("Student Training Accuracy", weight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(STUDENT_CURVES_PATH, dpi=150)
    plt.show()
    print(f"[FIGURE SAVED] Kurva disimpan ke {STUDENT_CURVES_PATH}")

plot_student_curves(student_results)"""
})

SECTIONS.append({
    "id": 41,
    "title": "41. Student Evaluation",
    "markdown": """# 41. Student Evaluation (Internal Labeled Fit)

Evaluasi kecocokan internal model Student pada 40 sampel manual berlabel:
- Memeriksa apakah penambahan data pseudo-labeled tetap menjaga atau meningkatkan kemampuan klasifikasi pada gold-standard 40 sampel berlabel manual.""",
    "code": """# ==============================================================================
# SECTION 41: STUDENT EVALUATION
# ==============================================================================
def evaluate_student_on_labeled(model_ckpt: Path, labeled_df: pd.DataFrame, device: torch.device):
    if not model_ckpt.exists() or len(labeled_df) == 0:
        print("[INFO] Model Student atau Labeled Pool belum tersedia.")
        return
        
    model = build_student_model(num_classes=NUM_CLASSES, pretrained=False, freeze_backbone=True)
    payload = torch.load(model_ckpt, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    
    dataset = FewShotLabeledDataset(labeled_df, PROJECT_ROOT, transform=eval_transforms)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    all_targets = []
    all_preds = []
    with torch.inference_mode():
        for images, targets, _ in loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            all_targets.extend(targets.numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            
    eval_met = calculate_macro_metrics(all_targets, all_preds, NUM_CLASSES)
    print("HASIL EVALUASI STUDENT PADA 40 CITRA LABELED POOL:")
    print(f"  Akurasi  : {eval_met['accuracy']*100:.2f}%")
    print(f"  Macro F1 : {eval_met['macro_f1']*100:.2f}%")

if STUDENT_BEST_CKPT.exists():
    evaluate_student_on_labeled(STUDENT_BEST_CKPT, labeled_pool_df, DEVICE)"""
})
