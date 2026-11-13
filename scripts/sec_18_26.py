# scripts/sec_18_26.py
SECTIONS = []

SECTIONS.append({
    "id": 18,
    "title": "18. Teacher Model Initialization",
    "markdown": r"""# 18. Teacher Model Initialization (EfficientNet-B2 Pretrained)

Inisialisasi model Teacher menggunakan arsitektur `EfficientNet-B2` dengan bobot resmi ImageNet-1K:
- Memuat bobot resmi `EfficientNet_B2_Weights.IMAGENET1K_V1`.
- Mempertahankan layer Dropout (0.3) bawaan torchvision.
- Mengganti layer proyeksi akhir linear dari 1408 fitur menjadi 8 kelas target.
- Membekukan seluruh layer fitur backbone (`features.parameters()`) agar pelatihan terfokus pada classifier head.""",
    "code": r"""# ==============================================================================
# SECTION 18: TEACHER MODEL INITIALIZATION
# ==============================================================================
def build_teacher_model(
    num_classes: int = 8,
    pretrained: bool = True,
    freeze_backbone: bool = True
) -> nn.Module:
    weights = EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b2(weights=weights)
    
    in_features = model.classifier[1].in_features
    # Ganti classifier akhir menjadi 8 kelas
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
            
    return model

teacher_model = build_teacher_model(num_classes=NUM_CLASSES, pretrained=True, freeze_backbone=True)
teacher_model = teacher_model.to(DEVICE)
print("[TEACHER INITIALIZED] EfficientNet-B2 Pretrained ImageNet-1K berhasil diinisialisasi.")"""
})

SECTIONS.append({
    "id": 19,
    "title": "19. Model Architecture Summary",
    "markdown": r"""# 19. Model Architecture Summary

Inspeksi struktur parameter model:
- Total parameter: ~7,71 juta.
- Trainable parameter: 11.272 (hanya linear classifier head `1408 * 8 + 8 = 11.272`).
- Frozen parameter: ~7,70 juta.
- Uji forward pass sintetis dengan tensor shape `(2, 3, 288, 288)`.""",
    "code": r"""# ==============================================================================
# SECTION 19: MODEL ARCHITECTURE SUMMARY
# ==============================================================================
def inspect_model_parameters(model: nn.Module) -> Dict[str, Any]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params
    }

param_summary = inspect_model_parameters(teacher_model)
print("PARAMETER SUMMARY:")
print(f"  Total Parameters     : {param_summary['total_params']:,}")
print(f"  Trainable Parameters : {param_summary['trainable_params']:,} (Head Only)")
print(f"  Frozen Parameters    : {param_summary['frozen_params']:,} (Backbone)")

# Forward pass sintetis untuk verifikasi output shape
with torch.no_grad():
    dummy_input = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
    dummy_output = teacher_model(dummy_input)
    print(f"  Input Shape          : {dummy_input.shape}")
    print(f"  Output Shape         : {dummy_output.shape} (Expected: (2, 8))")
    assert dummy_output.shape == (2, NUM_CLASSES)"""
})

SECTIONS.append({
    "id": 20,
    "title": "20. Few-Shot Cross-Validation Configuration",
    "markdown": r"""# 20. Few-Shot Cross-Validation Configuration

Konfigurasi hyperparameter untuk 5-Fold Stratified Cross-Validation model Teacher:
- Optimizer: `AdamW` (learning rate = 1e-3, weight decay = 1e-4)
- Loss Function: `CrossEntropyLoss`
- Batch Size: 8
- Maksimum Epoch: 20 per fold
- Early Stopping: `patience = 5`, memantau `val_loss`
- LR Scheduler: `ReduceLROnPlateau` (factor = 0.5, patience = 3, min_lr = 1e-6)""",
    "code": r"""# ==============================================================================
# SECTION 20: FEW-SHOT CROSS-VALIDATION CONFIGURATION
# ==============================================================================
CV_CONFIG = {
    "experiment_id": "fewshot_pseudolabel_b2_cv",
    "num_folds": 5,
    "batch_size": 8,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "max_epochs": 20,
    "early_stopping_patience": 5,
    "scheduler_factor": 0.5,
    "scheduler_patience": 3,
    "min_lr": 1e-6
}

print("KONFIGURASI CROSS-VALIDATION:")
for k, v in CV_CONFIG.items():
    print(f"  {k:24}: {v}")"""
})

SECTIONS.append({
    "id": 21,
    "title": "21. Cross-Validation Execution",
    "markdown": r"""# 21. Cross-Validation Execution (5 Folds)

Eksekusi pelatihan 5-Fold Cross-Validation untuk mengukur estimasi generalisasi out-of-fold yang jujur:
- Dilindungi oleh flag `RUN_CROSS_VALIDATION`.
- Monitoring per batch dengan `tqdm.auto`.
- Pencatatan metrik per epoch: train loss, train acc, val loss, val acc, macro precision, macro recall, macro F1.
- Checkpoint terbaik per fold disimpan ke `models/checkpoints/fewshot_pseudolabel_b2_v1/cv/fold_{i}/best.pt`.
- Riwayat lengkap disimpan secara persisten ke `outputs/metrics/fewshot_pseudolabel_b2_cv.json`.""",
    "code": r"""# ==============================================================================
# SECTION 21: MANUAL CROSS-VALIDATION EXECUTION
# ==============================================================================
CV_METRICS_PATH = OUTPUT_METRICS_DIR / "fewshot_pseudolabel_b2_cv.json"

def calculate_macro_metrics(y_true: List[int], y_pred: List[int], num_classes: int = 8) -> Dict[str, float]:
    acc = metrics.accuracy_score(y_true, y_pred)
    prec = metrics.precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = metrics.recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = metrics.f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": float(acc), "macro_precision": float(prec), "macro_recall": float(rec), "macro_f1": float(f1)}

def run_cv_fold_training(fold_idx: int, cv_df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    print()
    print(f"--- MEMULAI FOLD {fold_idx} ---")
    train_sub_df = cv_df[cv_df["fold"] != fold_idx].copy()
    val_sub_df = cv_df[cv_df["fold"] == fold_idx].copy()
    
    train_ds = FewShotLabeledDataset(train_sub_df, PROJECT_ROOT, transform=train_transforms)
    val_ds = FewShotLabeledDataset(val_sub_df, PROJECT_ROOT, transform=eval_transforms)
    
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=0)
    
    model = build_teacher_model(num_classes=NUM_CLASSES, pretrained=True, freeze_backbone=True).to(DEVICE)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=config["scheduler_factor"], patience=config["scheduler_patience"], min_lr=config["min_lr"])
    criterion = nn.CrossEntropyLoss()
    
    fold_ckpt_dir = CV_CHECKPOINT_DIR / f"fold_{fold_idx}"
    fold_ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = fold_ckpt_dir / "best.pt"
    
    best_val_loss = float("inf")
    patience_counter = 0
    history = []
    
    for epoch in range(config["max_epochs"]):
        model.train()
        train_loss_total = 0.0
        train_correct = 0
        train_count = 0
        
        pbar = tqdm(train_loader, desc=f"Fold {fold_idx} Epoch {epoch+1}/{config['max_epochs']}", leave=False)
        for images, targets, _ in pbar:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            bs = images.size(0)
            train_loss_total += loss.item() * bs
            preds = outputs.argmax(dim=1)
            train_correct += (preds == targets).sum().item()
            train_count += bs
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{train_correct/train_count:.3f}"})
            
        epoch_train_loss = train_loss_total / train_count
        epoch_train_acc = train_correct / train_count
        
        # Validation
        model.eval()
        val_loss_total = 0.0
        all_val_targets = []
        all_val_preds = []
        with torch.no_grad():
            for images, targets, _ in val_loader:
                images, targets = images.to(DEVICE), targets.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss_total += loss.item() * images.size(0)
                all_val_targets.extend(targets.cpu().numpy().tolist())
                all_val_preds.extend(outputs.argmax(dim=1).cpu().numpy().tolist())
                
        epoch_val_loss = val_loss_total / len(val_ds)
        val_metrics = calculate_macro_metrics(all_val_targets, all_val_preds, NUM_CLASSES)
        scheduler.step(epoch_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        
        saved_best = False
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            saved_best = True
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": best_val_loss,
                "val_metrics": val_metrics,
                "fold": fold_idx
            }, best_ckpt_path)
        else:
            patience_counter += 1
            
        rec = {
            "epoch": epoch + 1, "train_loss": epoch_train_loss, "train_acc": epoch_train_acc,
            "val_loss": epoch_val_loss, "val_acc": val_metrics["accuracy"], "val_f1": val_metrics["macro_f1"],
            "lr": current_lr, "saved_best": saved_best
        }
        history.append(rec)
        print(f"  Ep {epoch+1:02d} | Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.3f} | Val Loss: {epoch_val_loss:.4f} Acc: {val_metrics['accuracy']:.3f} F1: {val_metrics['macro_f1']:.3f} | Best: {'YES' if saved_best else 'NO'}")
        
        if patience_counter >= config["early_stopping_patience"]:
            print(f"  [EARLY STOPPING] Fold {fold_idx} berhenti pada epoch {epoch+1}.")
            break
            
    return {"fold": fold_idx, "best_val_loss": best_val_loss, "best_checkpoint": str(best_ckpt_path), "history": history}

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
    cv_results = []"""
})

SECTIONS.append({
    "id": 22,
    "title": "22. Cross-Validation Results",
    "markdown": r"""# 22. Cross-Validation Results Summary

Tabel rekapitulasi performa 5-Fold Out-Of-Fold Cross-Validation:
- Val Loss rata-rata dan per fold.
- Val Accuracy rata-rata dan per fold.
- Macro F1 rata-rata across all 5 folds.""",
    "code": r"""# ==============================================================================
# SECTION 22: CROSS-VALIDATION RESULTS SUMMARY
# ==============================================================================
def summarize_cv_results(results: List[Dict[str, Any]]):
    if not results:
        print("[INFO] Riwayat CV belum tersedia.")
        return
        
    records = []
    for r in results:
        last_or_best = min(r["history"], key=lambda h: h["val_loss"])
        records.append({
            "Fold": r["fold"],
            "Best Epoch": last_or_best["epoch"],
            "Best Val Loss": f"{last_or_best['val_loss']:.4f}",
            "Val Accuracy": f"{last_or_best['val_acc']*100:.2f}%",
            "Val Macro F1": f"{last_or_best['val_f1']*100:.2f}%",
            "Checkpoint": Path(r["best_checkpoint"]).name
        })
        
    df_cv = pd.DataFrame(records)
    print("REKAPITULASI 5-FOLD OUT-OF-FOLD CV TEACHER:")
    print(df_cv.to_string(index=False))
    
    val_losses = [min(r["history"], key=lambda h: h["val_loss"])["val_loss"] for r in results]
    val_accs = [min(r["history"], key=lambda h: h["val_loss"])["val_acc"] for r in results]
    val_f1s = [min(r["history"], key=lambda h: h["val_loss"])["val_f1"] for r in results]
    
    print("-" * 60)
    print(f"RATA-RATA VAL LOSS    : {np.mean(val_losses):.4f} +/- {np.std(val_losses):.4f}")
    print(f"RATA-RATA VAL ACCURACY: {np.mean(val_accs)*100:.2f}% +/- {np.std(val_accs)*100:.2f}%")
    print(f"RATA-RATA MACRO F1    : {np.mean(val_f1s)*100:.2f}% +/- {np.std(val_f1s)*100:.2f}%")

summarize_cv_results(cv_results)"""
})

SECTIONS.append({
    "id": 23,
    "title": "23. Teacher Training Configuration",
    "markdown": r"""# 23. Final Teacher Training Configuration

Setelah memverifikasi stabilitas melalui 5-Fold CV, kita mengonfigurasi pelatihan **Final Teacher** menggunakan seluruh 40 citra berlabel yang sah:
- Menggunakan seluruh 40 sampel memberikan sinyal pelatihan maksimal kepada Teacher sebelum tahap pseudo-labeling.
- **Catatan Validasi**: Karena seluruh 40 sampel digunakan untuk training, tidak ada holdout validation independen pada tahap ini; jumlah epoch (15 epoch) dipandu oleh epoch optimal dari CV.""",
    "code": r"""# ==============================================================================
# SECTION 23: FINAL TEACHER TRAINING CONFIGURATION
# ==============================================================================
FINAL_TEACHER_CONFIG = {
    "experiment_id": "fewshot_pseudolabel_b2_teacher_final",
    "total_training_samples": 40,
    "batch_size": 8,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 15,
    "optimizer": "AdamW",
    "loss": "CrossEntropyLoss"
}

print("KONFIGURASI FINAL TEACHER TRAINING:")
for k, v in FINAL_TEACHER_CONFIG.items():
    print(f"  {k:24}: {v}")"""
})

SECTIONS.append({
    "id": 24,
    "title": "24. Final Teacher Training",
    "markdown": r"""# 24. Final Teacher Training (All 40 Samples)

Pelatihan Final Teacher pada 40 citra berlabel:
- Dilindungi oleh flag `RUN_TEACHER_TRAINING`.
- Checkpoint disimpan ke `models/checkpoints/fewshot_pseudolabel_b2_v1/teacher/best.pt`.
- Riwayat epoch disimpan ke `outputs/metrics/fewshot_pseudolabel_b2_teacher.json`.""",
    "code": r"""# ==============================================================================
# SECTION 24: MANUAL FINAL TEACHER TRAINING
# ==============================================================================
TEACHER_BEST_CKPT = TEACHER_CHECKPOINT_DIR / "best.pt"
TEACHER_METRICS_PATH = OUTPUT_METRICS_DIR / "fewshot_pseudolabel_b2_teacher.json"

def train_final_teacher(labeled_df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    dataset = FewShotLabeledDataset(labeled_df, PROJECT_ROOT, transform=train_transforms)
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0)
    
    model = build_teacher_model(num_classes=NUM_CLASSES, pretrained=True, freeze_backbone=True).to(DEVICE)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    
    history = []
    print(f"Melatih Final Teacher pada {len(dataset)} citra selama {config['epochs']} epoch...")
    
    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(loader, desc=f"Final Teacher Ep {epoch+1}/{config['epochs']}", leave=False)
        for images, targets, _ in pbar:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            bs = images.size(0)
            total_loss += loss.item() * bs
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += bs
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct/total:.3f}"})
            
        ep_loss = total_loss / total
        ep_acc = correct / total
        history.append({"epoch": epoch + 1, "train_loss": ep_loss, "train_acc": ep_acc})
        print(f"  Ep {epoch+1:02d} | Train Loss: {ep_loss:.4f} | Train Acc: {ep_acc*100:.2f}%")
        
    torch.save({
        "epoch": config["epochs"],
        "model_state_dict": model.state_dict(),
        "final_loss": ep_loss,
        "final_acc": ep_acc,
        "class_names": CLASS_NAMES,
        "config": config
    }, TEACHER_BEST_CKPT)
    
    return {"final_loss": ep_loss, "final_acc": ep_acc, "history": history}

if TEACHER_BEST_CKPT.exists() and TEACHER_METRICS_PATH.exists():
    with open(TEACHER_METRICS_PATH, "r", encoding="utf-8") as f:
        teacher_results = json.load(f)
    print(f"[LOADED] Checkpoint Final Teacher tersedia: {TEACHER_BEST_CKPT}")
elif len(labeled_pool_df) > 0:
    teacher_results = train_final_teacher(labeled_pool_df, FINAL_TEACHER_CONFIG)
    with open(TEACHER_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(teacher_results, f, indent=2)
    print(f"[SAVED] Final Teacher Checkpoint disimpan ke {TEACHER_BEST_CKPT}")
else:
    print("[STATUS] Labeled pool belum tersedia.")
    teacher_results = {}"""
})

SECTIONS.append({
    "id": 25,
    "title": "25. Teacher Training History",
    "markdown": r"""# 25. Teacher Training History & Curves

Plot kurva pelatihan Final Teacher (loss dan akurasi) dari riwayat persisten pada disk.""",
    "code": r"""# ==============================================================================
# SECTION 25: TEACHER TRAINING HISTORY & CURVES
# ==============================================================================
def plot_teacher_history(results: Dict[str, Any]):
    if not results or "history" not in results:
        print("[INFO] Riwayat pelatihan Final Teacher belum tersedia.")
        return
        
    epochs = [h["epoch"] for h in results["history"]]
    losses = [h["train_loss"] for h in results["history"]]
    accs = [h["train_acc"] for h in results["history"]]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(epochs, losses, "o-", color="firebrick", label="Train Loss")
    ax1.set_title("Final Teacher Training Loss", weight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()
    
    ax2.plot(epochs, accs, "s-", color="navy", label="Train Accuracy")
    ax2.set_title("Final Teacher Training Accuracy", weight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

plot_teacher_history(teacher_results)"""
})

SECTIONS.append({
    "id": 26,
    "title": "26. Teacher Validation Diagnostics",
    "markdown": r"""# 26. Teacher Validation Diagnostics

Diskusi saintifik mengenai performa Teacher Model:
- Pada rejim few-shot (5 sampel/kelas), model berbasis transfer learning EfficientNet-B2 mampu menangkap representasi visual umum dari ImageNet.
- Namun, varians estimasi tinggi karena ukuran validasi yang kecil.
- Oleh karena itu, Teacher membutuhkan **threshold confidence yang tinggi (>= 0.95)** pada tahap pseudo-labeling agar tidak mentransfer noise atau misklasifikasi ke model Student.""",
    "code": r"""# ==============================================================================
# SECTION 26: TEACHER VALIDATION DIAGNOSTICS
# ==============================================================================
print("RINGKASAN DIAGNOSTIK TEACHER MODEL:")
print("1. Backbone     : Pretrained ImageNet-1K mempertahankan representasi filter kaya.")
print("2. Classifier   : 8 kelas sampah terhubung dengan 1408 fitur backbone.")
print("3. Risiko       : Konfirmasi bias (confirmation bias) jika pseudo-label tidak disaring ketat.")
print("4. Rekomendasi  : Gunakan confidence threshold >= 0.95 pada Section 30 untuk menyaring pseudo-label.")"""
})
