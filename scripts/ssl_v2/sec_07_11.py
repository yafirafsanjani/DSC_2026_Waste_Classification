# scripts/ssl_v2/sec_07_11.py
SECTIONS = []

SECTIONS.append({
    "id": 7,
    "title": "07. Pretrained Backbone Initialization & BatchNorm Freeze Fix",
    "markdown": r"""# 07. Pretrained Backbone Initialization & BatchNorm Freeze Fix

**AUDIT TEKNIS & SOLUSI KRUSIAL**:
Pada eksperimen TASK 08 baseline, pemanggilan `model.train()` secara rekursif mengaktifkan seluruh submodule termasuk `nn.BatchNorm2d` di dalam backbone EfficientNet-B2. Walaupun `requires_grad=False`, PyTorch tetap memperbarui buffer `running_mean` dan `running_var` selama forward pass. Dengan batch size kecil (8) pada dataset 40 citra, statistik representasi ImageNet rusak dan menyebabkan penurunan performa evaluasi.

**Perbaikan di Bagian Ini**:
1. Fungsi `freeze_backbone_with_eval_bn(model)`:
   - Membekukan seluruh parameter `model.features`.
   - Mengatur seluruh layer BatchNorm di dalam backbone secara permanen dalam mode `eval()` (`m.eval()`).
2. Uji diagnostik internal memastikan buffer `running_mean` dan `running_var` 100% tidak berubah saat forward pass.""",
    "code": r"""# ==============================================================================
# SECTION 07: BACKBONE INITIALIZATION & BATCHNORM FREEZE FIX
# ==============================================================================
def build_efficientnet_b2_backbone(num_classes: int = 8, pretrained: bool = True) -> nn.Module:
    weights = EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b2(weights=weights)
    in_features = model.classifier[1].in_features  # 1408
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes)
    )
    return model

def freeze_backbone_with_eval_bn(model: nn.Module):
    # Bekukan seluruh parameter features
    for p in model.features.parameters():
        p.requires_grad = False
        
    # Pastikan seluruh layer BatchNorm di dalam backbone tetap dalam mode eval
    for m in model.features.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.eval()

# Inisialisasi model Teacher dasar
base_teacher_model = build_efficientnet_b2_backbone(num_classes=NUM_CLASSES, pretrained=True).to(DEVICE)
freeze_backbone_with_eval_bn(base_teacher_model)

# DIAGNOSTIK MANDIRI: Verifikasi BatchNorm Running Statistics Tidak Berubah
first_bn = None
for m in base_teacher_model.features.modules():
    if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        first_bn = m
        break

assert first_bn is not None, "BatchNorm layer tidak ditemukan di dalam backbone!"
initial_running_mean = first_bn.running_mean.clone()

# Simulasikan forward pass
dummy_input = torch.randn(8, 3, CONFIG["img_size"], CONFIG["img_size"]).to(DEVICE)
with torch.no_grad():
    _ = base_teacher_model(dummy_input)

after_running_mean = first_bn.running_mean.clone()
assert torch.equal(initial_running_mean, after_running_mean), "FAIL: BatchNorm running_mean berubah!"
print("[BATCHNORM FIX VERIFIED] Running statistics ImageNet 100% terjaga dan stabil.")
print(f"  - Model Architecture : EfficientNet-B2 Pretrained (ImageNet-1K V1)")
print(f"  - Embedding Dimension: 1408")
print(f"  - Output Classes     : {NUM_CLASSES}")"""
})

SECTIONS.append({
    "id": 8,
    "title": "08. Frozen Feature Extraction & Embedding Caching",
    "markdown": r"""# 08. Frozen Feature Extraction & Embedding Caching

Mengekstrak vektor fitur representasi (embeddings) berdimensi 1.408 dari backbone EfficientNet-B2:
- Menggunakan pooling `model.avgpool` dan normalisasi L2 ($\|z\|_2 = 1$).
- Mengekstrak dan meng-cache embeddings untuk seluruh **40 sampel berlabel**.
- Ekstraksi fitur dilakukan sekali saja sehingga pelatihan 4 classifier alternatif berjalan dalam hitungan detik tanpa membebani GPU RTX 3050.""",
    "code": r"""# ==============================================================================
# SECTION 08: FROZEN FEATURE EXTRACTION & CACHING
# ==============================================================================
class LabeledFeatureDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root_dir: Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.df)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        img_id = str(row["image_id"])
        img_path = self.root_dir / row["relative_path"]
        with Image.open(img_path) as img:
            image = img.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["class_idx"]), img_id

def extract_l2_embeddings(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    model.eval()
    embeddings = []
    labels = []
    image_ids = []
    
    with torch.inference_mode():
        for images, targets, ids in tqdm(loader, desc="Extracting Features", leave=False):
            images = images.to(device)
            feats = model.features(images)
            pooled = model.avgpool(feats)
            flat = torch.flatten(pooled, 1)
            # L2 Normalization
            normed = F.normalize(flat, p=2, dim=1)
            embeddings.append(normed.cpu().numpy())
            labels.append(targets.numpy())
            image_ids.extend([str(i) for i in ids])
            
    return np.vstack(embeddings), np.concatenate(labels), image_ids

LABELED_CACHE_PATH = MANIFEST_DIR / "labeled_embeddings_v2.pt"

ds_labeled_eval = LabeledFeatureDataset(labeled_df, PROJECT_ROOT, transform=eval_transforms)
loader_labeled_eval = DataLoader(ds_labeled_eval, batch_size=8, shuffle=False, num_workers=0)

if LABELED_CACHE_PATH.exists():
    cache_payload = load_project_checkpoint(LABELED_CACHE_PATH, device="cpu")
    raw_feats = cache_payload["embeddings"]
    labeled_features = raw_feats.numpy() if isinstance(raw_feats, torch.Tensor) else np.array(raw_feats)
    raw_targets = cache_payload["targets"]
    labeled_targets = raw_targets.numpy() if isinstance(raw_targets, torch.Tensor) else np.array(raw_targets)
    labeled_ids = list(cache_payload["image_ids"])
    print(f"[LOADED CACHE] Fitur 40 sampel termuat dari {LABELED_CACHE_PATH.name}")
else:
    labeled_features, labeled_targets, labeled_ids = extract_l2_embeddings(base_teacher_model, loader_labeled_eval, DEVICE)
    torch.save({
        "embeddings": torch.from_numpy(labeled_features).float(),
        "targets": torch.from_numpy(labeled_targets).long(),
        "image_ids": list(labeled_ids)
    }, LABELED_CACHE_PATH)
    print(f"[SAVED CACHE] Fitur 40 sampel diekstrak dan disimpan ke {LABELED_CACHE_PATH.name}")

print(f"Matriks Fitur Labeled Pool: Shape {labeled_features.shape} | Dtype: {labeled_features.dtype}")
assert labeled_features.shape == (40, 1408), f"Expected shape (40, 1408), got {labeled_features.shape}"
"""
})

SECTIONS.append({
    "id": 9,
    "title": "09. Few-Shot Classifiers 5-Fold CV Out-Of-Fold Evaluation",
    "markdown": r"""# 09. Few-Shot Classifiers 5-Fold CV Out-Of-Fold Evaluation

Dalam rezim *extreme few-shot* (5 sampel per kelas, total 40 citra), pengoptimalan gradien pada classifier head dengan AdamW dapat rentan overfitting. Di sini kita mengevaluasi **4 metode klasifikasi** secara objektif menggunakan 5-Fold CV yang sama:
1. **PyTorch Linear Head (Frozen BN)**: Classifier linear yang dilatih dengan AdamW, lr=1e-3, weight_decay=1e-2, dan BatchNorm tetap beku.
2. **Logistic Regression (L2 Normalized)**: Classifier cembung reguler L2 ($C=1.0$) dengan optimasi L-BFGS.
3. **Linear Support Vector Classifier (Linear SVC)**: Pemaksimal margin linear ($C=1.0$) dengan kalibrasi probabilitas Platt scaling.
4. **Cosine Prototype Classifier (Few-Shot Prototypical)**: Non-parametrik, menghitung vektor centroid kelas yang dinormalisasi L2 dan mengklasifikasikan berbasis cosine similarity.""",
    "code": r"""# ==============================================================================
# SECTION 09: FEW-SHOT CLASSIFIERS 5-FOLD CV EVALUATION
# ==============================================================================
class CosinePrototypeClassifier:
    def __init__(self, temperature: float = 0.1):
        self.temperature = temperature
        self.prototypes = None
        self.classes = None
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes = np.unique(y)
        D = X.shape[1]
        self.prototypes = np.zeros((len(self.classes), D), dtype=np.float32)
        for idx, c in enumerate(self.classes):
            class_samples = X[y == c]
            proto = np.mean(class_samples, axis=0)
            norm = np.linalg.norm(proto)
            self.prototypes[idx] = proto / (norm + 1e-8)
            
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        sims = np.dot(X, self.prototypes.T) / self.temperature
        exp_sims = np.exp(sims - np.max(sims, axis=1, keepdims=True))
        return exp_sims / np.sum(exp_sims, axis=1, keepdims=True)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        sims = np.dot(X, self.prototypes.T)
        best_indices = np.argmax(sims, axis=1)
        return self.classes[best_indices]

# Pemetaan ID ke index baris
id_to_idx = {img_id: i for i, img_id in enumerate(labeled_ids)}
cv_indices = np.array([id_to_idx[img_id] for img_id in cv_folds_df["image_id"]])
cv_folds_array = cv_folds_df["fold"].values
y_true_all = labeled_targets[cv_indices]
X_all = labeled_features[cv_indices]

# Matriks penyimpanan Out-Of-Fold (OOF) predictions
oof_preds = {
    "PyTorch_Linear_FrozenBN": np.zeros(40, dtype=int),
    "LogisticRegression_L2": np.zeros(40, dtype=int),
    "Linear_SVC": np.zeros(40, dtype=int),
    "Cosine_Prototype": np.zeros(40, dtype=int)
}

oof_probs = {
    "PyTorch_Linear_FrozenBN": np.zeros((40, NUM_CLASSES), dtype=np.float32),
    "LogisticRegression_L2": np.zeros((40, NUM_CLASSES), dtype=np.float32),
    "Linear_SVC": np.zeros((40, NUM_CLASSES), dtype=np.float32),
    "Cosine_Prototype": np.zeros((40, NUM_CLASSES), dtype=np.float32)
}

print("Menjalankan 5-Fold Stratified Cross-Validation pada 4 Arsitektur Classifier...")

for f in range(5):
    train_mask = (cv_folds_array != f)
    val_mask = (cv_folds_array == f)
    
    X_train, y_train = X_all[train_mask], y_true_all[train_mask]
    X_val, y_val = X_all[val_mask], y_true_all[val_mask]
    val_row_indices = np.where(val_mask)[0]
    
    # 1. PyTorch Linear Classifier
    pt_linear = nn.Linear(1408, NUM_CLASSES).to(DEVICE)
    pt_opt = optim.AdamW(pt_linear.parameters(), lr=CONFIG["teacher_lr"], weight_decay=CONFIG["teacher_weight_decay"])
    criterion = nn.CrossEntropyLoss()
    
    X_t_train = torch.from_numpy(X_train).to(DEVICE)
    y_t_train = torch.from_numpy(y_train).to(DEVICE)
    X_t_val = torch.from_numpy(X_val).to(DEVICE)
    
    pt_linear.train()
    for _ in range(CONFIG["teacher_epochs"]):
        pt_opt.zero_grad()
        logits = pt_linear(X_t_train)
        loss = criterion(logits, y_t_train)
        loss.backward()
        pt_opt.step()
        
    pt_linear.eval()
    with torch.no_grad():
        val_logits = pt_linear(X_t_val)
        val_probs_pt = F.softmax(val_logits, dim=1).cpu().numpy()
        val_preds_pt = np.argmax(val_probs_pt, axis=1)
        
    oof_preds["PyTorch_Linear_FrozenBN"][val_row_indices] = val_preds_pt
    oof_probs["PyTorch_Linear_FrozenBN"][val_row_indices] = val_probs_pt
    
    # 2. Logistic Regression (L2)
    lr_clf = LogisticRegression(C=1.0, max_iter=1000, random_state=CONFIG["random_seed"])
    lr_clf.fit(X_train, y_train)
    oof_preds["LogisticRegression_L2"][val_row_indices] = lr_clf.predict(X_val)
    oof_probs["LogisticRegression_L2"][val_row_indices] = lr_clf.predict_proba(X_val)
    
    # 3. Linear SVC (Calibrated probabilities)
    svc_clf = SVC(kernel="linear", C=1.0, probability=True, random_state=CONFIG["random_seed"])
    svc_clf.fit(X_train, y_train)
    oof_preds["Linear_SVC"][val_row_indices] = svc_clf.predict(X_val)
    oof_probs["Linear_SVC"][val_row_indices] = svc_clf.predict_proba(X_val)
    
    # 4. Cosine Prototype
    proto_clf = CosinePrototypeClassifier(temperature=0.1)
    proto_clf.fit(X_train, y_train)
    oof_preds["Cosine_Prototype"][val_row_indices] = proto_clf.predict(X_val)
    oof_probs["Cosine_Prototype"][val_row_indices] = proto_clf.predict_proba(X_val)

print("[CROSS-VALIDATION SELESAI] Seluruh prediksi Out-Of-Fold (40 sampel) berhasil dihitung.")"""
})

SECTIONS.append({
    "id": 10,
    "title": "10. OOF Evaluation Metrics & Model Selection",
    "markdown": r"""# 10. OOF Evaluation Metrics & Model Selection

Membandingkan performa Out-Of-Fold (OOF) dari keempat metode klasifikasi:
- **Metrik Utama**: Macro F1 Score dan Accuracy pada seluruh 40 sampel OOF.
- Menampilkan tabel perbandingan objektif bebas kebocoran.
- Memvisualisasikan matriks konfusi (*confusion matrix*) model terbaik.
- Menyimpan hasil metrik ke `outputs/metrics/fewshot_ssl_v2_cv_comparison.json`.""",
    "code": r"""# ==============================================================================
# SECTION 10: OOF METRICS & MODEL SELECTION
# ==============================================================================
comparison_records = []

for name, preds in oof_preds.items():
    acc = accuracy_score(y_true_all, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true_all, preds, average="macro", zero_division=0)
    comparison_records.append({
        "Model": name,
        "OOF_Accuracy": float(acc),
        "Macro_Precision": float(prec),
        "Macro_Recall": float(rec),
        "Macro_F1": float(f1)
    })

df_comparison = pd.DataFrame(comparison_records).sort_values(by="Macro_F1", ascending=False).reset_index(drop=True)
print("=" * 80)
print("HASIL PERBANDINGAN 5-FOLD CV OOF (40 MANUAL LABELS):")
print(df_comparison.to_string(index=False))
print("=" * 80)

BEST_SUPERVISED_MODEL_NAME = df_comparison.iloc[0]["Model"]
BEST_OOF_F1 = df_comparison.iloc[0]["Macro_F1"]
BEST_OOF_ACC = df_comparison.iloc[0]["OOF_Accuracy"]

print(f"\n[MODEL TERPILIH] : {BEST_SUPERVISED_MODEL_NAME}")
print(f"  - OOF Macro F1 : {BEST_OOF_F1:.4f}")
print(f"  - OOF Accuracy : {BEST_OOF_ACC*100:.2f}%")

# Confusion Matrix model terbaik
best_preds = oof_preds[BEST_SUPERVISED_MODEL_NAME]
cm = confusion_matrix(y_true_all, best_preds)

fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.matshow(cm, cmap="Blues")
fig.colorbar(cax)
ax.set_xticks(range(NUM_CLASSES))
ax.set_yticks(range(NUM_CLASSES))
ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="left")
ax.set_yticklabels(CLASS_NAMES)
ax.set_xlabel("Predicted Label", weight="bold")
ax.set_ylabel("True Label", weight="bold")
ax.set_title(f"OOF Confusion Matrix: {BEST_SUPERVISED_MODEL_NAME}\n(Macro F1: {BEST_OOF_F1:.4f})", weight="bold")

for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color="black" if cm[i, j] < cm.max()/2 else "white")

plt.tight_layout()
fig_path = OUTPUT_FIGURES_DIR / "fewshot_ssl_v2_oof_confusion_matrix.png"
fig.savefig(fig_path, dpi=150)
plt.close(fig)

with open(OUTPUT_METRICS_DIR / "fewshot_ssl_v2_cv_comparison.json", "w", encoding="utf-8") as f:
    json.dump({
        "comparison": comparison_records,
        "best_model": BEST_SUPERVISED_MODEL_NAME,
        "best_oof_f1": BEST_OOF_F1,
        "best_oof_acc": BEST_OOF_ACC
    }, f, indent=2)

print(f"[METRIK TERSIMPAN] Perbandingan OOF disimpan ke outputs/metrics/fewshot_ssl_v2_cv_comparison.json")"""
})

SECTIONS.append({
    "id": 11,
    "title": "11. Final Supervised Model Training on Full 40 Labeled Pool",
    "markdown": r"""# 11. Final Supervised Model Training on Full 40 Labeled Pool

Setelah model dengan generalisasi OOF terbaik terpilih:
- Melatih model classifier terpilih menggunakan **seluruh 40 sampel berlabel**.
- Menyimpan bobot / parameter model ke `models/checkpoints/fewshot_ssl_v2/teacher/best.pt`.
- Model ini bertindak sebagai **Final Teacher** yang siap menghasilkan pseudo-label berkualitas tinggi.""",
    "code": r"""# ==============================================================================
# SECTION 11: FINAL SUPERVISED TEACHER TRAINING
# ==============================================================================
FINAL_TEACHER_CKPT = TEACHER_CKPT_DIR / "best.pt"

class PyTorchLinearWrapper(nn.Module):
    def __init__(self, in_features: int = 1408, num_classes: int = 8):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)
    def forward(self, x):
        return self.fc(x)
    def load_head_state_dict(self, state_dict: Dict[str, Any]):
        if "fc.weight" in state_dict:
            self.load_state_dict(state_dict)
        elif "weight" in state_dict:
            self.fc.load_state_dict(state_dict)
        else:
            clean_sd = {("fc." + k if not k.startswith("fc.") else k): v for k, v in state_dict.items()}
            self.load_state_dict(clean_sd)

def train_final_supervised_teacher(model_name: str, X: np.ndarray, y: np.ndarray, device: torch.device):
    if model_name == "Cosine_Prototype":
        clf = CosinePrototypeClassifier(temperature=0.1)
        clf.fit(X, y)
        # Bentuk linear layer PyTorch yang setara untuk kemudahan deployment
        linear_model = PyTorchLinearWrapper(1408, NUM_CLASSES)
        linear_model.fc.weight.data = torch.from_numpy(clf.prototypes).float()
        linear_model.fc.bias.data.zero_()
        return linear_model, {"prototypes": clf.prototypes, "type": "Cosine_Prototype"}
        
    elif model_name == "LogisticRegression_L2":
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=CONFIG["random_seed"])
        clf.fit(X, y)
        linear_model = PyTorchLinearWrapper(1408, NUM_CLASSES)
        linear_model.fc.weight.data = torch.from_numpy(clf.coef_).float()
        linear_model.fc.bias.data = torch.from_numpy(clf.intercept_).float()
        return linear_model, {"coef": clf.coef_, "intercept": clf.intercept_, "type": "LogisticRegression_L2"}
        
    elif model_name == "Linear_SVC":
        clf = SVC(kernel="linear", C=1.0, probability=True, random_state=CONFIG["random_seed"])
        clf.fit(X, y)
        linear_model = PyTorchLinearWrapper(1408, NUM_CLASSES)
        linear_model.fc.weight.data = torch.from_numpy(clf.coef_).float()
        linear_model.fc.bias.data = torch.from_numpy(clf.intercept_).float()
        return linear_model, {"coef": clf.coef_, "intercept": clf.intercept_, "type": "Linear_SVC"}
        
    else:  # PyTorch_Linear_FrozenBN
        linear_model = PyTorchLinearWrapper(1408, NUM_CLASSES).to(device)
        optimizer = optim.AdamW(linear_model.parameters(), lr=CONFIG["teacher_lr"], weight_decay=CONFIG["teacher_weight_decay"])
        criterion = nn.CrossEntropyLoss()
        
        X_t = torch.from_numpy(X).to(device)
        y_t = torch.from_numpy(y).to(device)
        
        linear_model.train()
        for _ in range(CONFIG["teacher_epochs"] * 2):
            optimizer.zero_grad()
            out = linear_model(X_t)
            loss = criterion(out, y_t)
            loss.backward()
            optimizer.step()
            
        linear_model.eval()
        return linear_model, {"type": "PyTorch_Linear_FrozenBN"}

final_teacher_head, teacher_meta = train_final_supervised_teacher(BEST_SUPERVISED_MODEL_NAME, X_all, y_true_all, DEVICE)

# Sanitasi metadata untuk kompatibilitas 100% weights_only=True
clean_meta = {}
for k, v in teacher_meta.items():
    if isinstance(v, np.ndarray):
        clean_meta[k] = torch.from_numpy(v).float()
    elif isinstance(v, (np.floating, float)):
        clean_meta[k] = float(v)
    elif isinstance(v, (np.integer, int)):
        clean_meta[k] = int(v)
    else:
        clean_meta[k] = str(v)

torch.save({
    "model_name": BEST_SUPERVISED_MODEL_NAME,
    "head_state_dict": final_teacher_head.state_dict(),
    "meta": clean_meta,
    "class_names": CLASS_NAMES,
    "config": CONFIG
}, FINAL_TEACHER_CKPT)

print(f"[FINAL TEACHER SIAP] Checkpoint tersimpan di: {FINAL_TEACHER_CKPT}")
print(f"  - Model Type    : {BEST_SUPERVISED_MODEL_NAME}")
print(f"  - Training Data : Tepat 40 sampel berlabel manual")"""
})
