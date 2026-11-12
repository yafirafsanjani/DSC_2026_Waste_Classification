# scripts/sec_12_17.py
SECTIONS = []

SECTIONS.append({
    "id": 12,
    "title": "12. Image Preprocessing",
    "markdown": r"""# 12. Image Preprocessing (EfficientNet-B2 Pretrained Standard)

Sesuai bobot resmi `EfficientNet_B2_Weights.IMAGENET1K_V1`, pipeline preprocessing untuk validasi dan inferensi:
1. Konversi ke RGB.
2. Resize ke 288 (Bicubic Interpolation).
3. Center Crop 288x288.
4. Konversi ke Tensor float32 dalam skala [0.0, 1.0].
5. Normalisasi ImageNet standar:
   - Mean: `[0.485, 0.456, 0.406]`
   - Std: `[0.229, 0.224, 0.225]`""",
    "code": r"""# ==============================================================================
# SECTION 12: IMAGE PREPROCESSING
# ==============================================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 288

def get_eval_transforms(image_size: int = 288):
    if USE_TRANSFORMS_V2:
        return v2.Compose([
            v2.ToImage(),
            v2.Resize(image_size, interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
            v2.CenterCrop(image_size),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        return torchvision.transforms.Compose([
            torchvision.transforms.Resize(image_size, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.CenterCrop(image_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

eval_transforms = get_eval_transforms(IMAGE_SIZE)
print("Transformasi Evaluasi & Inferensi:")
print(eval_transforms)"""
})

SECTIONS.append({
    "id": 13,
    "title": "13. Training Augmentation",
    "markdown": r"""# 13. Training Augmentation (Controlled & Photorealistic)

Pada skenario data sedikit (Few-Shot), augmentasi data sangat penting untuk mencegah penghafalan (memorization), namun tidak boleh terlalu ekstrem agar tidak merusak representasi visual kelas:
- `RandomHorizontalFlip(p=0.5)`
- `RandomRotation(degrees=15)`
- `RandomAffine(degrees=0, shear=10)`
- `ColorJitter(brightness=0.10)`
- Output berukuran tepat 288x288 dengan Normalisasi ImageNet.""",
    "code": r"""# ==============================================================================
# SECTION 13: TRAINING AUGMENTATION
# ==============================================================================
def get_train_transforms(image_size: int = 288):
    if USE_TRANSFORMS_V2:
        return v2.Compose([
            v2.ToImage(),
            v2.Resize(image_size, interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
            v2.CenterCrop(image_size),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(degrees=15),
            v2.RandomAffine(degrees=0, shear=10),
            v2.ColorJitter(brightness=0.10),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        return torchvision.transforms.Compose([
            torchvision.transforms.Resize(image_size, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.CenterCrop(image_size),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomRotation(degrees=15),
            torchvision.transforms.RandomAffine(degrees=0, shear=10),
            torchvision.transforms.ColorJitter(brightness=0.10),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

train_transforms = get_train_transforms(IMAGE_SIZE)
print("Transformasi Pelatihan (Augmentasi):")
print(train_transforms)"""
})

SECTIONS.append({
    "id": 14,
    "title": "14. Visualization of Original Images",
    "markdown": r"""# 14. Visualization of Original Images

Visualisasi 1 contoh citra asli dari labeled pool untuk masing-masing dari 8 kelas sampah target.""",
    "code": r"""# ==============================================================================
# SECTION 14: VISUALIZATION OF ORIGINAL IMAGES
# ==============================================================================
def plot_sample_images_per_class(labeled_df: pd.DataFrame, project_root: Path):
    if len(labeled_df) == 0:
        print("[INFO] Labeled pool belum tersedia untuk divisualisasikan.")
        return
        
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.flatten()
    
    for idx, cls_name in enumerate(CLASS_NAMES):
        cls_rows = labeled_df[labeled_df["class_name"] == cls_name]
        if len(cls_rows) == 0:
            continue
        first_row = cls_rows.iloc[0]
        img_path = project_root / first_row["relative_path"]
        with Image.open(img_path) as img:
            axes[idx].imshow(img)
            axes[idx].set_title(f"[{first_row['class_idx']}] {cls_name} ({img.size[0]}x{img.size[1]})")
            axes[idx].axis("off")
            
    plt.suptitle("Contoh Citra Asli dari Labeled Pool (8 Kelas Target)", fontsize=14, weight="bold")
    plt.tight_layout()
    plt.show()

plot_sample_images_per_class(labeled_pool_df, PROJECT_ROOT)"""
})

SECTIONS.append({
    "id": 15,
    "title": "15. Visualization of Augmented Images",
    "markdown": r"""# 15. Visualization of Augmented Images

Visualisasi variasi augmentasi pada satu citra sampel untuk memastikan transformasi tetap mempertahankan karakteristik semantik kelas objek.""",
    "code": r"""# ==============================================================================
# SECTION 15: VISUALIZATION OF AUGMENTED IMAGES
# ==============================================================================
def plot_augmented_variations(labeled_df: pd.DataFrame, project_root: Path):
    if len(labeled_df) == 0:
        print("[INFO] Labeled pool belum tersedia.")
        return
        
    sample_path = project_root / labeled_df.iloc[0]["relative_path"]
    with Image.open(sample_path) as raw_img:
        rgb_img = raw_img.convert("RGB")
        
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.5))
    axes[0].imshow(rgb_img)
    axes[0].set_title("Citra Asli")
    axes[0].axis("off")
    
    for i in range(1, 5):
        aug_tensor = train_transforms(rgb_img)
        np_img = aug_tensor.permute(1, 2, 0).cpu().numpy()
        np_img = std * np_img + mean
        np_img = np.clip(np_img, 0, 1)
        axes[i].imshow(np_img)
        axes[i].set_title(f"Augmentasi {i}")
        axes[i].axis("off")
        
    plt.suptitle(f"Variasi Augmentasi Training pada Kelas '{labeled_df.iloc[0]['class_name']}'", fontsize=12, weight="bold")
    plt.tight_layout()
    plt.show()

plot_augmented_variations(labeled_pool_df, PROJECT_ROOT)"""
})

SECTIONS.append({
    "id": 16,
    "title": "16. Dataset and DataLoader Construction",
    "markdown": r"""# 16. Dataset and DataLoader Construction

Implementasi kelas PyTorch `Dataset` yang mandiri di dalam notebook:
- `FewShotLabeledDataset`: membaca citra berlabel, mengembalikan `(tensor, class_idx, image_id)`.
- `UnlabeledDataset`: membaca citra tanpa label, mengembalikan `(tensor, image_id)` ? **tanpa label asli**.
- `CompetitionTestDataset`: membaca citra test kompetisi untuk inferensi akhir.""",
    "code": r"""# ==============================================================================
# SECTION 16: DATASET AND DATALOADER CONSTRUCTION
# ==============================================================================
class FewShotLabeledDataset(Dataset):
    def __init__(self, df: pd.DataFrame, project_root: Path, transform: Optional[Callable] = None):
        self.df = df.reset_index(drop=True)
        self.project_root = project_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        img_path = self.project_root / row["relative_path"]
        with Image.open(img_path) as img:
            image = img.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["class_idx"]), str(row["image_id"])

class UnlabeledDataset(Dataset):
    def __init__(self, df: pd.DataFrame, project_root: Path, transform: Optional[Callable] = None):
        self.df = df.reset_index(drop=True)
        self.project_root = project_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        row = self.df.iloc[idx]
        img_path = self.project_root / row["relative_path"]
        with Image.open(img_path) as img:
            image = img.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, str(row["image_id"])

class CompetitionTestDataset(Dataset):
    def __init__(self, image_paths: List[Path], transform: Optional[Callable] = None):
        self.image_paths = sorted(image_paths, key=lambda p: p.name)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        path = self.image_paths[idx]
        with Image.open(path) as img:
            image = img.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, path.stem

print("[DATASET TERDEFINISI]")
print("  - FewShotLabeledDataset : (tensor, class_idx, image_id)")
print("  - UnlabeledDataset      : (tensor, image_id) [NO LABEL]")
print("  - CompetitionTestDataset: (tensor, image_id)")"""
})

SECTIONS.append({
    "id": 17,
    "title": "17. Class Distribution Visualization",
    "markdown": r"""# 17. Class Distribution Visualization

Visualisasi distribusi sampel berlabel (tepat 5 citra per kelas) dan proporsi data berlabel vs data tanpa label dalam pipeline eksperimen semi-supervised.""",
    "code": r"""# ==============================================================================
# SECTION 17: CLASS DISTRIBUTION VISUALIZATION
# ==============================================================================
def plot_dataset_proportions(labeled_df: pd.DataFrame, unlabeled_df: pd.DataFrame, test_dir: Path):
    labeled_count = len(labeled_df)
    unlabeled_count = len(unlabeled_df)
    test_count = len(list(test_dir.glob("*.*"))) if test_dir.exists() else 0
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    if labeled_count > 0:
        counts = labeled_df["class_name"].value_counts().reindex(CLASS_NAMES).fillna(0)
        bars = ax1.bar(counts.index, counts.values, color="steelblue", edgecolor="black")
        ax1.set_title(f"Distribusi Labeled Pool (Total: {labeled_count} Citra)", weight="bold")
        ax1.set_ylabel("Jumlah Sampel")
        ax1.tick_params(axis="x", rotation=45)
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., h + 0.1, f"{int(h)}", ha="center", va="bottom", weight="bold")
    else:
        ax1.text(0.5, 0.5, "Labeled pool belum dimuat", ha="center", va="center")
        
    sizes = [labeled_count, unlabeled_count, test_count]
    labels = [f"Labeled Manual ({labeled_count})", f"Unlabeled Pool ({unlabeled_count})", f"Comp Test ({test_count})"]
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e"]
    
    if sum(sizes) > 0:
        ax2.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=140, explode=(0.1, 0, 0))
        ax2.set_title("Proporsi Alokasi Citra dalam Proyek", weight="bold")
    else:
        ax2.text(0.5, 0.5, "Dataset belum dialokasikan", ha="center", va="center")
        
    plt.tight_layout()
    plt.show()

plot_dataset_proportions(labeled_pool_df, unlabeled_pool_df, DATA_TEST_DIR)"""
})
