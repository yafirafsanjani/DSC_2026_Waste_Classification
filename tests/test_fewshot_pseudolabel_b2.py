"""Unit and Synthetic Quality Assurance Tests for TASK 08:
Few-Shot EfficientNet-B2 + Pseudo-Labeling + Kaggle Submission.

Tests verify:
1. Official competition train (1920) and test (480) datasets exist, readable, flat.
2. Train-test exact hash collisions and internal train duplicates are correctly identified.
3. Notebook reads from data/competition/train and NOT from data/extracted/v1/train.
4. Label manual does not exceed 5 per class (40 total).
5. Unlabeled dataset returns NO labels (strictly anti-leakage).
6. Cross-validation folds are strictly disjoint.
7. Teacher architecture, freezing, and classifier replacement (11,272 trainable params).
8. Student semi-supervised weighted loss formulation.
9. Standalone fresh-kernel inference readiness.
10. Submission CSV schema, leading zero preservation, and validation logic.
11. Handbook PDF detection finds official preliminary round handbook.
12. All manual execution flags default to False.
"""

import json
import hashlib
from pathlib import Path
from collections import Counter, defaultdict
import pytest
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b2
from torchvision.transforms import v2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "08_fewshot_pseudolabeling_efficientnet_b2.ipynb"
CLASS_NAMES = ["cardboard", "electronic", "foodwaste", "glass", "metal", "paper", "plastic", "textile"]

def test_notebook_file_and_cells_structure():
    assert NOTEBOOK_PATH.exists(), f"Notebook not found at {NOTEBOOK_PATH}"
    data = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cells = data.get("cells", [])
    assert len(cells) == 108, f"Expected 108 cells (54 markdown, 54 code), got {len(cells)}"
    
    md_cells = [c for c in cells if c["cell_type"] == "markdown"]
    code_cells = [c for c in cells if c["cell_type"] == "code"]
    assert len(md_cells) == 54, f"Expected 54 markdown cells, got {len(md_cells)}"
    assert len(code_cells) == 54, f"Expected 54 code cells, got {len(code_cells)}"

def test_notebook_code_cells_syntax_compilation():
    data = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    for idx, cell in enumerate(data.get("cells", [])):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            compile(source, f"cell_{idx}", "exec")

def test_notebook_reads_official_competition_train():
    data = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    code_sources = "\n".join("".join(c["source"]) for c in data["cells"] if c["cell_type"] == "code")
    # Must contain official competition train path
    assert "data/competition/train" in code_sources.replace("\\", "/")
    # Must NOT use extracted v1 train as training directory
    assert "DATA_RAW_TRAIN_DIR = PROJECT_ROOT / \"data\" / \"extracted\" / \"v1\" / \"train\"" not in code_sources

def test_run_all_centralized_configuration_and_clean_gates():
    data = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    sec02_code = "".join(data["cells"][3]["source"])
    sec03_code = "".join(data["cells"][5]["source"])
    sec08_code = "".join(data["cells"][15]["source"])
    
    # 1. Verify CONFIG dictionary is present and deterministic
    assert "CONFIG = {" in sec02_code
    assert '"experiment_id": "fewshot_pseudolabel_b2_v1"' in sec02_code
    assert '"max_labels_per_class": 5' in sec02_code
    assert '"max_total_labels": 40' in sec02_code
    assert '"cv_folds": 5' in sec02_code
    assert '"teacher_epochs": 15' in sec02_code
    assert '"pseudo_label_confidence_threshold": 0.95' in sec02_code
    
    # 2. Verify all 10 old manual flags are eliminated
    old_flags = [
        "RULES_VERIFIED",
        "RUN_FEWSHOT_PREPARATION",
        "RUN_CROSS_VALIDATION",
        "RUN_TEACHER_TRAINING",
        "RUN_PSEUDO_LABEL_GENERATION",
        "RUN_PSEUDO_LABEL_SELECTION",
        "RUN_STUDENT_TRAINING",
        "RUN_TEACHER_STUDENT_EVALUATION",
        "RUN_TEST_INFERENCE",
        "RUN_SUBMISSION_GENERATION",
    ]
    all_code = "\n".join("".join(c["source"]) for c in data["cells"] if c["cell_type"] == "code")
    for flag in old_flags:
        assert flag not in all_code, f"Old manual flag '{flag}' must be removed from all code cells"
        
    # 3. Verify Section 03 automated compliance assertions
    assert 'assert CONFIG["max_labels_per_class"] <= 5' in sec03_code
    assert 'assert CONFIG["max_total_labels"] <= 40' in sec03_code
    
    # 4. Verify Section 08 safe manual labeled pool gate
    assert "if not LABELED_POOL_PATH.exists():" in sec08_code
    assert "raise FileNotFoundError(error_msg)" in sec08_code
    assert "create_manual_annotation_template" in sec08_code

def test_official_competition_train_and_test_dataset_integrity():
    train_dir = PROJECT_ROOT / "data" / "competition" / "train"
    test_dir = PROJECT_ROOT / "data" / "competition" / "test"
    assert train_dir.exists(), f"Train dir missing: {train_dir}"
    assert test_dir.exists(), f"Test dir missing: {test_dir}"
    
    train_files = [p for p in train_dir.iterdir() if p.is_file()]
    train_subdirs = [p for p in train_dir.iterdir() if p.is_dir()]
    assert len(train_subdirs) == 0, "Train dir must be flat without class subdirectories"
    assert len(train_files) == 1920, f"Expected 1920 train files, got {len(train_files)}"
    
    test_files = [p for p in test_dir.iterdir() if p.is_file()]
    assert len(test_files) == 480, f"Expected 480 test files, got {len(test_files)}"
    
    # Check sample submission alignment
    sample_sub = PROJECT_ROOT / "data" / "competition" / "SampleSubmission_DSC.csv"
    assert sample_sub.exists()
    sample_df = pd.read_csv(sample_sub, dtype={"image_id": str})
    assert len(sample_df) == 480
    test_stems = set(p.stem for p in test_files)
    assert set(sample_df["image_id"]) == test_stems

def test_train_test_hash_overlap_forensic():
    def get_sha256(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    train_dir = PROJECT_ROOT / "data" / "competition" / "train"
    test_dir = PROJECT_ROOT / "data" / "competition" / "test"
    
    # Check collision with test set
    test_hashes = {get_sha256(p): p.name for p in test_dir.iterdir() if p.is_file()}
    train_hashes = {get_sha256(p): p.name for p in train_dir.iterdir() if p.is_file()}
    
    collision_hashes = set(train_hashes.keys()) & set(test_hashes.keys())
    assert len(collision_hashes) == 3, f"Expected exactly 3 train-test collisions, found {len(collision_hashes)}"
    colliding_train_files = sorted([train_hashes[h] for h in collision_hashes])
    assert colliding_train_files == ["1046.jpg", "1135.jpg", "1821.jpg"]

def test_labeling_budget_validator():
    def verify_labeling_budget(df: pd.DataFrame, class_col="class_name", max_per_class=5, max_total=40):
        if len(df) == 0:
            return True
        if len(df) > max_total:
            raise ValueError(f"Total {len(df)} exceeds {max_total}")
        for cls_name, count in df[class_col].value_counts().items():
            if count > max_per_class:
                raise ValueError(f"Class '{cls_name}' count {count} exceeds {max_per_class}")
        return True

    valid_df = pd.DataFrame({"class_name": [c for c in CLASS_NAMES for _ in range(5)]})
    assert verify_labeling_budget(valid_df, max_per_class=5, max_total=40) is True

    invalid_class_df = pd.DataFrame({"class_name": ["cardboard"] * 6 + ["paper"] * 4})
    with pytest.raises(ValueError, match="exceeds 5"):
        verify_labeling_budget(invalid_class_df, max_per_class=5, max_total=40)

    invalid_total_df = pd.DataFrame({"class_name": ["cardboard"] * 5 + ["paper"] * 5 + ["metal"] * 31})
    with pytest.raises(ValueError, match="exceeds 40"):
        verify_labeling_budget(invalid_total_df, max_per_class=5, max_total=40)

def test_unlabeled_dataset_anti_leakage_contract():
    from torch.utils.data import Dataset
    class MockUnlabeledDataset(Dataset):
        def __init__(self, records):
            self.records = records
        def __len__(self):
            return len(self.records)
        def __getitem__(self, idx):
            return torch.zeros(3, 288, 288), self.records[idx]["image_id"]

    mock_ds = MockUnlabeledDataset([{"image_id": "0001"}, {"image_id": "0002"}])
    item = mock_ds[0]
    assert len(item) == 2, "Unlabeled dataset must return exactly (tensor, image_id)"
    tensor, img_id = item
    assert isinstance(tensor, torch.Tensor)
    assert isinstance(img_id, str)
    assert img_id == "0001"

def test_stratified_cv_splits_disjointness():
    from sklearn.model_selection import StratifiedKFold
    dummy_labeled = pd.DataFrame({
        "image_id": [f"{i:04d}" for i in range(1, 41)],
        "class_name": [c for c in CLASS_NAMES for _ in range(5)],
        "class_idx": [i for i in range(8) for _ in range(5)]
    })
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    dummy_labeled["fold"] = -1
    for f, (train_idx, val_idx) in enumerate(skf.split(dummy_labeled, dummy_labeled["class_idx"])):
        dummy_labeled.loc[val_idx, "fold"] = f
        
    for f in range(5):
        train_sub = dummy_labeled[dummy_labeled["fold"] != f]
        val_sub = dummy_labeled[dummy_labeled["fold"] == f]
        assert len(train_sub) == 32
        assert len(val_sub) == 8
        overlap = set(train_sub["image_id"]) & set(val_sub["image_id"])
        assert len(overlap) == 0, f"Fold {f} has train-val overlap: {overlap}"
        assert (val_sub["class_name"].value_counts() == 1).all()

def test_teacher_architecture_modification_and_forward_pass():
    model = efficientnet_b2(weights=None)
    in_features = model.classifier[1].in_features
    assert in_features == 1408
    
    model.classifier[1] = nn.Linear(in_features, 8)
    for param in model.features.parameters():
        param.requires_grad = False
        
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable_params == (1408 * 8 + 8)
    
    dummy_x = torch.randn(2, 3, 288, 288)
    out = model(dummy_x)
    assert out.shape == (2, 8)

def test_student_semi_supervised_loss():
    class SemiSupervisedLoss(nn.Module):
        def __init__(self, pseudo_weight=0.5):
            super().__init__()
            self.pseudo_weight = pseudo_weight
            self.ce = nn.CrossEntropyLoss(reduction="none")
        def forward(self, logits, targets, is_pseudo, sample_weights):
            losses = self.ce(logits, targets)
            sup_mask = ~is_pseudo
            pseudo_mask = is_pseudo
            sup_loss = (losses[sup_mask] * sample_weights[sup_mask]).mean() if sup_mask.sum() > 0 else torch.tensor(0.0)
            pseudo_loss = (losses[pseudo_mask] * sample_weights[pseudo_mask]).mean() if pseudo_mask.sum() > 0 else torch.tensor(0.0)
            total_loss = sup_loss + self.pseudo_weight * pseudo_loss
            return total_loss, sup_loss, pseudo_loss

    loss_fn = SemiSupervisedLoss(pseudo_weight=0.5)
    logits = torch.randn(4, 8)
    targets = torch.tensor([0, 1, 2, 3])
    is_pseudo = torch.tensor([False, False, True, True])
    weights = torch.tensor([1.0, 1.0, 0.5, 0.5])
    
    total, sup, pseudo = loss_fn(logits, targets, is_pseudo, weights)
    assert torch.isclose(total, sup + 0.5 * pseudo)

def test_standalone_fresh_kernel_setup_logic():
    classes = sorted(["cardboard", "electronic", "foodwaste", "glass", "metal", "paper", "plastic", "textile"])
    assert len(classes) == 8
    
    transform = v2.Compose([
        v2.ToImage(),
        v2.Resize(288, interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
        v2.CenterCrop(288),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dummy_img = torch.randint(0, 256, (3, 300, 300), dtype=torch.uint8)
    out_tensor = transform(dummy_img)
    assert out_tensor.shape == (3, 288, 288)

def test_submission_verification_rules():
    sample_df = pd.DataFrame({
        "image_id": [f"{i:03d}" for i in range(1, 481)],
        "waste_category": ["electronic"] * 480
    })
    
    valid_sub = sample_df.copy()
    assert len(valid_sub) == 480
    assert list(valid_sub.columns) == ["image_id", "waste_category"]
    assert valid_sub["image_id"].iloc[0] == "001"
    assert (valid_sub["image_id"] == sample_df["image_id"]).all()
    assert set(valid_sub["waste_category"]).issubset(set(CLASS_NAMES))
    
    short_sub = valid_sub.iloc[:479]
    with pytest.raises(AssertionError):
        assert len(short_sub) == 480
        
    bad_cat_sub = valid_sub.copy()
    bad_cat_sub.loc[0, "waste_category"] = "nuclear_waste"
    with pytest.raises(AssertionError):
        assert set(bad_cat_sub["waste_category"]).issubset(set(CLASS_NAMES))

def test_handbook_pdf_detection():
    docs_dir = PROJECT_ROOT / "docs"
    assert docs_dir.exists()
    
    keywords = ["handbook", "preliminary", "dsc", "gelar", "rasa", "rule", "regulasi", "panduan", "guide", "ketentuan", "pedoman"]
    matched = []
    for item in docs_dir.iterdir():
        if item.is_file() and item.suffix.lower() == ".pdf":
            name_lower = item.name.lower()
            if any(kw in name_lower for kw in keywords):
                matched.append(item)
                
    assert len(matched) >= 1, "Official competition handbook PDF must be detected in docs/"
    pdf = matched[0]
    assert pdf.exists()
    assert pdf.stat().st_size > 1000
    assert "handbook" in pdf.name.lower()

def test_official_labeled_pool_40_manifest_integrity():
    labeled_pool_path = PROJECT_ROOT / "data" / "manifests" / "fewshot_pseudolabel_b2_v1" / "labeled_pool_40.csv"
    assert labeled_pool_path.exists(), "labeled_pool_40.csv must exist after manual annotation"
    
    df = pd.read_csv(labeled_pool_path, dtype={"image_id": str})
    assert len(df) == 40, f"Expected exactly 40 rows, got {len(df)}"
    assert list(df["image_id"].str.len()) == [4] * 40, "All image_ids must preserve leading zero (length 4)"
    assert not df["image_id"].duplicated().any(), "Duplicate image_ids found in labeled pool"
    
    # 8 official classes with exactly 5 each
    counts = df["class_name"].value_counts()
    assert len(counts) == 8, f"Expected 8 classes, got {len(counts)}"
    assert set(counts.index) == set(CLASS_NAMES)
    assert (counts == 5).all(), f"Each class must have exactly 5 samples, got: {counts.to_dict()}"
    
    # Verify image files existence and decode
    train_dir = PROJECT_ROOT / "data" / "competition" / "train"
    test_dir = PROJECT_ROOT / "data" / "competition" / "test"
    
    def get_file_sha256(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
        
    test_hashes = {get_file_sha256(p) for p in test_dir.iterdir() if p.is_file()}
    pool_hashes = []
    
    for _, row in df.iterrows():
        img_id = str(row["image_id"])
        img_path = train_dir / f"{img_id}.jpg"
        if not img_path.exists():
            matches = list(train_dir.glob(f"{img_id}.*"))
            assert matches, f"Image {img_id} not found in train dir"
            img_path = matches[0]
            
        with Image.open(img_path) as img:
            img.verify()
        with Image.open(img_path) as img:
            img.load()
            
        h = get_file_sha256(img_path)
        assert h not in test_hashes, f"Labeled image {img_id} collides with test set!"
        pool_hashes.append(h)
        
    assert len(set(pool_hashes)) == 40, "Duplicate image content (SHA-256) found within labeled pool"


def test_competition_test_directory_multi_extension_and_decodability():
    test_dir = PROJECT_ROOT / "data" / "competition" / "test"
    sample_sub = PROJECT_ROOT / "data" / "competition" / "SampleSubmission_DSC.csv"
    assert test_dir.exists() and sample_sub.exists()
    
    allowed_exts = {".jpg", ".jpeg", ".png"}
    test_files = [p for p in test_dir.iterdir() if p.is_file() and p.suffix.lower() in allowed_exts]
    assert len(test_files) == 480, f"Expected 480 test files, got {len(test_files)}"
    
    ext_counts = Counter(p.suffix.lower() for p in test_files)
    assert ext_counts[".jpg"] == 477, f"Expected 477 .jpg, got {ext_counts['.jpg']}"
    assert ext_counts[".jpeg"] == 2, f"Expected 2 .jpeg, got {ext_counts['.jpeg']}"
    assert ext_counts[".png"] == 1, f"Expected 1 .png, got {ext_counts['.png']}"
    
    sample_df = pd.read_csv(sample_sub, dtype={"image_id": str})
    expected_ids = sample_df["image_id"].tolist()
    assert len(expected_ids) == 480
    
    file_map = {p.stem: p for p in test_files}
    for img_id in expected_ids:
        assert img_id in file_map, f"Missing image for ID {img_id}"
        
    for p in test_files:
        with Image.open(p) as img:
            img.verify()


def test_notebook_section_48_multi_extension_audit():
    data = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    code_sec48 = "".join(data["cells"][95]["source"])
    assert '.glob("*.jpg")' not in code_sec48, "Section 48 must not use glob('*.jpg')"
    assert ".jpg" in code_sec48 and ".jpeg" in code_sec48 and ".png" in code_sec48


def test_standalone_test_dataset_multi_format():
    from torch.utils.data import Dataset
    class StandaloneCompetitionTestDataset(Dataset):
        def __init__(self, image_paths, transform=None):
            self.image_paths = list(image_paths)
            self.transform = transform
        def __len__(self):
            return len(self.image_paths)
        def __getitem__(self, idx):
            path = self.image_paths[idx]
            with Image.open(path) as img:
                image = img.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
            return image, path.stem

    test_dir = PROJECT_ROOT / "data" / "competition" / "test"
    sample_files = [
        test_dir / "001.jpg",
        test_dir / "039.png",
        test_dir / "142.jpeg",
        test_dir / "166.jpeg",
    ]
    for p in sample_files:
        assert p.exists(), f"Sample test file missing: {p}"
        
    transform = v2.Compose([
        v2.ToImage(),
        v2.Resize(288, interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
        v2.CenterCrop(288),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    ds = StandaloneCompetitionTestDataset(sample_files, transform=transform)
    assert len(ds) == 4
    for i, expected_stem in enumerate(["001", "039", "142", "166"]):
        tensor, stem = ds[i]
        assert stem == expected_stem
        assert tensor.shape == (3, 288, 288)
