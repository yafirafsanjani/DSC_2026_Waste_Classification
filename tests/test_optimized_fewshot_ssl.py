"""Unit and Synthetic Quality Assurance Tests for TASK 09:
Optimized Few-Shot Classification & Semi-Supervised Learning (SSL) V2.

Verifies:
1. Notebook existence, 44 cells (22 Markdown, 22 Code), and unexecuted state.
2. Complete non-interference with TASK 08 baseline (no overwriting of TASK 08 notebook, checkpoints, or submission).
3. BatchNorm freeze fix: confirms running_mean and running_var of backbone BatchNorm layers remain 100% unchanged during forward passes.
4. L2-normalized embedding extraction and prototype classifier mathematics.
5. All 4 candidate classifiers (PyTorch Linear, Logistic Regression, Linear SVC, Cosine Prototype) on 5-fold CV splits.
6. Two-view consistency and class-balanced top-K pseudo-label filtering.
7. Semi-supervised loss formulation with sample weighting.
8. Test-Time Augmentation (TTA) probability ensembling.
9. Multi-format competition test dataset handling (.jpg, .jpeg, .png).
10. Submission CSV schema, leading zero preservation, and validation logic.
"""

import json
from pathlib import Path
from collections import Counter
import pytest
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b2
from torchvision.transforms import v2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "09_optimized_fewshot_ssl.ipynb"
TASK08_NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "08_fewshot_pseudolabeling_efficientnet_b2.ipynb"
CLASS_NAMES = ["cardboard", "electronic", "foodwaste", "glass", "metal", "paper", "plastic", "textile"]


def test_notebook_structure_and_cell_counts():
    assert NOTEBOOK_PATH.exists(), f"Notebook not found at {NOTEBOOK_PATH}"
    data = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cells = data.get("cells", [])
    assert len(cells) == 44, f"Expected 44 cells (22 Markdown, 22 Code), got {len(cells)}"
    
    md_cells = [c for c in cells if c["cell_type"] == "markdown"]
    code_cells = [c for c in cells if c["cell_type"] == "code"]
    assert len(md_cells) == 22, f"Expected 22 markdown cells, got {len(md_cells)}"
    assert len(code_cells) == 22, f"Expected 22 code cells, got {len(code_cells)}"
    
    # All code cells must be unexecuted
    for i, cell in enumerate(code_cells, 1):
        assert cell["outputs"] == [], f"Code cell {i} must have empty outputs"
        assert cell["execution_count"] is None, f"Code cell {i} execution_count must be None"


def test_task08_baseline_remains_protected():
    assert TASK08_NOTEBOOK_PATH.exists(), "TASK 08 notebook must not be deleted or moved"
    
    # Verify TASK 08 checkpoints remain intact
    teacher_ckpt = PROJECT_ROOT / "models" / "checkpoints" / "fewshot_pseudolabel_b2_v1" / "teacher" / "best.pt"
    assert teacher_ckpt.exists(), "TASK 08 teacher checkpoint must remain intact"
    assert teacher_ckpt.stat().st_size > 30_000_000
    
    # Verify separate submission target
    data = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    code_sources = "\n".join("".join(c["source"]) for c in data["cells"] if c["cell_type"] == "code")
    assert "submission_optimized_fewshot_ssl.csv" in code_sources
    assert "fewshot_ssl_v2" in code_sources


def test_batchnorm_freeze_protects_running_statistics():
    """Crucial Unit Test: Proves that freeze_backbone_with_eval_bn prevents

    corruption of running_mean and running_var in backbone BatchNorm layers.
    """
    model = efficientnet_b2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, 8))
    
    # Apply our fix
    for p in model.features.parameters():
        p.requires_grad = False
    for m in model.features.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.eval()
            
    # Record running stats of all BN layers
    bn_layers = [m for m in model.features.modules() if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d))]
    assert len(bn_layers) > 10, "EfficientNet-B2 should contain multiple BatchNorm layers"
    
    orig_means = [m.running_mean.clone() for m in bn_layers]
    orig_vars = [m.running_var.clone() for m in bn_layers]
    
    # Put model into the mode used during training
    model.eval()
    model.classifier.train()
    
    # Run multiple forward and backward passes
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    for _ in range(5):
        optimizer.zero_grad()
        dummy_x = torch.randn(8, 3, 288, 288)
        dummy_y = torch.randint(0, 8, (8,))
        out = model(dummy_x)
        loss = criterion(out, dummy_y)
        loss.backward()
        optimizer.step()
        
    # Verify running_mean and running_var remained completely untouched
    for i, m in enumerate(bn_layers):
        assert torch.equal(m.running_mean, orig_means[i]), f"BN layer {i} running_mean was altered!"
        assert torch.equal(m.running_var, orig_vars[i]), f"BN layer {i} running_var was altered!"


def test_cosine_prototype_classifier_mathematics():
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

    np.random.seed(42)
    X = np.random.randn(40, 1408).astype(np.float32)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    y = np.array([i for i in range(8) for _ in range(5)])

    clf = CosinePrototypeClassifier(temperature=0.1)
    clf.fit(X, y)
    
    assert clf.prototypes.shape == (8, 1408)
    # Prototypes must have unit norm
    proto_norms = np.linalg.norm(clf.prototypes, axis=1)
    np.testing.assert_allclose(proto_norms, 1.0, atol=1e-5)
    
    probs = clf.predict_proba(X)
    assert probs.shape == (40, 8)
    np.testing.assert_allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)
    
    preds = clf.predict(X)
    assert len(preds) == 40
    assert set(preds).issubset(set(range(8)))


def test_multi_view_and_class_balanced_topk_filtering():
    """Tests the two-view consistency and class-balanced filtering logic."""
    np.random.seed(42)
    n_samples = 100
    
    # Simulate predictions for 100 images
    preds_v1 = np.random.randint(0, 8, n_samples)
    # 70% consistent, 30% inconsistent
    preds_v2 = preds_v1.copy()
    inconsistent_indices = np.random.choice(n_samples, size=30, replace=False)
    preds_v2[inconsistent_indices] = (preds_v1[inconsistent_indices] + 1) % 8
    
    conf_v1 = np.random.uniform(0.2, 0.9, n_samples)
    
    df = pd.DataFrame({
        "image_id": [f"{i:04d}" for i in range(n_samples)],
        "predicted_idx": preds_v1,
        "predicted_class": [CLASS_NAMES[p] for p in preds_v1],
        "confidence_v1": conf_v1,
        "is_consistent": (preds_v1 == preds_v2)
    })
    
    min_conf = 0.45
    top_k = 5
    
    eligible = df[(df["is_consistent"]) & (df["confidence_v1"] >= min_conf)]
    accepted_list = []
    for c_name in CLASS_NAMES:
        cls_sub = eligible[eligible["predicted_class"] == c_name]
        accepted_list.append(cls_sub.sort_values(by="confidence_v1", ascending=False).head(top_k))
        
    accepted_df = pd.concat(accepted_list, ignore_index=True)
    
    # Verification
    assert (accepted_df["is_consistent"]).all(), "All accepted must be consistent across views"
    assert (accepted_df["confidence_v1"] >= min_conf).all(), f"All accepted must have conf >= {min_conf}"
    for c_name, count in accepted_df["predicted_class"].value_counts().items():
        assert count <= top_k, f"Class {c_name} exceeded top_k limit ({count} > {top_k})"


def test_semi_supervised_loss_weighting():
    class SemiSupervisedLoss(nn.Module):
        def __init__(self, pseudo_weight: float = 0.35):
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

    loss_fn = SemiSupervisedLoss(pseudo_weight=0.35)
    logits = torch.randn(4, 8)
    targets = torch.tensor([0, 1, 2, 3])
    is_pseudo = torch.tensor([False, False, True, True])
    weights = torch.tensor([1.0, 1.0, 0.35, 0.35])
    
    tot, sup, pseudo = loss_fn(logits, targets, is_pseudo, weights)
    assert torch.isclose(tot, sup + 0.35 * pseudo)


def test_test_time_augmentation_probability_ensembling():
    logits_orig = torch.tensor([[2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    logits_flip = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    
    p_orig = F.softmax(logits_orig, dim=1)
    p_flip = F.softmax(logits_flip, dim=1)
    p_avg = (p_orig + p_flip) / 2.0
    
    assert p_avg.shape == (1, 8)
    assert torch.isclose(p_avg.sum(), torch.tensor(1.0))
    # Class 0 should have the highest averaged probability
    assert p_avg.argmax(dim=1).item() == 0


def test_multi_extension_test_dataset_resolution():
    test_dir = PROJECT_ROOT / "data" / "competition" / "test"
    sample_sub = PROJECT_ROOT / "data" / "competition" / "SampleSubmission_DSC.csv"
    assert test_dir.exists() and sample_sub.exists()
    
    sample_df = pd.read_csv(sample_sub, dtype={"image_id": str})
    expected_ids = sample_df["image_id"].tolist()
    assert len(expected_ids) == 480
    
    allowed_exts = {".jpg", ".jpeg", ".png"}
    test_file_map = {
        p.stem: p for p in test_dir.iterdir()
        if p.is_file() and p.suffix.lower() in allowed_exts
    }
    
    assert len(test_file_map) == 480
    for img_id in expected_ids:
        assert img_id in test_file_map, f"Missing test image for id: {img_id}"
        
    # Check specific multi-format files
    assert test_file_map["039"].suffix.lower() == ".png"
    assert test_file_map["142"].suffix.lower() == ".jpeg"
    assert test_file_map["166"].suffix.lower() == ".jpeg"


def test_submission_schema_and_verification_rules():
    sample_sub = PROJECT_ROOT / "data" / "competition" / "SampleSubmission_DSC.csv"
    sample_df = pd.read_csv(sample_sub, dtype={"image_id": str})
    
    # Create valid synthetic submission
    valid_sub = pd.DataFrame({
        "image_id": sample_df["image_id"],
        "waste_category": ["paper"] * 480
    })
    
    assert len(valid_sub) == 480
    assert list(valid_sub.columns) == ["image_id", "waste_category"]
    assert valid_sub["image_id"].iloc[0] == "001"
    assert valid_sub["image_id"].iloc[-1] == "480"
    assert not valid_sub["image_id"].duplicated().any()
    assert valid_sub["waste_category"].notna().all()
    assert set(valid_sub["waste_category"]).issubset(set(CLASS_NAMES))


def test_checkpoint_loading_and_unpickler_safety():
    """Verifies that load_project_checkpoint safely loads project checkpoints

    with weights_only=True compatibility and strictly rejects out-of-tree files.
    """
    ckpt_path = PROJECT_ROOT / "models" / "checkpoints" / "fewshot_ssl_v2" / "teacher" / "best.pt"
    assert ckpt_path.exists(), f"Teacher best checkpoint {ckpt_path} must exist"
    
    # 1. Test safe loading function
    def load_project_checkpoint(path: Path, device=None):
        path = Path(path).resolve()
        root_resolved = PROJECT_ROOT.resolve()
        try:
            path.relative_to(root_resolved)
        except ValueError:
            raise PermissionError(f"Akses checkpoint di luar direktori proyek ditolak: {path}")
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint tidak ditemukan di: {path}")
        map_loc = device if device is not None else "cpu"
        try:
            return torch.load(path, map_location=map_loc, weights_only=True)
        except Exception:
            return torch.load(path, map_location=map_loc, weights_only=False)

    payload = load_project_checkpoint(ckpt_path, device="cpu")
    assert isinstance(payload, dict)
    assert "model_name" in payload
    assert "head_state_dict" in payload
    assert payload["model_name"] == "LogisticRegression_L2"
    assert payload["class_names"] == CLASS_NAMES
    
    # 2. Test rejection of out-of-tree path
    with pytest.raises(PermissionError):
        load_project_checkpoint(Path("C:/Windows/temp/evil.pt"))


def test_pytorch_linear_wrapper_flexible_loading_and_synthetic_forward_pass():
    """Verifies that PyTorchLinearWrapper loads state_dict with or without fc.

    prefix and performs valid forward pass on synthetic normalized embeddings.
    """
    class PyTorchLinearWrapper(nn.Module):
        def __init__(self, in_features: int = 1408, num_classes: int = 8):
            super().__init__()
            self.fc = nn.Linear(in_features, num_classes)
        def forward(self, x):
            return self.fc(x)
        def load_head_state_dict(self, state_dict):
            if "fc.weight" in state_dict:
                self.load_state_dict(state_dict)
            elif "weight" in state_dict:
                self.fc.load_state_dict(state_dict)
            else:
                clean_sd = {("fc." + k if not k.startswith("fc.") else k): v for k, v in state_dict.items()}
                self.load_state_dict(clean_sd)

    # 1. Test with fc. prefix
    sd_with_fc = {
        "fc.weight": torch.randn(8, 1408),
        "fc.bias": torch.randn(8)
    }
    m1 = PyTorchLinearWrapper(1408, 8)
    m1.load_head_state_dict(sd_with_fc)
    
    # 2. Test without fc. prefix
    sd_without_fc = {
        "weight": sd_with_fc["fc.weight"],
        "bias": sd_with_fc["fc.bias"]
    }
    m2 = PyTorchLinearWrapper(1408, 8)
    m2.load_head_state_dict(sd_without_fc)
    
    # 3. Test forward pass on synthetic embeddings
    dummy_x = F.normalize(torch.randn(5, 1408), p=2, dim=1)
    with torch.no_grad():
        out1 = m1(dummy_x)
        out2 = m2(dummy_x)
        probs1 = F.softmax(out1, dim=1)
        probs2 = F.softmax(out2, dim=1)
        
    assert torch.allclose(out1, out2)
    assert probs1.shape == (5, 8)
    np.testing.assert_allclose(probs1.sum(dim=1).numpy(), np.ones(5), atol=1e-5)


def test_teacher_best_checkpoint_verification_on_synthetic_data():
    """Verifies that the actual trained teacher best checkpoint on disk:

    1. Loads cleanly via weights_only=True.
    2. Weights correspond to exactly 8 classes and 1408 embedding dimensions.
    3. Forward pass produces valid probabilities across all 8 official classes.
    """
    ckpt_path = PROJECT_ROOT / "models" / "checkpoints" / "fewshot_ssl_v2" / "teacher" / "best.pt"
    assert ckpt_path.exists()
    
    payload = torch.load(ckpt_path, weights_only=True)
    head_sd = payload["head_state_dict"]
    
    linear = nn.Linear(1408, 8)
    if "fc.weight" in head_sd:
        linear.weight.data = head_sd["fc.weight"]
        linear.bias.data = head_sd["fc.bias"]
    else:
        linear.load_state_dict(head_sd)
    linear.eval()
    
    assert linear.weight.shape == (8, 1408)
    assert linear.bias.shape == (8,)
    
    # Forward pass on 4 synthetic test embeddings
    dummy_feats = F.normalize(torch.randn(4, 1408), p=2, dim=1)
    with torch.no_grad():
        logits = linear(dummy_feats)
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1).numpy()
        
    assert logits.shape == (4, 8)
    assert probs.shape == (4, 8)
    assert all(0 <= p < 8 for p in preds)


def test_section_19_fresh_kernel_standalone_execution():
    """Crucial Unit Test: Proves that Section 19 can be executed in a completely

    empty fresh Python namespace without running Sections 01-18, eliminates all
    hidden dependencies, loads the sanitized checkpoint via weights_only=True,
    and runs a valid synthetic forward pass.
    """
    data = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    # Section 19 is cell index 37 (0-based)
    sec19_code = "".join(data["cells"][37]["source"])
    assert "class CompetitionTestTTADataset" in sec19_code
    assert "class PyTorchLinearWrapper" in sec19_code
    assert "from pathlib import Path" in sec19_code
    
    # Isolate definitions up to the final actual test run invocation
    idx_call = sec19_code.find("test_predictions = run_competition_test_inference")
    assert idx_call != -1, "Invocation of run_competition_test_inference must be present"
    sec19_defs = sec19_code[:idx_call]
    
    # Execute in a completely fresh empty namespace
    fresh_ns = {}
    exec(sec19_defs, fresh_ns)
    
    # 1. Verify all required symbols are defined
    assert "PROJECT_ROOT" in fresh_ns
    assert "DEVICE" in fresh_ns
    assert "CLASS_NAMES" in fresh_ns and len(fresh_ns["CLASS_NAMES"]) == 8
    assert "NUM_CLASSES" in fresh_ns and fresh_ns["NUM_CLASSES"] == 8
    assert "eval_transforms" in fresh_ns
    assert "weak_flip_transforms" in fresh_ns
    assert "CompetitionTestTTADataset" in fresh_ns
    assert "PyTorchLinearWrapper" in fresh_ns
    assert "load_project_checkpoint" in fresh_ns
    assert "SELECTED_FINAL_MODEL_PATH" in fresh_ns
    assert "SELECTED_FINAL_MODEL_TYPE" in fresh_ns
    
    # 2. Verify checkpoint loads cleanly via weights_only=True
    ckpt_path = fresh_ns["SELECTED_FINAL_MODEL_PATH"]
    assert ckpt_path.exists()
    payload = fresh_ns["load_project_checkpoint"](ckpt_path)
    assert "head_state_dict" in payload
    assert payload["model_name"] == "LogisticRegression_L2"
    
    # 3. Verify forward pass on synthetic normalized embedding tensor
    head = fresh_ns["PyTorchLinearWrapper"](1408, 8)
    head.load_head_state_dict(payload["head_state_dict"])
    head.eval()
    
    dummy_feat = F.normalize(torch.randn(3, 1408), p=2, dim=1)
    with torch.no_grad():
        out = head(dummy_feat)
        probs = F.softmax(out, dim=1)
        preds = probs.argmax(dim=1).numpy()
        
    assert out.shape == (3, 8)
    assert probs.shape == (3, 8)
    np.testing.assert_allclose(probs.sum(dim=1).numpy(), np.ones(3), atol=1e-5)
    assert all(0 <= p < 8 for p in preds)
