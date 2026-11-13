"""
Unit tests for tools/manual_fewshot_annotation.py.
Tests verify:
1. Candidate scanning and test-hash collision quarantine (3 images quarantined, 1917 candidates).
2. Per-class quota enforcement (<= 5 per class, refuses 6th).
3. Overall budget limit (<= 40 total).
4. Skip mechanics and skip reset.
5. Undo mechanics and count decrement.
6. Progress persistence and clean state recovery from JSON.
7. Strict CSV export format matching Section 08 notebook contract.
"""

import json
import shutil
import tempfile
from pathlib import Path
import pytest
import pandas as pd

from tools.manual_fewshot_annotation import (
    FewShotAnnotationEngine,
    CLASSES,
    CLASS_TO_IDX,
    MAX_PER_CLASS,
    MAX_TOTAL
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

@pytest.fixture
def temp_engine(tmp_path):
    """Fixture creating an engine using real train/test images but isolated temporary manifest directory."""
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    engine = FewShotAnnotationEngine(
        project_root=PROJECT_ROOT,
        manifest_dir=manifest_dir
    )
    return engine

def test_engine_candidate_scanning_and_quarantine(temp_engine):
    assert len(temp_engine.candidate_files) == 1917
    assert len(temp_engine.quarantined_files) == 3
    quarantined_names = {item[0] for item in temp_engine.quarantined_files}
    assert quarantined_names == {"1046.jpg", "1135.jpg", "1821.jpg"}
    
    # Verify quarantined files are never in candidate files
    candidate_names = {p.name for p in temp_engine.candidate_files}
    assert candidate_names.isdisjoint(quarantined_names)

def test_engine_single_annotation_and_counters(temp_engine):
    cand = temp_engine.get_current_candidate()
    assert cand is not None
    img_id = cand["image_id"]
    
    # Annotate with cardboard
    success, msg = temp_engine.annotate_current("cardboard")
    assert success is True
    assert img_id in temp_engine.annotations
    assert temp_engine.annotations[img_id]["class_name"] == "cardboard"
    assert temp_engine.get_total_count() == 1
    
    counts = temp_engine.get_class_counts()
    assert counts["cardboard"] == 1
    assert all(counts[c] == 0 for c in CLASSES if c != "cardboard")
    
    # Next candidate should not be the same image
    next_cand = temp_engine.get_current_candidate()
    assert next_cand["image_id"] != img_id

def test_engine_per_class_quota_enforcement(temp_engine):
    # Fill cardboard to 5
    for i in range(5):
        assert temp_engine.can_annotate_class("cardboard") is True
        success, _ = temp_engine.annotate_current("cardboard")
        assert success is True
        
    counts = temp_engine.get_class_counts()
    assert counts["cardboard"] == 5
    
    # 6th attempt must be strictly rejected
    assert temp_engine.can_annotate_class("cardboard") is False
    success, msg = temp_engine.annotate_current("cardboard")
    assert success is False
    assert "penuh" in msg.lower()
    assert temp_engine.get_class_counts()["cardboard"] == 5

def test_engine_skip_and_reset_skips(temp_engine):
    cand1 = temp_engine.get_current_candidate()
    id1 = cand1["image_id"]
    
    success, _ = temp_engine.skip_current()
    assert success is True
    assert id1 in temp_engine.skipped_ids
    
    cand2 = temp_engine.get_current_candidate()
    assert cand2["image_id"] != id1
    
    # Reset skips restores candidate availability
    temp_engine.reset_skips()
    assert len(temp_engine.skipped_ids) == 0

def test_engine_undo_mechanics(temp_engine):
    cand = temp_engine.get_current_candidate()
    img_id = cand["image_id"]
    
    temp_engine.annotate_current("electronic")
    assert temp_engine.get_class_counts()["electronic"] == 1
    assert temp_engine.get_total_count() == 1
    
    # Undo
    success, msg = temp_engine.undo_last()
    assert success is True
    assert temp_engine.get_class_counts()["electronic"] == 0
    assert temp_engine.get_total_count() == 0
    assert img_id not in temp_engine.annotations
    
    # Candidate pointer points back to undone image
    curr = temp_engine.get_current_candidate()
    assert curr["image_id"] == img_id

def test_engine_persistence_and_reload(tmp_path):
    manifest_dir = tmp_path / "manifests"
    engine1 = FewShotAnnotationEngine(project_root=PROJECT_ROOT, manifest_dir=manifest_dir)
    
    engine1.annotate_current("glass")
    engine1.annotate_current("metal")
    engine1.skip_current()
    
    assert engine1.get_total_count() == 2
    assert len(engine1.skipped_ids) == 1
    assert manifest_dir.joinpath("annotation_progress.json").exists()
    
    # New engine instance pointing to same manifest_dir
    engine2 = FewShotAnnotationEngine(project_root=PROJECT_ROOT, manifest_dir=manifest_dir)
    assert engine2.get_total_count() == 2
    assert engine2.get_class_counts()["glass"] == 1
    assert engine2.get_class_counts()["metal"] == 1
    assert len(engine2.skipped_ids) == 1

def test_engine_export_csv_validation_contract(temp_engine):
    # Incomplete export must fail
    success, msg = temp_engine.export_csv()
    assert success is False
    assert "Belum lengkap" in msg
    
    # Synthetically fill exactly 5 per class (40 total)
    for cname in CLASSES:
        for _ in range(5):
            cand = temp_engine.get_current_candidate()
            temp_engine.annotate_current(cname)
            
    assert temp_engine.is_complete() is True
    assert temp_engine.get_total_count() == 40
    
    # Export CSV
    export_csv_path = temp_engine.manifest_dir / "labeled_pool_40.csv"
    success, msg = temp_engine.export_csv(export_csv_path)
    assert success is True
    assert export_csv_path.exists()
    
    # Verify Section 08 notebook contract
    df = pd.read_csv(export_csv_path, dtype={"image_id": str})
    assert len(df) == 40
    assert list(df.columns) == ["image_id", "class_name", "class_idx", "relative_path", "sha256"]
    assert set(df["class_name"]) == set(CLASSES)
    assert (df["class_name"].value_counts() == 5).all()
    assert not df["image_id"].duplicated().any()
    
    # Verify no test hash collisions
    test_hashes = temp_engine.test_hashes
    for h in df["sha256"]:
        assert h not in test_hashes
