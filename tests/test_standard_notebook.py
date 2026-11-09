from __future__ import annotations

import os
import sys
from pathlib import Path

import nbformat


def test_cnn_v2_notebook_has_unexecuted_standard_workflow_and_standalone_setup() -> None:
    root = Path.cwd().resolve()
    notebook = nbformat.read(root / "notebooks" / "01b_cnn_from_scratch_v2.ipynb", as_version=4)
    assert len(notebook.cells) == 20
    code_sources = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    for flag in ("RUN_TRAINING = False", "RUN_VALIDATION_EVALUATION = False", "RUN_TEST_INFERENCE = False"):
        assert flag in code_sources
    for required_term in ("history_review", "plot_learning_curves", "validate_one_epoch", "CompetitionTestDataset", "write_submission_csv", "validate_submission_csv"):
        assert required_term in code_sources
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is None and not cell.outputs
            compile(cell.source, "cnn-v2-notebook-cell", "exec")


def test_cnn_v2_standalone_setup_runs_from_notebooks_without_training_state(monkeypatch) -> None:
    root = Path.cwd().resolve()
    notebook = nbformat.read(root / "notebooks" / "01b_cnn_from_scratch_v2.ipynb", as_version=4)
    setup_source = notebook.cells[16].source
    monkeypatch.chdir(root / "notebooks")
    original_path = list(sys.path)
    try:
        sys.path[:] = [entry for entry in sys.path if Path(entry or Path.cwd()).resolve() != root]
        namespace: dict[str, object] = {"__name__": "__main__"}
        exec(compile(setup_source, "cnn-v2-standalone-setup", "exec"), namespace)
        assert str(root) in sys.path
        assert namespace["INFERENCE_EXPERIMENT_ID"] == "cnn_scratch_v2"
        assert namespace["INFERENCE_AUDIT"].image_count == 480
        assert namespace["INFERENCE_AUDIT"].sample_row_count == 480
        assert namespace["INFERENCE_TRANSFORM"].config == namespace["INFERENCE_VALIDATION_TRANSFORM"].config
        assert namespace["INFERENCE_TRANSFORM"].augmentation is None
        assert namespace["INFERENCE_CHECKPOINT_METADATA"] is None
        assert not (root / "Final Dataset for Submission to Kaggle" / "submission_cnn_v2.csv").exists()
    finally:
        sys.path[:] = original_path
        os.chdir(root)
