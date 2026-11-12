from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import nbformat
import pytest
import torch
from PIL import Image
from torch import nn
from torchvision.models import EfficientNet_B2_Weights

from src.callbacks.early_stopping import EarlyStopping
from src.callbacks.model_checkpoint import load_checkpoint
from src.data.dataloader import load_manifest_records
from src.data.preprocessing import load_active_class_mapping
from src.inference.competition import audit_competition_test, load_submission_template
from src.inference.predict import Prediction
from src.inference.submission import validate_predictions, validate_submission_csv
from src.models.efficientnet_b2 import (
    FINETUNE_EXPERIMENT_ID, HEAD_EXPERIMENT_ID, MODEL_IDENTIFIER,
    apply_batch_norm_policy, build_efficientnet_b2, build_efficientnet_b2_transforms,
    build_fine_tuning_optimizer, count_parameters, configure_partial_fine_tuning,
    ensure_experiment_is_safe_to_start, freeze_for_head_training, head_training_config,
    load_efficientnet_b2_checkpoint, load_experiment_config, load_selected_checkpoint,
    save_selected_checkpoint, select_best_phase_checkpoint, validate_preprocessing, fine_tuning_config,
    validate_resume_configuration,
)

ROOT = Path(__file__).resolve().parents[1]


def test_notebook_structure_and_manual_flags() -> None:
    notebook = nbformat.read(ROOT / 'notebooks/04_efficientnet.ipynb', as_version=4)
    headings = [cell.source for cell in notebook.cells if cell.cell_type == 'markdown']
    assert len(headings) == 25
    assert all(f'## {index:02d}.' in heading for index, heading in enumerate(headings, 1))
    source = '\n'.join(cell.source for cell in notebook.cells if cell.cell_type == 'code')
    for flag in ('RUN_HEAD_TRAINING', 'RUN_FINE_TUNING', 'RUN_VALIDATION_EVALUATION', 'RUN_TEST_INFERENCE'):
        assert f'{flag} = False' in source
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            assert cell.execution_count is None and not cell.outputs
            compile(cell.source, 'efficientnet-notebook', 'exec')
    setup = notebook.cells[43].source
    assert 'sys.path.insert' in setup and 'load_selected_checkpoint' in setup
    assert 'INFERENCE_CONFIG = load_experiment_config' in setup
    assert '480' in setup and 'predict_torch_classifier(' not in setup


def test_real_dataset_and_template_audit() -> None:
    mapping = load_active_class_mapping(ROOT)
    assert list(mapping) == ['cardboard', 'electronic', 'foodwaste', 'glass', 'metal', 'paper', 'plastic', 'textile']
    assert len(load_manifest_records('train', ROOT)) == 1340
    assert len(load_manifest_records('val', ROOT)) == 383
    assert len(load_manifest_records('test_internal', ROOT)) == 191
    audit = audit_competition_test(project_root=ROOT, expected_count=480)
    assert audit.image_count == audit.readable_image_count == audit.sample_row_count == 480
    template = load_submission_template(ROOT / 'data/competition/SampleSubmission_DSC.csv')
    assert template.image_ids[0] == '001' and len(set(template.image_ids)) == 480
    assert template.columns == ('image_id', 'waste_category')
    assert load_experiment_config(ROOT)['model_type'] == 'efficientnet_b2'


def test_pretrained_constructor_and_failure_is_not_silent(monkeypatch) -> None:
    import src.models.efficientnet_b2 as module
    calls = []
    actual = module.efficientnet_b2
    def fake_constructor(*, weights):
        calls.append(weights)
        return actual(weights=None)
    monkeypatch.setattr(module, 'efficientnet_b2', fake_constructor)
    model = build_efficientnet_b2(8, pretrained=True)
    assert calls == [EfficientNet_B2_Weights.IMAGENET1K_V1]
    assert model.classifier[-1].out_features == 8
    assert model.classifier[0].p == 0.3
    assert count_parameters(model)['total'] == 7712266
    def failing_constructor(*, weights):
        raise OSError('unavailable')
    monkeypatch.setattr(module, 'efficientnet_b2', failing_constructor)
    with pytest.raises(module.PretrainedWeightsUnavailableError, match='not used'):
        build_efficientnet_b2(8, pretrained=True)


def test_freeze_bn_differential_lr_and_shapes() -> None:
    model = build_efficientnet_b2(8, pretrained=False)
    freeze_for_head_training(model)
    assert count_parameters(model)['trainable'] == 11272
    assert not any(parameter.requires_grad for parameter in model.features.parameters())
    model.train(); apply_batch_norm_policy(model, trainable_batch_norm=False)
    assert all(not layer.training for layer in model.features.modules() if isinstance(layer, nn.BatchNorm2d))
    assert configure_partial_fine_tuning(model) == ['features.7', 'features.8']
    model.train(); apply_batch_norm_policy(model, trainable_batch_norm=True)
    for layer in model.features.modules():
        if isinstance(layer, nn.BatchNorm2d):
            assert layer.training == any(parameter.requires_grad for parameter in layer.parameters())
    assert not any(parameter.requires_grad for parameter in model.features[0].parameters())
    optimizer = build_fine_tuning_optimizer(model)
    assert [group['lr'] for group in optimizer.param_groups] == [1e-5, 1e-4]
    model.eval()
    with torch.inference_mode():
        assert model(torch.zeros(1, 3, 288, 288)).shape == (1, 8)


def test_preprocessing_is_official_and_train_normalizes_once() -> None:
    training, validation, metadata = build_efficientnet_b2_transforms()
    sample = Image.new('RGB', (421, 300), (123, 45, 67))
    assert torch.equal(validation(sample), EfficientNet_B2_Weights.IMAGENET1K_V1.transforms()(sample))
    assert validate_preprocessing(validation)['shape'] == (3, 288, 288)
    assert training(sample).shape == (3, 288, 288)
    assert sum(type(transform).__name__ == 'Normalize' for transform in training.transforms) == 1
    assert metadata['normalization']['mean'] == [0.485, 0.456, 0.406]


def _checkpoint(root: Path, experiment: str, loss: float, mapping: dict[str, int]) -> Path:
    path = root / 'models/checkpoints' / experiment / 'best.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    model = build_efficientnet_b2(8, pretrained=False)
    torch.save({'model_state_dict': model.state_dict(), 'model_identifier': MODEL_IDENTIFIER,
                'class_mapping': mapping, 'best_metric': loss, 'best_epoch': 0, 'epoch': 0,
                'monitor': 'val_loss', 'mode': 'min', 'training_config': head_training_config() if experiment == HEAD_EXPERIMENT_ID else fine_tuning_config()}, path)
    return path


def test_checkpoint_selection_overwrite_guard_and_resume(tmp_path: Path) -> None:
    mapping = load_active_class_mapping(ROOT)
    (tmp_path / 'data/metadata').mkdir(parents=True)
    shutil.copy(ROOT / 'data/metadata/split_manifest.csv', tmp_path / 'data/metadata/split_manifest.csv')
    _checkpoint(tmp_path, HEAD_EXPERIMENT_ID, 0.3, mapping)
    with pytest.raises(FileExistsError): ensure_experiment_is_safe_to_start(tmp_path, HEAD_EXPERIMENT_ID)
    ensure_experiment_is_safe_to_start(tmp_path, HEAD_EXPERIMENT_ID, resume=True)
    assert select_best_phase_checkpoint(tmp_path).phase == 'head'
    save_selected_checkpoint(tmp_path)
    _checkpoint(tmp_path, FINETUNE_EXPERIMENT_ID, 0.5, mapping)
    assert load_selected_checkpoint(tmp_path).phase == 'head'
    path = _checkpoint(tmp_path, FINETUNE_EXPERIMENT_ID, 0.2, mapping)
    assert load_selected_checkpoint(tmp_path).phase == 'finetune'
    loaded = load_efficientnet_b2_checkpoint(project_root=tmp_path, checkpoint_path=path, expected_class_mapping=mapping)
    assert not loaded.model.training and loaded.model.classifier[-1].out_features == 8
    original = head_training_config(); changed = dict(original, batch_size=4)
    validate_resume_configuration(original, changed)
    with pytest.raises(RuntimeError): validate_resume_configuration(original, dict(original, learning_rate=2))
    stopper = EarlyStopping(patience=2, min_delta=0)
    assert not stopper.step(1.0, 0) and not stopper.step(1.2, 1) and stopper.step(1.3, 2)


def test_fresh_kernel_standalone_setup_without_training_state(tmp_path: Path, monkeypatch) -> None:
    for relative in ('configs/models', 'data/metadata', 'data/competition/test', 'notebooks'):
        (tmp_path / relative).mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / 'configs/models/efficientnet.yaml', tmp_path / 'configs/models/efficientnet.yaml')
    shutil.copy(ROOT / 'data/metadata/split_manifest.csv', tmp_path / 'data/metadata/split_manifest.csv')
    shutil.copy(ROOT / 'data/competition/SampleSubmission_DSC.csv', tmp_path / 'data/competition/SampleSubmission_DSC.csv')
    shutil.copytree(ROOT / 'src', tmp_path / 'src', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for index in range(1, 481):
        Image.new('RGB', (2, 2), (index % 255, 0, 0)).save(tmp_path / f'data/competition/test/{index:03}.jpg')
    _checkpoint(tmp_path, HEAD_EXPERIMENT_ID, 0.4, load_active_class_mapping(ROOT))
    notebook = nbformat.read(ROOT / 'notebooks/04_efficientnet.ipynb', as_version=4)
    monkeypatch.chdir(tmp_path / 'notebooks')
    namespace = {'__name__': '__main__'}
    exec(compile(notebook.cells[43].source, 'standalone', 'exec'), namespace)
    assert namespace['INFERENCE_ROOT'] == tmp_path
    assert namespace['INFERENCE_AUDIT'].image_count == 480
    assert namespace['INFERENCE_LOADED'].model.classifier[-1].out_features == 8
    assert 'model' not in namespace and 'optimizer' not in namespace and 'CONFIG' not in namespace
    assert not namespace['INFERENCE_OUTPUT'].exists()
    script = notebook.cells[43].source + '\nassert INFERENCE_AUDIT.image_count == 480\nassert "model" not in globals() and "optimizer" not in globals() and "CONFIG" not in globals()\n'
    completed = subprocess.run([sys.executable, '-c', script], cwd=tmp_path / 'notebooks', capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stderr


def test_template_schema_and_no_sample_label_leakage() -> None:
    template = load_submission_template(ROOT / 'data/competition/SampleSubmission_DSC.csv')
    assert not hasattr(template, 'categories')
    mapping = load_active_class_mapping(ROOT)
    predictions = [Prediction(image_id=identifier, waste_category='cardboard') for identifier in template.image_ids]
    validate_predictions(template, predictions, allowed_classes=mapping)
    with pytest.raises(ValueError): validate_predictions(template, list(reversed(predictions)), allowed_classes=mapping)
