# ==============================================================================
# SECTION 02: ENVIRONMENT SETUP, DEPENDENCIES & MANUAL FLAGS
# ==============================================================================
import os
import sys
import json
import time
import math
import random
import shutil
import hashlib
import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Sequence, Set, Union, Iterable, Iterator, Mapping, Type
from dataclasses import dataclass

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import sklearn.metrics as metrics
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
import torchvision.transforms.functional as TF

try:
    from torchvision.transforms import v2
    USE_TRANSFORMS_V2 = True
except ImportError:
    import torchvision.transforms as v2
    USE_TRANSFORMS_V2 = False

from tqdm.auto import tqdm

# --- REPRODUCIBILITY SEED SETUP ---
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

# --- HARDWARE ACCELERATION DETECTION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()

print("STATUS ENVIRONMENT:")
print(f"  Python Version      : {sys.version.split()[0]}")
print(f"  PyTorch Version     : {torch.__version__}")
print(f"  Torchvision Version : {torchvision.__version__}")
print(f"  Device              : {DEVICE}")
if torch.cuda.is_available():
    print(f"  GPU Device Name     : {torch.cuda.get_device_name(0)}")
    print(f"  GPU VRAM Total      : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
print(f"  Mixed Precision     : {'Enabled (AMP)' if USE_AMP else 'Disabled (FP32)'}")
print(f"  Transforms Engine   : {'Torchvision v2' if USE_TRANSFORMS_V2 else 'Torchvision v1'}")

# ==============================================================================

# ==============================================================================
# CENTRALIZED EXPERIMENT CONFIGURATION (DETERMINISTIC RUN-ALL)
# ==============================================================================
CONFIG = {
    "experiment_id": "fewshot_pseudolabel_b2_v1",
    "random_seed": 42,
    "image_size": 288,
    "batch_size": 8,
    "num_workers": 0,
    "max_labels_per_class": 5,
    "max_total_labels": 40,
    "cv_folds": 5,
    "teacher_epochs": 15,
    "teacher_lr": 1e-3,
    "teacher_patience": 5,
    "pseudo_label_confidence_threshold": 0.95,
    "pseudo_label_loss_weight": 0.5,
    "student_epochs": 25,
    "student_lr": 1e-4,
    "student_patience": 7,
    "mixed_precision": USE_AMP,
    "device": str(DEVICE)
}

print("\n" + "=" * 75)
print("KONFIGURASI EKSPERIMEN TERPUSAT (CONFIG):")
for k, v in CONFIG.items():
    print(f"  {k:35s}: {v}")
print("=" * 75)