"""
GridSearchCV parameter grids mirroring the manuscript Table 4.

Notes:
- For OCSVM, sklearn uses `nu` (equivalent to ϑ in the paper) and `gamma`.
- For CTGAN parameters, these are passed to the CTGAN class (ctgan package).
- For LSTM, SciKeras expects model hyperparameters prefixed with `model__`.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional

PARAM_GRIDS: Dict[str, Dict[str, Any]] = {
    "ocsvm": {
        "kernel": ["rbf", "linear", "sigmoid"],
        "nu": [0.03, 0.05, 0.07, 0.10],
        "gamma": [0.001, 0.01, 0.1, 1.0],
    },
    "tomek_knn": {
        "k": [3, 5, 7],
    },
    "rf": {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [None, 10, 15, 20],
        "min_samples_leaf": [1, 2],
        "min_samples_split": [2, 3, 5],
    },
    "ctgan": {
        "epochs": [150, 200, 300],
        "batch_size": [32, 64],
        "pac": [8, 10, 16],
        # dims are fixed in Table 4; included as singletons to keep the API uniform
        "generator_dim": [(256, 512, 256)],
        "discriminator_dim": [(256, 128, 64)],
    },
    "lstm": {
        "model__optimizer": ["adam"],
        "model__memory_cells": [50],
        # Dropout rate is not explicitly ranged in the table; keep default (or tune if desired)
        # "model__dropout_rate": [0.1, 0.2, 0.3],
        "epochs": [150],
        "batch_size": [32],
    },
}
