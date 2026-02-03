"""
GridSearchCV parameter.
Notes:
- For OCSVM, sklearn uses `nu` (equivalent to ϑ in the paper) and `gamma`.
- For CTGAN parameters, these are passed to the CTGAN class (ctgan package).
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
        "epochs": [100, 200, 300],
        "batch_size": [16, 32, 64],
    },
}
