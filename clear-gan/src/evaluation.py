
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

from .metrics import class_balance_accuracy, g_mean_score, mean_sensitivity, confusion_entropy


@dataclass
class RFCfg:
    n_estimators: int = 500
    random_state: int = 42
    n_jobs: int = -1


def evaluate_with_cv_rf(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    rf_cfg: Optional[RFCfg] = None,
) -> pd.DataFrame:
    """5-fold Stratified CV evaluation for RF (fixed bugs from notebook)."""
    cfg = rf_cfg or RFCfg()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.random_state)

    results: Dict[str, list] = {
        "Class Balance Accuracy": [],
        "G-Mean": [],
        "Mean Sensitivity": [],
        "CEN": [],
    }

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        model = RandomForestClassifier(
            n_estimators=cfg.n_estimators,
            random_state=cfg.random_state,
            n_jobs=cfg.n_jobs,
        )
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        results["Class Balance Accuracy"].append(class_balance_accuracy(y_te, y_pred))
        results["G-Mean"].append(g_mean_score(y_te, y_pred))
        results["Mean Sensitivity"].append(mean_sensitivity(y_te, y_pred))
        results["CEN"].append(confusion_entropy(y_te, y_pred))

    return pd.DataFrame(results).mean().to_frame(name="Average").T
