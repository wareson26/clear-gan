
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from .metrics import class_balance_accuracy, g_mean_score, mean_sensitivity, confusion_entropy
from .models import make_lstm_estimator, to_lstm_3d


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
    """5-fold Stratified CV evaluation for RF."""
    cfg = rf_cfg if rf_cfg is not None else RFCfg()
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


# --------------------------
# LSTM evaluation (CV)
# --------------------------

@dataclass
class LSTMCfg:
    random_state: int = 42
    epochs: int = 150
    batch_size: int = 32
    
    # For the prepriotry dataset, the timesteps=8, while tabular datasets,timesteps=1 and n_features=X.shape[1].
    timesteps: int = 1


def evaluate_with_cv_lstm(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    lstm_cfg: Optional[LSTMCfg] = None,
    n_features: Optional[int] = None,
) -> pd.DataFrame:
    """
    5-fold Stratified CV evaluation for LSTM (SciKeras wrapper).
    - X: features (DataFrame)
    - y: integer-encoded labels
    """
    cfg = lstm_cfg if lstm_cfg is not None else LSTMCfg()

    y_arr = np.asarray(y)
    if not np.issubdtype(y_arr.dtype, np.integer):
        # If labels are strings/categories, convert
        y_arr = pd.Series(y_arr).astype("category").cat.codes.values

    X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)

    if n_features is None:
        if cfg.timesteps == 1:
            n_features = X_arr.shape[1]
        else:
            if X_arr.shape[1] % cfg.timesteps != 0:
                raise ValueError(
                    f"X has {X_arr.shape[1]} columns, not divisible by timesteps={cfg.timesteps}."
                )
            n_features = X_arr.shape[1] // cfg.timesteps

    # Convert to 3D for LSTM: (n_samples, timesteps, n_features)
    X_3d = to_lstm_3d(X_arr, timesteps=cfg.timesteps, n_features=n_features)

    n_classes = int(np.unique(y_arr).shape[0])
    input_shape: Tuple[int, int] = (cfg.timesteps, n_features)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.random_state)

    results: Dict[str, list] = {
        "Class Balance Accuracy": [],
        "G-Mean": [],
        "Mean Sensitivity": [],
        "CEN": [],
    }

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_3d, y_arr), start=1):
        X_tr, X_te = X_3d[train_idx], X_3d[test_idx]
        y_tr, y_te = y_arr[train_idx], y_arr[test_idx]

        # Build an estimator per fold (prevents weight carry-over)
        lstm_est = make_lstm_estimator(
            input_shape=input_shape,
            n_classes=n_classes,
            random_state=cfg.random_state,
        )

        # Set training params (Table 4)
        lstm_est.set_params(epochs=cfg.epochs, batch_size=cfg.batch_size, verbose=0)

        lstm_est.fit(X_tr, y_tr)
        y_pred = lstm_est.predict(X_te)

        results["Class Balance Accuracy"].append(class_balance_accuracy(y_te, y_pred))
        results["G-Mean"].append(g_mean_score(y_te, y_pred))
        results["Mean Sensitivity"].append(mean_sensitivity(y_te, y_pred))
        results["CEN"].append(confusion_entropy(y_te, y_pred))

    return pd.DataFrame(results).mean().to_frame(name="Average").T
