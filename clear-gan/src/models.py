from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

import numpy as np

# TensorFlow / Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# SciKeras wrapper (for sklearn GridSearchCV compatibility)
from scikeras.wrappers import KerasClassifier

from sklearn.ensemble import RandomForestClassifier


@dataclass
class RFConfig:
    n_estimators: int = 200
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    random_state: int = 42
    n_jobs: int = -1


@dataclass
class LSTMConfig:
    # Table 4 (paper): memory cells=50, optimizer=Adam, epochs=150, batch_size=32, activation=softmax
    memory_cells: int = 50
    dropout_rate: float = 0.2
    optimizer: str = "adam"
    epochs: int = 150
    batch_size: int = 32
    random_state: int = 42


def make_rf(cfg: RFConfig) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        min_samples_split=cfg.min_samples_split,
        min_samples_leaf=cfg.min_samples_leaf,
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
    )


def build_lstm_model(
    input_shape: Tuple[int, int],
    n_classes: int,
    memory_cells: int = 50,
    dropout_rate: float = 0.2,
    optimizer: str = "adam",
) -> keras.Model:
    """
    Build an LSTM classifier for ordinal multiclass tasks.
    input_shape: (timesteps, n_features)
    n_classes: number of classes
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(memory_cells, return_sequences=False),
        layers.Dropout(dropout_rate),
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def make_lstm_estimator(
    input_shape: Tuple[int, int],
    n_classes: int,
    cfg: Optional[LSTMConfig] = None,
) -> KerasClassifier:
    """
    SciKeras-compatible estimator so you can use sklearn GridSearchCV.
    """
    cfg = cfg or LSTMConfig()
    tf.keras.utils.set_random_seed(cfg.random_state)

    est = KerasClassifier(
        model=build_lstm_model,
        model__input_shape=input_shape,
        model__n_classes=n_classes,
        # default (can be overridden by GridSearchCV)
        model__memory_cells=cfg.memory_cells,
        model__dropout_rate=cfg.dropout_rate,
        model__optimizer=cfg.optimizer,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        verbose=0,
    )
    return est


def to_sequence_3d(X: np.ndarray, timesteps: int, n_features: int) -> np.ndarray:
    """
    Convert 2D tabular matrix into 3D tensor for LSTM: (n_samples, timesteps, n_features).
    Use timesteps=1 for purely tabular datasets.
    """
    if X.ndim == 3:
        return X
    if X.ndim != 2:
        raise ValueError(f"Expected X with 2 or 3 dims, got shape {X.shape}")
    expected = timesteps * n_features
    if X.shape[1] != expected:
        raise ValueError(
            f"Expected {expected} columns (=timesteps*n_features), got {X.shape[1]}. "
            "If your data are not longitudinal, set timesteps=1 and n_features=X.shape[1]."
        )
    return X.reshape((X.shape[0], timesteps, n_features))
