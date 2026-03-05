"""
CLEAR-GAN: Overlap-aware hybrid resampling for ordinal multiclass data.

This package provides:
- CLEAR-GAN resampling pipeline
- Overlap detection and cleaning
- CTGAN-based oversampling
- Evaluation utilities for RF and LSTM models
"""

from .clear_gan import clear_gan_resample, ClearGANConfig
from .evaluation import evaluate_with_cv_rf, evaluate_with_cv_lstm
from .metrics import (
    class_balance_accuracy,
    g_mean_score,
    mean_sensitivity,
    confusion_entropy,
)

__all__ = [
    "clear_gan_resample",
    "ClearGANConfig",
    "evaluate_with_cv_rf",
    "evaluate_with_cv_lstm",
    "class_balance_accuracy",
    "g_mean_score",
    "mean_sensitivity",
    "confusion_entropy",
]
