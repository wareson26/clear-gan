
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any, Dict

import pandas as pd

from .overlap_detection import adjacent_class_pairs, ocsvm_overlap_flag, mutual_knn_clean_majority, OCSVMParams
from .ctgan_oversampling import ctgan_oversample_to_global_majority, CTGANParams


@dataclass
class ClearGANConfig:
    target_col: str = "Class"
    k_neighbors: int = 3
    ocsvm: OCSVMParams = OCSVMParams()
    ctgan: CTGANParams = CTGANParams()
    ir_threshold: float = 0.50
    target_fraction: float = 0.80


def clear_gan_resample(df: pd.DataFrame, cfg: Optional[ClearGANConfig] = None) -> pd.DataFrame:
    """Run CLEAR-GAN pipeline: overlap flag -> neighbourhood cleaning -> CTGAN oversampling.

    This is a cleaned, modular version of your Colab notebook.
    """
    c = cfg or ClearGANConfig()
    if c.target_col not in df.columns:
        raise ValueError(f"target column '{c.target_col}' not found")

    X = df.drop(columns=[c.target_col])
    y = df[c.target_col]

    pairs = adjacent_class_pairs(y)

    # OCSVM overlap flag (stored but not used directly in notebook cleaning; kept for traceability)
    overlap_flag = ocsvm_overlap_flag(X, ocsvm_params=c.ocsvm)
    df2 = df.copy()
    df2["Overlap_Label"] = overlap_flag

    # Clean overlap regions (majority removal)
    cleaned = mutual_knn_clean_majority(df2.drop(columns=["Overlap_Label"], errors="ignore"), c.target_col, pairs, k_neighbors=c.k_neighbors)

    # Oversample relative to global majority AFTER cleaning
    balanced = ctgan_oversample_to_global_majority(
        cleaned,
        target_col=c.target_col,
        params=c.ctgan,
        threshold=c.ir_threshold,
        target_fraction=c.target_fraction,
    )
    return balanced
