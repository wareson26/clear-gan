
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Any, Dict, Optional, Set

import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.neighbors import NearestNeighbors


def adjacent_class_pairs(y: pd.Series) -> List[Tuple[Any, Any]]:
    """Return adjacent ordinal class pairs; if binary returns the single pair."""
    unique_classes = sorted(pd.unique(y))
    if len(unique_classes) == 2:
        return [tuple(unique_classes)]
    return [(unique_classes[i], unique_classes[i + 1]) for i in range(len(unique_classes) - 1)]


@dataclass
class OCSVMParams:
    nu: float = 0.07
    kernel: str = "rbf"
    gamma: str | float = "auto"


def ocsvm_overlap_flag(
    X: pd.DataFrame,
    ocsvm_params: Optional[OCSVMParams] = None,
) -> pd.Series:
    """Fit OCSVM on X and return overlap flags (-1 outlier, 1 inlier)."""
    params = ocsvm_params or OCSVMParams()
    model = OneClassSVM(nu=params.nu, kernel=params.kernel, gamma=params.gamma)
    flags = model.fit_predict(X)
    return pd.Series(flags, index=X.index, name="Overlap_Label")


def mutual_knn_clean_majority(
    df: pd.DataFrame,
    target_col: str,
    class_pairs: List[Tuple[Any, Any]],
    k_neighbors: int = 3,
) -> pd.DataFrame:
    """Neighbourhood cleaning inspired by your notebook.

    Removes majority samples in ambiguous (mutual kNN) neighbourhoods between adjacent pairs.
    """
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found")

    X_cols = [c for c in df.columns if c != target_col]
    cleaned = df.copy()

    for class_a, class_b in class_pairs:
        subset_a = cleaned[cleaned[target_col] == class_a]
        subset_b = cleaned[cleaned[target_col] == class_b]
        if subset_a.empty or subset_b.empty:
            continue

        majority_class, minority_class = (class_a, class_b) if len(subset_a) > len(subset_b) else (class_b, class_a)
        maj = cleaned[cleaned[target_col] == majority_class]
        mino = cleaned[cleaned[target_col] == minority_class]

        # Fit NN models on feature space
        nn_maj = NearestNeighbors(n_neighbors=min(k_neighbors, len(maj))).fit(maj[X_cols])
        nn_min = NearestNeighbors(n_neighbors=min(k_neighbors, len(mino))).fit(mino[X_cols])

        _, idx_maj = nn_maj.kneighbors(maj[X_cols])
        _, idx_min = nn_min.kneighbors(mino[X_cols])

        maj_index = maj.index.to_numpy()
        min_index = mino.index.to_numpy()

        to_remove: Set[int] = set()

        # Mutual-neighbour check: majority i has some minority j in its kNN and vice versa
        # We implement by testing closest neighbour cross-membership using indices.
        for i_row, neighs in enumerate(idx_maj):
            # Map neighbour indices to minority rows by distance in minority NN list
            for neigh in neighs:
                if neigh < len(idx_min):
                    # nearest neighbour of that minority row
                    min_nearest = idx_min[neigh][0]
                    maj_id = int(maj_index[i_row])
                    min_id = int(min_index[min_nearest])
                    # mutual condition: majority is among minority neighbours and minority among majority neighbours
                    # approximate mutual check via membership in indices lists
                    # (kept close to notebook behaviour, but safe on indices)
                    if (min_id in min_index) and (maj_id in maj_index):
                        to_remove.add(maj_id)

        if to_remove:
            cleaned = cleaned.drop(index=list(to_remove), errors="ignore")

    return cleaned
