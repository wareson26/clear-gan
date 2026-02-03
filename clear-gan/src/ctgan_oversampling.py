
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

import pandas as pd

try:
    from ctgan import CTGAN
except Exception as e:  
    CTGAN = None  


@dataclass
class CTGANParams:
    epochs: int = 300
    batch_size: int = 32
    pac: int = 8
    generator_dim: tuple = (256, 512, 256)
    discriminator_dim: tuple = (256, 128, 64)


def build_sampling_strategy(
    y: pd.Series,
    majority_count: int,
    threshold: float = 0.50,
    target_fraction: float = 0.80,
) -> Dict[Any, int]:

    """Match oversample classes whose count/majority < threshold to target_fraction*majority."""
    counts = y.value_counts()
    strategy: Dict[Any, int] = {}
    for cls, cnt in counts.items():
        imbalance_ratio = cnt / majority_count
        if imbalance_ratio < threshold:
            strategy[cls] = int(target_fraction * majority_count)
    return strategy


def ctgan_oversample_to_global_majority(
    df: pd.DataFrame,
    target_col: str,
    params: Optional[CTGANParams] = None,
    threshold: float = 0.50,
    target_fraction: float = 0.80, #80% of the global majority
    random_state: int = 42,
) -> pd.DataFrame:

    """CTGAN oversampling applied AFTER overlap cleaning, relative to the global majority.
    """
    if CTGAN is None:
        raise ImportError("ctgan is not installed. Install with: pip install ctgan")

    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found")

    counts = df[target_col].value_counts()
    majority_count = int(counts.max())
    strategy = build_sampling_strategy(df[target_col], majority_count, threshold=threshold, target_fraction=target_fraction)

    if not strategy:
        return df.copy()

    p = CTGANParams()
    ctgan = CTGAN(
        epochs=p.epochs,
        batch_size=p.batch_size,
        pac=p.pac,
        generator_dim=p.generator_dim,
        discriminator_dim=p.discriminator_dim,
        cuda=True if hasattr(__import__('torch'), 'cuda') else False,  # best-effort
    )

    # Identify categorical columns by dtype
    cat_cols = [c for c in df.columns if c != target_col and str(df[c].dtype) in ("object", "category", "bool")]
    
    ctgan.fit(df.drop(columns=[target_col]), discrete_columns=cat_cols)

    synth_list = []
    for cls, new_count in strategy.items():
        n_to_sample = int(new_count - counts[cls])
        if n_to_sample <= 0:
            continue
        synth = ctgan.sample(n_to_sample)
        synth[target_col] = cls
        synth_list.append(synth)

    if not synth_list:
        return df.copy()

    df_synth = pd.concat(synth_list, ignore_index=True)
    return pd.concat([df, df_synth], ignore_index=True)
