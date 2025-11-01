import numpy as np
import pandas as pd
from typing import Iterable, Optional, Set


def _is_binary(series: pd.Series) -> bool:
    vals = pd.Series(series).dropna().unique()
    return len(vals) > 0 and set(vals).issubset({0, 1})


def select_feature_subset(
    df: pd.DataFrame,
    keep_always: Optional[Iterable[str]] = None,
    corr_threshold: float = 0.95,
) -> pd.DataFrame:
    work = df.copy()
    keep: Set[str] = set(keep_always or [])

    # Drop strictly constant columns (except keep_always)
    const_cols = [c for c in work.columns if work[c].nunique(dropna=True) <= 1 and c not in keep]
    if const_cols:
        work = work.drop(columns=const_cols, errors="ignore")

    # Correlation pruning on continuous numeric only
    num_cols = work.select_dtypes(include=[np.number]).columns.tolist()
    cont_cols = [c for c in num_cols if not _is_binary(work[c])]

    if len(cont_cols) >= 2:
        corr = work[cont_cols].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = set()
        for col in upper.columns:
            for row in upper.index:
                if row == col:
                    continue
                val = upper.loc[row, col]
                if pd.notna(val) and val > corr_threshold:
                    a, b = row, col
                    # protect keep_always
                    if a in keep and b in keep:
                        continue
                    if b in keep:
                        to_drop.add(a); continue
                    if a in keep:
                        to_drop.add(b); continue
                    # drop the one with lower variance (more conservative)
                    a_var = np.nanvar(work[a].values)
                    b_var = np.nanvar(work[b].values)
                    drop = a if a_var < b_var else b
                    to_drop.add(drop)
        if to_drop:
            to_drop = to_drop - keep
            work = work.drop(columns=list(to_drop), errors="ignore")

    return work
