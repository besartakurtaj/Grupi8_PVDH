import pandas as pd
import numpy as np

def reduce_dimensions_enhanced(
    df: pd.DataFrame,
    protected_cols: list = None,
    corr_threshold: float = 0.98,
    log: bool = True
) -> pd.DataFrame:
    """
    Enhanced dimensionality reduction with explanation logging.
    - Removes exact duplicate columns
    - Removes highly correlated columns
    - Keeps protected columns
    - Explains WHY each column was removed
    """
    df_reduced = df.copy()
    protected_cols = protected_cols or []
    reasons = []

    duplicate_cols = df_reduced.T[df_reduced.T.duplicated()].index.tolist()
    duplicate_cols = [c for c in duplicate_cols if c not in protected_cols]
    if duplicate_cols:
        for col in duplicate_cols:
            reasons.append((col, "Exact duplicate of another column"))
        df_reduced.drop(columns=duplicate_cols, inplace=True, errors="ignore")
        if log:
            print(f"Removed exact duplicate columns ({len(duplicate_cols)}): {duplicate_cols}")

    numeric_df = df_reduced.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    drop_corr = set()
    for col in upper.columns:
        correlated = upper.index[upper[col] > corr_threshold].tolist()
        for corr_col in correlated:
            if col not in protected_cols and corr_col not in protected_cols:
                var_col = np.nanvar(df_reduced[col].values)
                var_corr = np.nanvar(df_reduced[corr_col].values)
                to_drop = col if var_col < var_corr else corr_col
                keep = corr_col if to_drop == col else col
                drop_corr.add(to_drop)
                reasons.append((to_drop, f"Highly correlated with '{keep}' (corr={upper.loc[corr_col, col]:.3f})"))

    if drop_corr:
        df_reduced.drop(columns=list(drop_corr), inplace=True, errors="ignore")
        if log:
            print(f"Removed highly correlated columns ({len(drop_corr)}): {list(drop_corr)}")

    before_rows = df_reduced.shape[0]
    df_reduced = df_reduced.drop_duplicates().reset_index(drop=True)
    after_rows = df_reduced.shape[0]
    if before_rows != after_rows and log:
        print(f"Removed {before_rows - after_rows} duplicate rows.")

    print(f"\nEnhanced reduction complete: {df.shape[1]} → {df_reduced.shape[1]} features\n")
    if log and reasons:
        print("Removal reasons:")
        for col, reason in reasons:
            print(f" - {col}: {reason}")

    return df_reduced
