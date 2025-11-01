import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

def apply_discretization(df: pd.DataFrame, column: str, n_bins: int = 4, strategy: str = "uniform") -> pd.DataFrame:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    if not pd.api.types.is_numeric_dtype(df[column]):
        raise TypeError(f"Column '{column}' is not numeric and cannot be discretized.")

    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
    df[column + "_binned"] = discretizer.fit_transform(df[[column]])

    print(f"Discretization applied successfully. Created new column: '{column}_binned'")

    # # Optionally convert the new column to category for later use
    # df[column + "_binned"] = df[column + "_binned"].astype("category")

    return df
