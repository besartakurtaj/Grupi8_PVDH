import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

def apply_discretization(df: pd.DataFrame, column: str, n_bins: int = 4, strategy: str = "uniform") -> pd.DataFrame:
    if column not in df.columns:
        print(f"Warning: Column '{column}' not found in DataFrame. Skipping discretization.")
        return df

    if not pd.api.types.is_numeric_dtype(df[column]):
        print(f"Warning: Column '{column}' is not numeric and cannot be discretized. Skipping.")
        return df

    data_for_discretizer = df[[column]].astype('float64')

    try:
        discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
        
        binned_data = discretizer.fit_transform(data_for_discretizer)
        
        df[column] = binned_data.astype(int)
        
        print(f"Discretization applied successfully: Column '{column}' was overwritten with {n_bins} ordinal bins.")

    except ValueError as e:
        print(f"Error applying discretization to column '{column}': {e}")

    return df
