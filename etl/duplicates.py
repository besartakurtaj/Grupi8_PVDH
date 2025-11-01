import pandas as pd
from typing import Iterable, Optional

def remove_duplicates(
    df: pd.DataFrame,
    subset: Optional[Iterable[str]] = None,
    keep: str = "first",
    report: bool = True,
) -> pd.DataFrame:
    
    before = len(df)
    result = df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
    
    if report:
        removed = before - len(result)
        print(f"Removed {removed} duplicate rows (subset={subset}, keep='{keep}'). "
              f"New shape: {result.shape}")
    
    return result
