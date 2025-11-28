import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv("../data/processed_dataset.csv")
print("Dataset loaded:", df.shape)

numeric_cols = df.select_dtypes(include=[np.number]).columns
print("\nNumeric columns found:")
print(list(numeric_cols))

Q1 = df[numeric_cols].quantile(0.25)
Q3 = df[numeric_cols].quantile(0.75)
IQR = Q3 - Q1

IQR[IQR == 0] = np.nan

print("\nInterquartile ranges (IQR):")
print(IQR)

outliers_iqr = ((df[numeric_cols] < (Q1 - 1.5 * IQR)) |
                (df[numeric_cols] > (Q3 + 1.5 * IQR)))

print("\nOutlier counts per column:")
print(outliers_iqr.sum())

z_scores = np.abs(stats.zscore(df[numeric_cols]))
outliers_z = (z_scores > 3)

outliers_z = pd.DataFrame(outliers_z, columns=numeric_cols, index=df.index)

print("\nOutlier counts per column (Z-Score > 3):")
print(outliers_z.sum())

df["is_outlier"] = (outliers_iqr | outliers_z).any(axis=1)

print(f"\nTotal outliers detected: {df['is_outlier'].sum()} of {len(df)} rows")