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

removed_outliers = df[df["is_outlier"] == True]
removed_outliers.to_csv("../data/removed_outliers_log.csv", index=False)
print("Saved → /data/removed_outliers_log.csv (trace of removed rows)")

df.to_csv("../data/dataset_with_outliers_flag.csv", index=False)
print("Saved → /data/dataset_with_outliers_flag.csv")

print("\nDiagnostic info:")
for col in numeric_cols:
    n_unique = df[col].nunique()
    print(f"{col}: unique values = {n_unique}, IQR = {IQR[col]}")

df_clean = df[df["is_outlier"] == False].copy()
print(f"\nOriginal dataset: {df.shape}")
print(f"After removing outliers: {df_clean.shape}")

df_clean.to_csv("../data/cleaned_dataset.csv", index=False)
print("Saved → /data/cleaned_dataset.csv")

print("\nSummary statistics (cleaned):")
print(df_clean.describe(include='all'))

outlier_percentage = (df['is_outlier'].sum() / len(df)) * 100
print(f"Outlier percentage: {outlier_percentage:.2f}%")

print("\nOutlier distribution per row:")
print(df['is_outlier'].value_counts())