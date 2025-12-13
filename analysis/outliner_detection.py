import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('../analysis/output', exist_ok=True)
print("✓ Output directory ready\n")

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

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

plt.figure(figsize=(10, 6))
outlier_counts = outliers_iqr.sum().sort_values(ascending=False)
plt.bar(range(len(outlier_counts)), outlier_counts.values, color='coral')
plt.xticks(range(len(outlier_counts)), outlier_counts.index, rotation=45, ha='right')
plt.title('Outlier Count per Column (IQR Method)')
plt.ylabel('Number of Outliers')
plt.xlabel('Column')
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
plt.tight_layout()
plt.savefig('../analysis/output/outlier_counts_iqr.png', dpi=300, bbox_inches='tight')
plt.show()
print("\n✓ Saved: outlier_counts_iqr.png")

z_scores = np.abs(stats.zscore(df[numeric_cols]))
outliers_z = (z_scores > 3)

outliers_z = pd.DataFrame(outliers_z, columns=numeric_cols, index=df.index)

print("\nOutlier counts per column (Z-Score > 3):")
print(outliers_z.sum())

plt.figure(figsize=(10, 6))
z_outlier_counts = outliers_z.sum().sort_values(ascending=False)
plt.bar(range(len(z_outlier_counts)), z_outlier_counts.values, color='steelblue')
plt.xticks(range(len(z_outlier_counts)), z_outlier_counts.index, rotation=45, ha='right')
plt.title('Outlier Count per Column (Z-Score Method)')
plt.ylabel('Number of Outliers')
plt.xlabel('Column')
plt.tight_layout()
plt.savefig('../analysis/output/outlier_counts_zscore.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: outlier_counts_zscore.png")

df["is_outlier"] = (outliers_iqr | outliers_z).any(axis=1)

print(f"\nTotal outliers detected: {df['is_outlier'].sum()} of {len(df)} rows")

plt.figure(figsize=(8, 6))
outlier_dist = df['is_outlier'].value_counts()
colors = ['lightgreen', 'salmon']
plt.pie(outlier_dist.values, labels=['Clean Data', 'Outliers'], autopct='%1.1f%%',
        colors=colors, startangle=90)
plt.title(f'Outlier Distribution\n(Total: {len(df)} rows)')
plt.tight_layout()
plt.savefig('../analysis/output/outlier_pie_chart.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: outlier_pie_chart.png")

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