import pandas as pd
import numpy as np

df = pd.read_csv("../data/cleaned_dataset.csv")

print("Cleaned data loaded:", df.shape)
print("\nBasic Information:")
print(df.info())

print("\nSummary Statistics (All Columns):")
print(df.describe(include="all"))

print("\nMissing Values per Column:")

print(df.isnull().sum())

print("\nTotal Missing Values in Dataset:", df.isnull().sum().sum())

numeric_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(exclude=[np.number]).columns

print("\nNumeric Columns:", list(numeric_cols))
print("\nCategorical Columns:", list(categorical_cols))

print("\nDetailed Numeric Stats:")
for col in numeric_cols:
    print(f"\nColumn: {col}")
    print("Min:", df[col].min())
    print("Max:", df[col].max())
    print("Mean:", df[col].mean())
    print("Median:", df[col].median())
    print("Standard Deviation:", df[col].std())

print("\nDetailed Categorical Stats:")
for col in categorical_cols:
    print(f" Analyzing category column: {col}")

    print("\nCategory Distribution:")
    print(df[col].value_counts())

    print("\nPercentage Distribution:")
    print((df[col].value_counts(normalize=True) * 100).round(2))


if len(numeric_cols) > 1:
    print("\nCorrelation Matrix:")
    corr = df[numeric_cols].corr()
    print(corr.round(3))

    print("\nTop 10 Strongest Correlations:")
    corr_pairs = corr.abs().unstack()
    corr_pairs = corr_pairs[corr_pairs < 1]
    corr_pairs = corr_pairs.dropna().sort_values(ascending=False)
    print(corr_pairs.head(10))
else:
    print("\nNot enough numeric columns for correlation.")

print("\nMultivariate Analysis (Grouped Averages):")
for col in categorical_cols:
    print(f"\nGrouping by: {col}")

    print("\nAverage Numeric Values by Category:")
    group_stats = df.groupby(col)[numeric_cols].mean().round(2)
    print(group_stats)

    print("\nVariation (Std Dev) by Category:")
    group_std = df.groupby(col)[numeric_cols].std().round(2)
    print(group_std)

    print("\nTop numeric feature differences:")
    print(group_stats.max() - group_stats.min())