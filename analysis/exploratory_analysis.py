import pandas as pd
import numpy as np

df = pd.read_csv("cleaned_dataset.csv")

print("Cleaned data loaded:", df.shape)
print("\nBasic Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe(include="all"))

print("\nMissing Values:")

print(df.isnull().sum())

print("\nTotal Missing Values in Dataset:", df.isnull().sum().sum())

numeric_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(exclude=[np.number]).columns

print("\nNumeric columns")
print(list(numeric_cols))

print("\nCategorical columns:")
print(list(categorical_cols))

print("\nDetailed Numeric Stats:")
for col in numeric_cols:
    print(f"\nKolona: {col}")
    print("Min:", df[col].min())
    
print("\nTop Categories:")
for col in categorical_cols:
    print(f"\nCategory: {col}")
    print(df[col].value_counts())

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
    print(f" Analyzing category: {col}")

    print("\nCategory Distribution:")
    print(df[col].value_counts())
    print("\nPercentage Distribution:")
    print((df[col].value_counts(normalize=True) * 100).round(2))

    print("\nAverage Numeric Values by Category:")
    group_stats = df.groupby(col)[numeric_cols].mean().round(2)
    print(group_stats)

    print("\nVariation (Std Dev) by Category:")
    print(df.groupby(col)[numeric_cols].std().round(2))

    print("\nTop numeric feature differences:")
    print(group_stats.max() - group_stats.min())


print("\nDetailed Numeric Stats:")
for col in numeric_cols:
    print(f"\nKolona: {col}")
    print("Min:", df[col].min())
    print("Max:", df[col].max())
    print("Mesatarja:", df[col].mean())
    print("Mediana:", df[col].median())
    print("Devijimi standard:", df[col].std())
