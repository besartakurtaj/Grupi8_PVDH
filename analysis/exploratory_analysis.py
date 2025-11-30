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
    print("Max:", df[col].max())
    print("Mesatarja:", df[col].mean())
    print("Mediana:", df[col].median())
    print("Devijimi standard:", df[col].std())
