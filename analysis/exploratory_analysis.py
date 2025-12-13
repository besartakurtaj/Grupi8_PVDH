import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('../output', exist_ok=True)
print("✓ Output directory ready\n")

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

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


if df.isnull().sum().sum() > 0:
    plt.figure(figsize=(10, 6))
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    plt.bar(range(len(missing_data)), missing_data.values)
    plt.xticks(range(len(missing_data)), missing_data.index, rotation=45, ha='right')
    plt.title('Missing Values per Column')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('../output/missing_values.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\n✓ Saved: missing_values.png")

if len(numeric_cols) > 0:
    n_cols = min(3, len(numeric_cols))
    n_rows = int(np.ceil(len(numeric_cols) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if len(numeric_cols) > 1 else [axes]

    for idx, col in enumerate(numeric_cols):
        sns.histplot(df[col].dropna(), kde=True, ax=axes[idx], color='steelblue')
        axes[idx].set_title(f'Distribution of {col}')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequency')

    for idx in range(len(numeric_cols), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig('../output/numeric_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Saved: numeric_distributions.png")

if len(numeric_cols) > 0:
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if len(numeric_cols) > 1 else [axes]

    for idx, col in enumerate(numeric_cols):
        sns.boxplot(y=df[col], ax=axes[idx], color='lightcoral')
        axes[idx].set_title(f'Box Plot of {col}')
        axes[idx].set_ylabel(col)

    for idx in range(len(numeric_cols), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig('../output/numeric_boxplots.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Saved: numeric_boxplots.png")

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
    print(f"\nAnalyzing category column: {col}")

    print("\nCategory Distribution:")
    print(df[col].value_counts())

    print("\nPercentage Distribution:")
    print((df[col].value_counts(normalize=True) * 100).round(2))

    plt.figure(figsize=(10, 6))
    value_counts = df[col].value_counts()

    if len(value_counts) <= 20:  # Only plot if reasonable number of categories
        sns.countplot(data=df, y=col, order=value_counts.index, palette='viridis')
        plt.title(f'Distribution of {col}')
        plt.xlabel('Count')
        plt.tight_layout()
        plt.savefig(f'../output/categorical_{col}.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✓ Saved: categorical_{col}.png")
    else:
        print(f"  (Skipping plot - too many categories: {len(value_counts)})")


if len(numeric_cols) > 1:
    print("\nCorrelation Matrix:")
    corr = df[numeric_cols].corr()
    print(corr.round(3))

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix Heatmap')
    plt.tight_layout()
    plt.savefig('../output/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Saved: correlation_heatmap.png")

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

    if len(df[col].unique()) <= 10 and len(numeric_cols) > 0:
        n_numeric = len(numeric_cols)
        fig, axes = plt.subplots(1, min(3, n_numeric), figsize=(15, 5))
        if n_numeric == 1:
            axes = [axes]
        elif n_numeric < 3:
            axes = axes.flatten()

        for idx, num_col in enumerate(list(numeric_cols)[:3]):
            if idx < len(axes):
                group_stats[num_col].plot(kind='bar', ax=axes[idx], color='teal')
                axes[idx].set_title(f'Average {num_col} by {col}')
                axes[idx].set_ylabel(f'Mean {num_col}')
                axes[idx].set_xlabel(col)
                axes[idx].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(f'../output/grouped_analysis_{col}.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✓ Saved: grouped_analysis_{col}.png")

print("\n" + "=" * 50)
print("All visualizations saved to ../output/ directory!")
print("=" * 50)