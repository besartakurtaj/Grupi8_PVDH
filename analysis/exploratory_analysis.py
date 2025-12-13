import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('../analysis/output', exist_ok=True)
print("✓ Output directory ready\n")

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 11

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


print("\n" + "=" * 70)
print("GRAPH 1: Top 10 Strongest Correlations")
print("=" * 70)

corr = df[numeric_cols].corr()
corr_pairs = corr.abs().unstack()
corr_pairs = corr_pairs[corr_pairs < 1]
corr_pairs = corr_pairs.dropna().sort_values(ascending=False).head(10)

correlations = []
labels = []
for idx, value in corr_pairs.items():
    actual_corr = corr.loc[idx[0], idx[1]]
    correlations.append(actual_corr)
    label1 = idx[0].replace('Social Platform Preference ', '').replace('_', ' ')
    label2 = idx[1].replace('Social Platform Preference ', '').replace('_', ' ')
    labels.append(f"{label1}\n& {label2}")

fig, ax = plt.subplots(figsize=(14, 9))
colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in correlations]
bars = ax.barh(range(len(correlations)), correlations, color=colors,
               edgecolor='black', linewidth=1.5, alpha=0.8)

for i, v in enumerate(correlations):
    ax.text(v + 0.02 if v > 0 else v - 0.02, i, f'{v:.3f}',
            va='center', fontweight='bold', fontsize=11,
            ha='left' if v > 0 else 'right')

ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel('Correlation Coefficient', fontsize=13, fontweight='bold')
ax.set_title('Top 10 Strongest Variable Correlations\n(Positive correlations shown in green)',
             fontsize=16, fontweight='bold', pad=20)
ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
ax.set_xlim(-0.1, 1.05)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('../analysis/output/top_correlations.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: top_correlations.png")
print(f"  • Strongest: Productivity measures")
print(f"  • Critical: Social Media ↔ Distraction Load")


print("\n" + "=" * 70)
print("GRAPH 2: Risk Factor Prevalence (ALARMING STATISTICS)")
print("=" * 70)

risk_factors = {
    'Too Many Notifications': 'Too Many Notifications',
    'Burnout Risk': 'Burnout Risk',
    'High Stress': 'High Stress',
    'Low Sleep': 'Low Sleep',
    'Social Addicted': 'Social Media Addiction'
}

risk_percentages = []
risk_labels = []
risk_counts = []

for col, label in risk_factors.items():
    if col in df.columns:
        count = df[col].sum()
        percentage = (count / len(df)) * 100
        risk_percentages.append(percentage)
        risk_labels.append(label)
        risk_counts.append(count)

fig, ax = plt.subplots(figsize=(12, 8))

colors_gradient = ['#c0392b', '#e74c3c', '#e67e22', '#f39c12', '#3498db']
bars = ax.barh(range(len(risk_labels)), risk_percentages,
               color=colors_gradient, edgecolor='black', linewidth=2, alpha=0.85)

for i, (pct, count) in enumerate(zip(risk_percentages, risk_counts)):
    ax.text(pct + 1.5, i, f'{pct:.1f}%\n({count:,} people)',
            va='center', fontweight='bold', fontsize=11)

ax.set_yticks(range(len(risk_labels)))
ax.set_yticklabels(risk_labels, fontsize=12, fontweight='bold')
ax.set_xlabel('Percentage of Population Affected (%)', fontsize=13, fontweight='bold')
ax.set_title('Risk Factor Prevalence in Dataset\nCritical Health & Wellbeing Indicators',
             fontsize=16, fontweight='bold', pad=20, color='#c0392b')
ax.set_xlim(0, 100)
ax.grid(True, alpha=0.3, axis='x')

ax.axvline(x=50, color='red', linestyle='--', linewidth=2, alpha=0.5, label='50% threshold')
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig('../analysis/output/risk_prevalence.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: risk_prevalence.png")


print("\n" + "=" * 70)
print("GRAPH 3: Stress Impact on Productivity & Job Satisfaction")
print("=" * 70)

if all(col in df.columns for col in ['Stress Level', 'Actual Productivity Score', 'Job Satisfaction Score']):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    scatter1 = axes[0].scatter(df['Stress Level'],
                               df['Actual Productivity Score'],
                               c=df['Job Satisfaction Score'],
                               s=50, alpha=0.6, cmap='RdYlGn',
                               edgecolors='black', linewidth=0.3)

    z = np.polyfit(df['Stress Level'], df['Actual Productivity Score'], 1)
    p = np.poly1d(z)
    axes[0].plot(df['Stress Level'].sort_values(),
                 p(df['Stress Level'].sort_values()),
                 "r--", linewidth=3, label=f'Trend: r={df["Stress Level"].corr(df["Actual Productivity Score"]):.3f}')

    axes[0].set_xlabel('Stress Level', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Actual Productivity Score', fontsize=12, fontweight='bold')
    axes[0].set_title('Stress vs Productivity\n(Color = Job Satisfaction)',
                      fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    cbar1 = plt.colorbar(scatter1, ax=axes[0])
    cbar1.set_label('Job Satisfaction', fontweight='bold')

    scatter2 = axes[1].scatter(df['Stress Level'],
                               df['Job Satisfaction Score'],
                               c=df['Actual Productivity Score'],
                               s=50, alpha=0.6, cmap='viridis',
                               edgecolors='black', linewidth=0.3)

    z2 = np.polyfit(df['Stress Level'], df['Job Satisfaction Score'], 1)
    p2 = np.poly1d(z2)
    axes[1].plot(df['Stress Level'].sort_values(),
                 p2(df['Stress Level'].sort_values()),
                 "r--", linewidth=3, label=f'Trend: r={df["Stress Level"].corr(df["Job Satisfaction Score"]):.3f}')

    axes[1].set_xlabel('Stress Level', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Job Satisfaction Score', fontsize=12, fontweight='bold')
    axes[1].set_title('Stress vs Job Satisfaction\n(Color = Productivity)',
                      fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    cbar2 = plt.colorbar(scatter2, ax=axes[1])
    cbar2.set_label('Productivity', fontweight='bold')

    plt.suptitle('Impact of Stress on Work Performance',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('../analysis/output/stress_impact.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Saved: stress_impact.png")

    corr_stress_prod = df['Stress Level'].corr(df['Actual Productivity Score'])
    corr_stress_sat = df['Stress Level'].corr(df['Job Satisfaction Score'])
    print(f"  • Stress-Productivity correlation: {corr_stress_prod:.3f}")
    print(f"  • Stress-Satisfaction correlation: {corr_stress_sat:.3f}")


print("\n" + "=" * 70)
print("GRAPH 4: Dataset Demographics")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

if 'Gender' in df.columns:
    gender_counts = df['Gender'].value_counts()
    colors_gender = ['#3498db', '#e74c3c', '#95a5a6']
    explode = (0.05, 0.05, 0.1)

    axes[0, 0].pie(gender_counts.values, labels=gender_counts.index,
                   autopct='%1.1f%%', colors=colors_gender, explode=explode,
                   startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'},
                   shadow=True)
    axes[0, 0].set_title('Gender Distribution', fontsize=14, fontweight='bold', pad=15)

if 'Gender' in df.columns:
    axes[0, 1].bar(gender_counts.index, gender_counts.values,
                   color=colors_gender, edgecolor='black', linewidth=2)
    axes[0, 1].set_ylabel('Count', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Gender Count', fontsize=14, fontweight='bold', pad=15)
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    for i, v in enumerate(gender_counts.values):
        axes[0, 1].text(i, v + 100, f'{v:,}\n({v / len(df) * 100:.1f}%)',
                        ha='center', fontweight='bold', fontsize=11)

if 'Job Type' in df.columns:
    job_counts = df['Job Type'].value_counts()
    colors_job = sns.color_palette("husl", len(job_counts))

    axes[1, 0].barh(range(len(job_counts)), job_counts.values,
                    color=colors_job, edgecolor='black', linewidth=1.5)
    axes[1, 0].set_yticks(range(len(job_counts)))
    axes[1, 0].set_yticklabels(job_counts.index, fontsize=11)
    axes[1, 0].set_xlabel('Count', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Job Type Distribution', fontsize=14, fontweight='bold', pad=15)
    axes[1, 0].grid(True, alpha=0.3, axis='x')

    for i, v in enumerate(job_counts.values):
        axes[1, 0].text(v + 30, i, f'{v:,} ({v / len(df) * 100:.1f}%)',
                        va='center', fontweight='bold', fontsize=10)

if 'Job Optimism' in df.columns:
    optimism_counts = df['Job Optimism'].value_counts()
    colors_opt = ['#2ecc71', '#f39c12', '#e74c3c']

    axes[1, 1].bar(range(len(optimism_counts)), optimism_counts.values,
                   color=colors_opt, edgecolor='black', linewidth=2)
    axes[1, 1].set_xticks(range(len(optimism_counts)))
    axes[1, 1].set_xticklabels([x.replace(' Job', '') for x in optimism_counts.index],
                               fontsize=11, rotation=15)
    axes[1, 1].set_ylabel('Count', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Job Optimism Distribution', fontsize=14, fontweight='bold', pad=15)
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    for i, v in enumerate(optimism_counts.values):
        axes[1, 1].text(i, v + 100, f'{v:,}\n({v / len(df) * 100:.1f}%)',
                        ha='center', fontweight='bold', fontsize=11)

plt.suptitle(f'Dataset Demographics Overview\n(Total: {len(df):,} participants)',
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('../analysis/output/demographics.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: demographics.png")