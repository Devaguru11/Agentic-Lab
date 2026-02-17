# =============================================================
# DAY 1–2: Dataset Setup & Exploratory Data Analysis (EDA)
# AI Benchmarking Project — Week 1
# =============================================================
# HOW TO RUN:
#   1. Open this file in VS Code
#   2. Install the Jupyter extension in VS Code
#   3. Click "Run All" at the top
# =============================================================

# ── CELL 1: Install dependencies (run once) ──────────────────
# Run this in your VS Code terminal first:
#   pip install -r requirements.txt

# ── CELL 2: Imports ──────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

print("✅ All libraries imported successfully!")

# ── CELL 3: Load Dataset ─────────────────────────────────────
# The PIMA Indians Diabetes Dataset is included here directly
# so you do NOT need to download anything from Kaggle.
# Source: UCI Machine Learning Repository (public domain)

COLUMNS = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
]

# Download using urllib (no Kaggle account needed)
import urllib.request

DATA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
DATA_PATH = "data/pima_diabetes.csv"

os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

if not os.path.exists(DATA_PATH):
    print("Downloading dataset...")
    import requests
    response = requests.get(DATA_URL, verify=False)
    with open(DATA_PATH, 'wb') as f:
        f.write(response.content)
    print(f"✅ Dataset saved to {DATA_PATH}")
else:
    print(f"✅ Dataset already exists at {DATA_PATH}")

df = pd.read_csv(DATA_PATH, header=None, names=COLUMNS)
print(f"\n📊 Dataset shape: {df.shape}")
print(f"   Rows: {df.shape[0]} patients")
print(f"   Columns: {df.shape[1]} features")

# ── CELL 4: First Look ───────────────────────────────────────
print("\n" + "="*55)
print("FIRST 5 ROWS")
print("="*55)
print(df.head())

print("\n" + "="*55)
print("BASIC STATISTICS")
print("="*55)
print(df.describe().round(2))

# ── CELL 5: Class Distribution ───────────────────────────────
print("\n" + "="*55)
print("CLASS DISTRIBUTION")
print("="*55)
counts = df['Outcome'].value_counts()
print(f"  Non-Diabetic (0): {counts[0]} patients ({counts[0]/len(df)*100:.1f}%)")
print(f"  Diabetic     (1): {counts[1]} patients ({counts[1]/len(df)*100:.1f}%)")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('PIMA Diabetes Dataset — Class Distribution', fontsize=14, fontweight='bold')

# Bar chart
axes[0].bar(['Non-Diabetic', 'Diabetic'], [counts[0], counts[1]],
            color=['#2563EB', '#DC2626'], alpha=0.85, edgecolor='white', linewidth=1.5)
axes[0].set_title('Patient Count by Class')
axes[0].set_ylabel('Count')
for i, v in enumerate([counts[0], counts[1]]):
    axes[0].text(i, v + 5, str(v), ha='center', fontweight='bold')

# Pie chart
axes[1].pie([counts[0], counts[1]], labels=['Non-Diabetic', 'Diabetic'],
            colors=['#2563EB', '#DC2626'], autopct='%1.1f%%',
            startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
axes[1].set_title('Class Proportion')

plt.tight_layout()
plt.savefig('outputs/01_class_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: outputs/01_class_distribution.png")

# ── CELL 6: Missing / Zero Value Analysis ────────────────────
# In PIMA dataset, 0s in medical columns are actually missing values
ZERO_INVALID_COLS = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

print("\n" + "="*55)
print("ZERO VALUES (= MISSING DATA) IN MEDICAL COLUMNS")
print("="*55)
for col in ZERO_INVALID_COLS:
    zeros = (df[col] == 0).sum()
    pct = zeros / len(df) * 100
    print(f"  {col:<30} {zeros:>3} zeros  ({pct:.1f}%)")

# ── CELL 7: Handle Missing Values ────────────────────────────
# Strategy: replace 0s with column median (safe, robust)
df_clean = df.copy()
for col in ZERO_INVALID_COLS:
    median_val = df_clean[col][df_clean[col] != 0].median()
    df_clean[col] = df_clean[col].replace(0, median_val)

print("\n✅ Replaced invalid zeros with column medians")
print("   Clean dataset ready for modelling.")

# Save clean dataset
df_clean.to_csv('data/pima_diabetes_clean.csv', index=False)
print("✅ Saved: data/pima_diabetes_clean.csv")

# ── CELL 8: Feature Distributions ───────────────────────────
FEATURES = ['Pregnancies', 'Glucose', 'BloodPressure', 'BMI', 'Age', 'Insulin']
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Feature Distributions by Diabetic Status', fontsize=14, fontweight='bold')

for i, feat in enumerate(FEATURES):
    ax = axes[i // 3][i % 3]
    diabetic     = df_clean[df_clean['Outcome'] == 1][feat]
    non_diabetic = df_clean[df_clean['Outcome'] == 0][feat]
    ax.hist(non_diabetic, bins=25, alpha=0.6, color='#2563EB', label='Non-Diabetic')
    ax.hist(diabetic,     bins=25, alpha=0.6, color='#DC2626', label='Diabetic')
    ax.set_title(feat, fontweight='bold')
    ax.set_xlabel('Value')
    ax.set_ylabel('Count')
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('outputs/02_feature_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: outputs/02_feature_distributions.png")

# ── CELL 9: Correlation Heatmap ──────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
corr = df_clean.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=ax, linewidths=0.5,
            annot_kws={'size': 9})
ax.set_title('Feature Correlation Matrix', fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('outputs/03_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: outputs/03_correlation_heatmap.png")

# ── CELL 10: Summary ─────────────────────────────────────────
print("\n" + "="*55)
print("DAY 1–2 COMPLETE ✅")
print("="*55)
print("Files created:")
print("  data/pima_diabetes.csv         ← raw dataset")
print("  data/pima_diabetes_clean.csv   ← cleaned dataset")
print("  outputs/01_class_distribution.png")
print("  outputs/02_feature_distributions.png")
print("  outputs/03_correlation_heatmap.png")
print("\nNext → Open day3_4_rule_based.py")