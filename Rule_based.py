# =============================================================
# DAY 3–4: Rule-Based Expert System (Clinical Threshold Rules)
# AI Benchmarking Project — Week 1
# =============================================================
# This is your BASELINE 1: a hand-coded expert system
# using WHO / clinical literature thresholds.
# No machine learning — pure if/then logic.
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, roc_auc_score,
                              roc_curve, classification_report)
import warnings
warnings.filterwarnings('ignore')

# ── CELL 1: Load clean data ───────────────────────────────────
df = pd.read_csv('data/pima_diabetes_clean.csv')
X = df.drop('Outcome', axis=1)
y = df['Outcome']

print(f"✅ Loaded {len(df)} patient records")

# ── CELL 2: Define the Rule-Based System ─────────────────────
# Each rule assigns a RISK SCORE (0–5).
# Final threshold: score >= 2 → predict Diabetic

def rule_based_predict(row):
    """
    Clinical rule engine for diabetes risk.
    Based on WHO / ADA (American Diabetes Association) guidelines.

    Risk scoring:
      Glucose >= 140      → +2  (Impaired glucose tolerance)
      Glucose >= 126      → +1  (Pre-diabetic range)
      BMI >= 30           → +1  (Obese)
      Age >= 45           → +1  (Higher risk age group)
      Pregnancies >= 3    → +1  (Gestational history)
      DiabetesPedigreeFunction >= 0.5 → +1 (Family history)
      BloodPressure >= 90 → +1  (Hypertension — comorbidity)

    Decision:
      Score >= 2 → Diabetic (1)
      Score <  2 → Non-Diabetic (0)
    """
    score = 0
    reasons = []

    # Rule 1: Glucose (most important marker)
    if row['Glucose'] >= 140:
        score += 2
        reasons.append(f"Glucose={row['Glucose']:.0f} [HIGH >=140, +2]")
    elif row['Glucose'] >= 126:
        score += 1
        reasons.append(f"Glucose={row['Glucose']:.0f} [ELEVATED >=126, +1]")

    # Rule 2: BMI
    if row['BMI'] >= 30:
        score += 1
        reasons.append(f"BMI={row['BMI']:.1f} [OBESE >=30, +1]")

    # Rule 3: Age
    if row['Age'] >= 45:
        score += 1
        reasons.append(f"Age={row['Age']:.0f} [HIGH RISK >=45, +1]")

    # Rule 4: Pregnancy history
    if row['Pregnancies'] >= 3:
        score += 1
        reasons.append(f"Pregnancies={row['Pregnancies']:.0f} [>=3, +1]")

    # Rule 5: Diabetes Pedigree (family history proxy)
    if row['DiabetesPedigreeFunction'] >= 0.5:
        score += 1
        reasons.append(f"DPF={row['DiabetesPedigreeFunction']:.2f} [>=0.5, +1]")

    # Rule 6: Blood Pressure (hypertension comorbidity)
    if row['BloodPressure'] >= 90:
        score += 1
        reasons.append(f"BP={row['BloodPressure']:.0f} [HYPERTENSIVE >=90, +1]")

    # Final decision
    prediction = 1 if score >= 2 else 0

    return prediction, score, reasons


# ── CELL 3: Run predictions on all patients ───────────────────
predictions = []
scores      = []
all_reasons = []

for _, row in X.iterrows():
    pred, score, reasons = rule_based_predict(row)
    predictions.append(pred)
    scores.append(score)
    all_reasons.append(reasons)

y_pred_rules = np.array(predictions)
risk_scores  = np.array(scores)

print("✅ Rule-based predictions complete")

# ── CELL 4: Compute All Benchmark Metrics ────────────────────
tn, fp, fn, tp = confusion_matrix(y, y_pred_rules).ravel()

accuracy    = accuracy_score(y, y_pred_rules)
precision   = precision_score(y, y_pred_rules)
recall      = recall_score(y, y_pred_rules)       # = Sensitivity
f1          = f1_score(y, y_pred_rules)
specificity = tn / (tn + fp)
auc         = roc_auc_score(y, risk_scores)        # use score, not binary pred

# Error severity: classify each False Negative and False Positive
false_negatives = ((y_pred_rules == 0) & (y == 1)).sum()  # Missed diabetics — DANGEROUS
false_positives = ((y_pred_rules == 1) & (y == 0)).sum()  # Flagged healthy — less dangerous

print("\n" + "="*55)
print("RULE-BASED SYSTEM — BENCHMARK RESULTS")
print("="*55)
print(f"  Accuracy         : {accuracy:.4f}  ({accuracy*100:.1f}%)")
print(f"  Precision        : {precision:.4f}")
print(f"  Recall/Sensitivity: {recall:.4f}")
print(f"  Specificity      : {specificity:.4f}")
print(f"  F1-Score         : {f1:.4f}")
print(f"  AUC-ROC          : {auc:.4f}")
print(f"\n  Confusion Matrix:")
print(f"    True Positive  : {tp}  (correctly flagged diabetics)")
print(f"    True Negative  : {tn}  (correctly cleared non-diabetics)")
print(f"    False Positive : {fp}  (healthy patients flagged — low severity)")
print(f"    False Negative : {fn}  (diabetics missed ← CLINICALLY DANGEROUS)")

# ── CELL 5: Save metrics to file ─────────────────────────────
import json, os
os.makedirs('outputs', exist_ok=True)

metrics_rule = {
    "system": "Rule-Based Expert System",
    "accuracy": round(accuracy, 4),
    "precision": round(precision, 4),
    "recall_sensitivity": round(recall, 4),
    "specificity": round(specificity, 4),
    "f1_score": round(f1, 4),
    "auc_roc": round(auc, 4),
    "true_positives": int(tp),
    "true_negatives": int(tn),
    "false_positives": int(fp),
    "false_negatives": int(fn),
    "error_severity_notes": {
        "false_negatives_count": int(fn),
        "clinical_risk": "HIGH — missed diabetics may go untreated",
        "false_positives_count": int(fp),
        "clinical_risk_fp": "LOW — triggers unnecessary follow-up but not harmful"
    }
}

with open('outputs/metrics_rule_based.json', 'w') as f:
    json.dump(metrics_rule, f, indent=2)

print("\n✅ Saved: outputs/metrics_rule_based.json")

# ── CELL 6: Visualizations ───────────────────────────────────

# 6a: Confusion Matrix Heatmap
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Rule-Based Expert System — Results', fontsize=14, fontweight='bold')

cm = confusion_matrix(y, y_pred_rules)
sns_colors = ['#DBEAFE', '#FEE2E2', '#FEE2E2', '#DCFCE7']
im = axes[0].imshow(cm, cmap='Blues')
axes[0].set_title('Confusion Matrix', fontweight='bold')
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')
axes[0].set_xticks([0, 1])
axes[0].set_yticks([0, 1])
axes[0].set_xticklabels(['Non-Diabetic', 'Diabetic'])
axes[0].set_yticklabels(['Non-Diabetic', 'Diabetic'])
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, str(cm[i, j]), ha='center', va='center',
                     fontsize=22, fontweight='bold',
                     color='white' if cm[i, j] > cm.max()/2 else 'black')
plt.colorbar(im, ax=axes[0])

# 6b: Metric bar chart
metric_names  = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1', 'AUC-ROC']
metric_values = [accuracy, precision, recall, specificity, f1, auc]
colors = ['#2563EB', '#7C3AED', '#DC2626', '#059669', '#D97706', '#1D4ED8']
bars = axes[1].barh(metric_names, metric_values, color=colors, alpha=0.85, edgecolor='white')
axes[1].set_xlim(0, 1.1)
axes[1].set_title('Performance Metrics', fontweight='bold')
axes[1].set_xlabel('Score')
for bar, val in zip(bars, metric_values):
    axes[1].text(val + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{val:.3f}', va='center', fontsize=10, fontweight='bold')
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('outputs/04_rule_based_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: outputs/04_rule_based_results.png")

# 6c: ROC Curve
fpr, tpr, _ = roc_curve(y, risk_scores)
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color='#DC2626', lw=2.5, label=f'Rule-Based (AUC = {auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.6, label='Random Classifier')
ax.fill_between(fpr, tpr, alpha=0.08, color='#DC2626')
ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=11)
ax.set_title('ROC Curve — Rule-Based Expert System', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('outputs/05_rule_based_roc.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: outputs/05_rule_based_roc.png")

# ── CELL 7: Sample predictions walkthrough ───────────────────
import seaborn as sns

print("\n" + "="*55)
print("SAMPLE PREDICTIONS (First 5 patients)")
print("="*55)
for i in range(5):
    pred = predictions[i]
    actual = y.iloc[i]
    score = scores[i]
    status = "✅ CORRECT" if pred == actual else "❌ WRONG"
    label = "Diabetic" if pred == 1 else "Non-Diabetic"
    actual_label = "Diabetic" if actual == 1 else "Non-Diabetic"
    print(f"\nPatient {i+1}: {status}")
    print(f"  Predicted: {label} (risk score: {score})")
    print(f"  Actual   : {actual_label}")
    print(f"  Rules triggered:")
    if all_reasons[i]:
        for r in all_reasons[i]:
            print(f"    → {r}")
    else:
        print("    → No risk rules triggered (score = 0)")

print("\n" + "="*55)
print("DAY 3–4 COMPLETE ✅")
print("="*55)
print("Outputs:")
print("  outputs/metrics_rule_based.json")
print("  outputs/04_rule_based_results.png")
print("  outputs/05_rule_based_roc.png")
print("\nNext → Open day5_6_ml_baseline.py")