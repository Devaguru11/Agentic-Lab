# =============================================================
# DAY 5–6: Single-Model AI Baselines
#           Logistic Regression + Random Forest
# AI Benchmarking Project — Week 1
# =============================================================
# These are your BASELINE 2 and BASELINE 3 systems.
# Standard supervised ML — no reasoning, no planning.
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json, os, time

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, roc_auc_score,
                              roc_curve, classification_report)
from sklearn.calibration import calibration_curve

try:
    from sklearn.metrics import CalibrationDisplay
except ImportError:
    CalibrationDisplay = None
import warnings
warnings.filterwarnings('ignore')

os.makedirs('outputs', exist_ok=True)

# ── CELL 1: Load & Split Data ────────────────────────────────
df = pd.read_csv('data/pima_diabetes_clean.csv')
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Stratified split — preserves class ratio in train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"✅ Data split:")
print(f"   Training  : {len(X_train)} patients")
print(f"   Testing   : {len(X_test)} patients")
print(f"   Diabetic % in test: {y_test.mean()*100:.1f}%")

# ── CELL 2: Scale features (required for Logistic Regression) ─
scaler  = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("✅ Features scaled with StandardScaler")

# ── CELL 3: Helper — compute all benchmark metrics ────────────
def compute_all_metrics(system_name, y_true, y_pred, y_proba, train_time):
    """Compute the full set of benchmark metrics for one system."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    accuracy    = accuracy_score(y_true, y_pred)
    precision   = precision_score(y_true, y_pred)
    recall      = recall_score(y_true, y_pred)
    f1          = f1_score(y_true, y_pred)
    specificity = tn / (tn + fp)
    auc         = roc_auc_score(y_true, y_proba)

    metrics = {
        "system": system_name,
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
        "training_time_seconds": round(train_time, 4),
        "error_severity_notes": {
            "false_negatives_count": int(fn),
            "clinical_risk": "HIGH — missed diabetics may go untreated",
            "false_positives_count": int(fp),
            "clinical_risk_fp": "LOW — triggers unnecessary follow-up"
        }
    }
    return metrics, accuracy, precision, recall, specificity, f1, auc, (tn, fp, fn, tp)


# ══════════════════════════════════════════════════════════════
# MODEL 1: LOGISTIC REGRESSION
# ══════════════════════════════════════════════════════════════

print("\n" + "="*55)
print("MODEL 1: LOGISTIC REGRESSION")
print("="*55)

t0 = time.time()
lr_model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
lr_model.fit(X_train_scaled, y_train)
lr_train_time = time.time() - t0

lr_pred  = lr_model.predict(X_test_scaled)
lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

lr_metrics, lr_acc, lr_prec, lr_rec, lr_spec, lr_f1, lr_auc, lr_cm = \
    compute_all_metrics("Logistic Regression", y_test, lr_pred, lr_proba, lr_train_time)

print(f"  Accuracy     : {lr_acc:.4f}")
print(f"  Sensitivity  : {lr_rec:.4f}")
print(f"  Specificity  : {lr_spec:.4f}")
print(f"  F1-Score     : {lr_f1:.4f}")
print(f"  AUC-ROC      : {lr_auc:.4f}")
print(f"  Train time   : {lr_train_time:.3f}s")

# Cross-validation (5-fold) for robustness
cv_scores = cross_val_score(lr_model, X_train_scaled, y_train, cv=5, scoring='f1')
lr_metrics['cross_val_f1_mean'] = round(cv_scores.mean(), 4)
lr_metrics['cross_val_f1_std']  = round(cv_scores.std(), 4)
print(f"  CV F1 (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

with open('outputs/metrics_logistic_regression.json', 'w') as f:
    json.dump(lr_metrics, f, indent=2)
print("✅ Saved: outputs/metrics_logistic_regression.json")


# ── Feature Importance (Coefficients) for LR ─────────────────
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lr_model.coef_[0]
}).sort_values('Coefficient', key=abs, ascending=True)

fig, ax = plt.subplots(figsize=(9, 5))
colors = ['#DC2626' if c > 0 else '#2563EB' for c in coef_df['Coefficient']]
ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, alpha=0.85, edgecolor='white')
ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_title('Logistic Regression — Feature Coefficients\n(Red = increases risk, Blue = decreases risk)',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Coefficient Value')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('outputs/06_lr_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: outputs/06_lr_feature_importance.png")


# ══════════════════════════════════════════════════════════════
# MODEL 2: RANDOM FOREST
# ══════════════════════════════════════════════════════════════

print("\n" + "="*55)
print("MODEL 2: RANDOM FOREST")
print("="*55)

t0 = time.time()
rf_model = RandomForestClassifier(
    n_estimators=200, max_depth=8, min_samples_split=5,
    random_state=42, n_jobs=-1
)
rf_model.fit(X_train, y_train)   # RF does not need scaling
rf_train_time = time.time() - t0

rf_pred  = rf_model.predict(X_test)
rf_proba = rf_model.predict_proba(X_test)[:, 1]

rf_metrics, rf_acc, rf_prec, rf_rec, rf_spec, rf_f1, rf_auc, rf_cm = \
    compute_all_metrics("Random Forest", y_test, rf_pred, rf_proba, rf_train_time)

print(f"  Accuracy     : {rf_acc:.4f}")
print(f"  Sensitivity  : {rf_rec:.4f}")
print(f"  Specificity  : {rf_spec:.4f}")
print(f"  F1-Score     : {rf_f1:.4f}")
print(f"  AUC-ROC      : {rf_auc:.4f}")
print(f"  Train time   : {rf_train_time:.3f}s")

cv_scores_rf = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='f1')
rf_metrics['cross_val_f1_mean'] = round(cv_scores_rf.mean(), 4)
rf_metrics['cross_val_f1_std']  = round(cv_scores_rf.std(), 4)
print(f"  CV F1 (5-fold): {cv_scores_rf.mean():.4f} ± {cv_scores_rf.std():.4f}")

with open('outputs/metrics_random_forest.json', 'w') as f:
    json.dump(rf_metrics, f, indent=2)
print("✅ Saved: outputs/metrics_random_forest.json")


# ── Feature Importance for RF (SHAP-style bar) ───────────────
fi_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=True)

fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(fi_df['Feature'], fi_df['Importance'], color='#7C3AED', alpha=0.85, edgecolor='white')
ax.set_title('Random Forest — Feature Importances\n(Higher = more important for prediction)',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Importance Score')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for bar, val in zip(ax.patches, fi_df['Importance']):
    ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig('outputs/07_rf_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: outputs/07_rf_feature_importance.png")


# ══════════════════════════════════════════════════════════════
# SIDE-BY-SIDE COMPARISON: Both Models
# ══════════════════════════════════════════════════════════════

print("\n" + "="*55)
print("SIDE-BY-SIDE MODEL COMPARISON")
print("="*55)

metrics_labels = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1', 'AUC-ROC']
lr_vals = [lr_acc, lr_prec, lr_rec, lr_spec, lr_f1, lr_auc]
rf_vals = [rf_acc, rf_prec, rf_rec, rf_spec, rf_f1, rf_auc]

for m, l, r in zip(metrics_labels, lr_vals, rf_vals):
    winner = "LR ✓" if l > r else "RF ✓" if r > l else "TIE"
    print(f"  {m:<15} LR: {l:.4f}   RF: {r:.4f}   → {winner}")

# Grouped bar chart
x = np.arange(len(metrics_labels))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
b1 = ax.bar(x - width/2, lr_vals, width, label='Logistic Regression', color='#2563EB', alpha=0.85, edgecolor='white')
b2 = ax.bar(x + width/2, rf_vals, width, label='Random Forest',       color='#7C3AED', alpha=0.85, edgecolor='white')

ax.set_ylabel('Score', fontsize=11)
ax.set_title('Logistic Regression vs Random Forest — All Metrics', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics_labels, fontsize=10)
ax.set_ylim(0, 1.12)
ax.legend(fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for bar in [*b1, *b2]:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.3f}',
            ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/08_model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: outputs/08_model_comparison.png")


# ── Overlaid ROC Curves ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
for (proba, label, color) in [
    (lr_proba, f'Logistic Regression (AUC={lr_auc:.3f})', '#2563EB'),
    (rf_proba, f'Random Forest (AUC={rf_auc:.3f})', '#7C3AED'),
]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    ax.plot(fpr, tpr, lw=2.5, label=label, color=color)
    ax.fill_between(fpr, tpr, alpha=0.05, color=color)

ax.plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.5, label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=11)
ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=11)
ax.set_title('ROC Curves — ML Baselines', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('outputs/09_ml_roc_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: outputs/09_ml_roc_curves.png")

print("\n" + "="*55)
print("DAY 5–6 COMPLETE ✅")
print("="*55)
print("Outputs:")
print("  outputs/metrics_logistic_regression.json")
print("  outputs/metrics_random_forest.json")
print("  outputs/06_lr_feature_importance.png")
print("  outputs/07_rf_feature_importance.png")
print("  outputs/08_model_comparison.png")
print("  outputs/09_ml_roc_curves.png")
print("\nNext → Open day7_calibration_consistency.py")