# =============================================================
# DAY 7: Calibration & Decision Consistency Testing
# AI Benchmarking Project — Week 1
# =============================================================
# TWO KEY CLINICALLY RELEVANT TESTS:
#
# 1. CALIBRATION — Does the model's confidence match reality?
#    e.g. If it says "70% diabetic", is it right ~70% of the time?
#    A poorly calibrated model gives false confidence to clinicians.
#
# 2. DECISION CONSISTENCY — Does the model give the same answer
#    on identical inputs every time?
#    Deterministic models (LR, RF) should be 100% consistent.
#    This becomes critical when we test the LLM agent in Week 2.
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import json, os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

os.makedirs('outputs', exist_ok=True)

# ── CELL 1: Reload data & retrain models ─────────────────────
df = pd.read_csv('data/pima_diabetes_clean.csv')
X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

lr_model = LogisticRegression(max_iter=1000, random_state=42).fit(X_train_scaled, y_train)
rf_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1).fit(X_train, y_train)

print("✅ Models retrained")


# ══════════════════════════════════════════════════════════════
# PART 1: CALIBRATION ANALYSIS
# ══════════════════════════════════════════════════════════════

# ── CELL 2: Compute calibration curves ───────────────────────
lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
rf_proba = rf_model.predict_proba(X_test)[:, 1]

# calibration_curve bins probabilities into buckets and checks
# how often the actual positive rate matches the predicted rate
lr_frac_pos, lr_mean_pred = calibration_curve(y_test, lr_proba, n_bins=10, strategy='uniform')
rf_frac_pos, rf_mean_pred = calibration_curve(y_test, rf_proba, n_bins=10, strategy='uniform')

# Brier Score: lower = better calibrated (0 = perfect, 1 = worst)
lr_brier = brier_score_loss(y_test, lr_proba)
rf_brier = brier_score_loss(y_test, rf_proba)

print("\n" + "="*55)
print("CALIBRATION RESULTS")
print("="*55)
print(f"  Logistic Regression Brier Score: {lr_brier:.4f}")
print(f"  Random Forest Brier Score      : {rf_brier:.4f}")
print("  (Lower Brier Score = better calibrated)")
print("  Note: Perfect calibration = diagonal line in plot below")


# ── CELL 3: Calibration plots ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Calibration Reliability Plots\n(How well model confidence matches actual outcomes)',
             fontsize=13, fontweight='bold')

for ax, frac_pos, mean_pred, label, color, brier in [
    (axes[0], lr_frac_pos, lr_mean_pred, 'Logistic Regression', '#2563EB', lr_brier),
    (axes[1], rf_frac_pos, rf_mean_pred, 'Random Forest',       '#7C3AED', rf_brier),
]:
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Perfect Calibration', alpha=0.7)
    ax.plot(mean_pred, frac_pos, 'o-', color=color, lw=2.5, markersize=8, label=f'{label}\nBrier={brier:.4f}')
    ax.fill_between(mean_pred, frac_pos, mean_pred, alpha=0.1, color=color)
    ax.set_xlabel('Mean Predicted Probability', fontsize=10)
    ax.set_ylabel('Fraction of Positives (Actual)', fontsize=10)
    ax.set_title(label, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Annotate interpretation
    note = "Good calibration" if brier < 0.20 else "Overconfident — needs calibration"
    ax.text(0.5, 0.05, note, ha='center', fontsize=9,
            color='#15803D' if brier < 0.20 else '#DC2626',
            transform=ax.transAxes, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/10_calibration_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: outputs/10_calibration_curves.png")


# ══════════════════════════════════════════════════════════════
# PART 2: DECISION CONSISTENCY TESTING
# ══════════════════════════════════════════════════════════════
# Run the same 20 patients through each model 3 times.
# Deterministic models must give identical predictions each time.
# Inconsistency = unreliable clinical decision support.

print("\n" + "="*55)
print("DECISION CONSISTENCY TESTING")
print("="*55)
print("Running each model 3× on 20 identical patients...")

# Pick 20 random test patients
np.random.seed(99)
sample_idx = np.random.choice(len(X_test), size=20, replace=False)
X_sample        = X_test.iloc[sample_idx]
X_sample_scaled = X_test_scaled[sample_idx]

def run_consistency_test(model, inputs, n_runs=3, needs_scaling=False, scale_inputs=None):
    """Run model n_runs times on same inputs, check if predictions match."""
    all_preds = []
    for run in range(n_runs):
        if needs_scaling:
            preds = model.predict(scale_inputs)
        else:
            preds = model.predict(inputs)
        all_preds.append(preds)

    all_preds = np.array(all_preds)  # shape: (n_runs, n_samples)

    # Consistency: all runs identical for each patient
    consistency_per_patient = np.all(all_preds == all_preds[0], axis=0)
    consistency_rate = consistency_per_patient.mean() * 100

    return all_preds, consistency_rate, consistency_per_patient


lr_preds_runs, lr_consistency, lr_consistent = run_consistency_test(
    lr_model, X_sample, needs_scaling=True, scale_inputs=X_sample_scaled
)
rf_preds_runs, rf_consistency, rf_consistent = run_consistency_test(
    rf_model, X_sample, needs_scaling=False
)

print(f"\n  Logistic Regression consistency: {lr_consistency:.1f}%")
print(f"  Random Forest consistency      : {rf_consistency:.1f}%")
print("  (Expected: 100% for deterministic models)")

# ── CELL 5: Consistency heatmap ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Decision Consistency — 3 Runs on 20 Identical Patients\n(All cells same colour = perfectly consistent)',
             fontsize=12, fontweight='bold')

run_labels = ['Run 1', 'Run 2', 'Run 3']

for ax, preds, label, color, consistency in [
    (axes[0], lr_preds_runs, 'Logistic Regression', 'Blues', lr_consistency),
    (axes[1], rf_preds_runs, 'Random Forest',       'Purples', rf_consistency),
]:
    sns.heatmap(preds, ax=ax, cmap=color, vmin=0, vmax=1,
                xticklabels=[f'P{i+1}' for i in range(20)],
                yticklabels=run_labels,
                cbar_kws={'label': '0=Non-Diabetic, 1=Diabetic'},
                linewidths=0.5, linecolor='white')
    ax.set_title(f'{label}\nConsistency: {consistency:.1f}%', fontweight='bold')
    ax.set_xlabel('Patient')
    ax.set_ylabel('Run')

plt.tight_layout()
plt.savefig('outputs/11_consistency_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: outputs/11_consistency_heatmap.png")


# ── CELL 6: Save calibration & consistency metrics ───────────
calibration_consistency_metrics = {
    "logistic_regression": {
        "brier_score": round(lr_brier, 4),
        "calibration_interpretation": "Well-calibrated" if lr_brier < 0.20 else "Needs calibration",
        "decision_consistency_rate": f"{lr_consistency:.1f}%",
        "consistency_interpretation": "Fully consistent (deterministic)" if lr_consistency == 100 else "INCONSISTENT — investigate"
    },
    "random_forest": {
        "brier_score": round(rf_brier, 4),
        "calibration_interpretation": "Well-calibrated" if rf_brier < 0.20 else "Needs calibration",
        "decision_consistency_rate": f"{rf_consistency:.1f}%",
        "consistency_interpretation": "Fully consistent (deterministic)" if rf_consistency == 100 else "INCONSISTENT — investigate"
    },
    "note_for_week2": (
        "Agentic LLM systems are non-deterministic (temperature > 0). "
        "Expect consistency < 100%. This difference is a key research finding."
    )
}

with open('outputs/metrics_calibration_consistency.json', 'w') as f:
    json.dump(calibration_consistency_metrics, f, indent=2)
print("\n✅ Saved: outputs/metrics_calibration_consistency.json")


# ── CELL 7: Load all Week 1 metrics and print master summary ──
print("\n" + "="*55)
print("WEEK 1 — COMPLETE METRICS SUMMARY")
print("="*55)

files = [
    ('Rule-Based',           'outputs/metrics_rule_based.json'),
    ('Logistic Regression',  'outputs/metrics_logistic_regression.json'),
    ('Random Forest',        'outputs/metrics_random_forest.json'),
]

rows = []
for label, path in files:
    if os.path.exists(path):
        with open(path) as f:
            m = json.load(f)
        rows.append({
            'System':      label,
            'Accuracy':    m['accuracy'],
            'Sensitivity': m['recall_sensitivity'],
            'Specificity': m['specificity'],
            'F1':          m['f1_score'],
            'AUC-ROC':     m['auc_roc'],
        })

summary_df = pd.DataFrame(rows)
print(summary_df.to_string(index=False))
summary_df.to_csv('outputs/week1_metrics_summary.csv', index=False)
print("\n✅ Saved: outputs/week1_metrics_summary.csv")


# ── CELL 8: Final comparison bar chart ───────────────────────
metrics_to_plot = ['Accuracy', 'Sensitivity', 'Specificity', 'F1', 'AUC-ROC']
systems = summary_df['System'].tolist()
colors  = ['#DC2626', '#2563EB', '#7C3AED']

x      = np.arange(len(metrics_to_plot))
width  = 0.25
n      = len(systems)

fig, ax = plt.subplots(figsize=(13, 6))

for i, (system, color) in enumerate(zip(systems, colors)):
    row = summary_df[summary_df['System'] == system].iloc[0]
    vals = [row[m] for m in metrics_to_plot]
    offset = (i - n/2 + 0.5) * width
    bars = ax.bar(x + offset, vals, width, label=system, color=color, alpha=0.85, edgecolor='white')

ax.set_ylabel('Score', fontsize=11)
ax.set_title('Week 1 Baseline Comparison — All 3 Systems', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics_to_plot, fontsize=11)
ax.set_ylim(0, 1.12)
ax.legend(fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.axhline(0.75, color='gray', linestyle='--', lw=1, alpha=0.5, label='0.75 threshold')
ax.text(len(metrics_to_plot)-0.1, 0.755, 'Clinical target (0.75)', color='gray', fontsize=8)
plt.tight_layout()
plt.savefig('outputs/12_week1_final_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: outputs/12_week1_final_comparison.png")

print("\n" + "="*55)
print("DAY 7 — COMPLETE ✅")
print("WEEK 1 — COMPLETE ✅")
print("="*55)
print("\nAll outputs saved in: outputs/")
print("\nYour baselines are ready.")
print("Proceed to Week 2: Agentic LLM system design.")