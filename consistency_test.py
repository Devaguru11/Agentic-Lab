# =============================================================
# DAY 12-13: Decision Consistency Test for LLM Agent
# AI Benchmarking Project — Week 2
# =============================================================
# This is your KEY RESEARCH FINDING:
#   Week 1 baseline: 100% consistency (all deterministic)
#   LLM agent: expect 70-90% consistency (non-deterministic)
#
# This consistency gap is the core trade-off of agentic AI
# in clinical settings — your paper's novel contribution.
# =============================================================

import pandas as pd
import numpy as np
import json
import os
import time
from groq import Groq
from llm_agent import run_agent, GROQ_API_KEY

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

os.makedirs('outputs', exist_ok=True)

print("\n" + "="*60)
print("DAY 12-13: LLM AGENT CONSISTENCY TEST")
print("="*60)

# ══════════════════════════════════════════════════════════════
# CONSISTENCY TEST
# ══════════════════════════════════════════════════════════════

def run_consistency_test(n_patients=20, n_runs=3):
    """
    Run the same patients through the LLM agent multiple times
    to measure decision consistency.
    
    Args:
        n_patients: Number of patients to test
        n_runs: Number of times to run each patient
    
    Returns:
        DataFrame with results, consistency rate
    """
    print(f"\n📊 Running consistency test:")
    print(f"   Patients: {n_patients}")
    print(f"   Runs per patient: {n_runs}")
    print(f"   Total API calls: {n_patients * n_runs}\n")
    
    # Load test data
    df_clean = pd.read_csv('data/pima_diabetes_clean.csv')
    test_size = int(len(df_clean) * 0.25)
    df_test = df_clean.iloc[-test_size:].reset_index(drop=True)
    
    # Sample patients (use fixed seed for reproducibility)
    df_sample = df_test.sample(n=n_patients, random_state=99)
    
    all_results = []
    
    for patient_idx, (idx, row) in enumerate(df_sample.iterrows()):
        patient_data = {
            'Pregnancies': row['Pregnancies'],
            'Glucose': row['Glucose'],
            'BloodPressure': row['BloodPressure'],
            'SkinThickness': row['SkinThickness'],
            'Insulin': row['Insulin'],
            'BMI': row['BMI'],
            'DiabetesPedigreeFunction': row['DiabetesPedigreeFunction'],
            'Age': row['Age']
        }
        
        true_label = row['Outcome']
        
        print(f"  Patient {patient_idx+1}/{n_patients} (True: {'DIABETIC' if true_label==1 else 'NON_DIABETIC'}): ", end='')
        
        # Run N times on same patient
        predictions = []
        confidences = []
        
        for run in range(n_runs):
            result = run_agent(patient_data)
            pred_binary = 1 if result['prediction'] == 'DIABETIC' else 0
            predictions.append(pred_binary)
            confidences.append(result['confidence'])
            time.sleep(0.5)  # Small delay to avoid rate limits
        
        # Check consistency
        all_same = len(set(predictions)) == 1
        
        patient_result = {
            'patient_id': patient_idx,
            'true_label': int(true_label),
            'run_1': predictions[0],
            'run_2': predictions[1],
            'run_3': predictions[2],
            'consistent': all_same,
            'confidence_mean': np.mean(confidences),
            'confidence_std': np.std(confidences)
        }
        
        all_results.append(patient_result)
        
        status = "✓ Consistent" if all_same else "✗ INCONSISTENT"
        preds_str = f"[{predictions[0]}, {predictions[1]}, {predictions[2]}]"
        print(f"{status} {preds_str}")
    
    results_df = pd.DataFrame(all_results)
    
    # Calculate consistency rate
    consistency_rate = results_df['consistent'].mean() * 100
    
    return results_df, consistency_rate

# ══════════════════════════════════════════════════════════════
# RUN TEST
# ══════════════════════════════════════════════════════════════

results_df, consistency_rate = run_consistency_test(n_patients=20, n_runs=3)

# Save results
results_df.to_csv('outputs/llm_consistency_results.csv', index=False)
print(f"\n✅ Saved: outputs/llm_consistency_results.csv")

# Create consistency heatmap data (for visualization)
consistency_matrix = results_df[['run_1', 'run_2', 'run_3']].values
np.save('outputs/llm_consistency_matrix.npy', consistency_matrix)

# ══════════════════════════════════════════════════════════════
# COMPUTE CONSISTENCY METRICS
# ══════════════════════════════════════════════════════════════

metrics_consistency = {
    "llm_agent": {
        "decision_consistency_rate": f"{consistency_rate:.1f}%",
        "n_patients_tested": len(results_df),
        "n_runs_per_patient": 3,
        "n_consistent": int(results_df['consistent'].sum()),
        "n_inconsistent": int((~results_df['consistent']).sum()),
        "consistency_interpretation": "Non-deterministic (temperature > 0)" if consistency_rate < 100 else "Fully consistent",
        "avg_confidence": round(results_df['confidence_mean'].mean(), 4),
        "confidence_std": round(results_df['confidence_std'].mean(), 4)
    },
    "baseline_comparison": {
        "rule_based": "100.0% (deterministic)",
        "logistic_regression": "100.0% (deterministic)",
        "random_forest": "100.0% (deterministic)",
        "llm_agent": f"{consistency_rate:.1f}%"
    },
    "note": "This consistency gap is the core trade-off of agentic AI in clinical settings. LLMs provide reasoning ability but sacrifice determinism."
}

with open('outputs/metrics_llm_consistency.json', 'w') as f:
    json.dump(metrics_consistency, f, indent=2)

print("\n" + "="*60)
print("CONSISTENCY TEST RESULTS")
print("="*60)
print(f"  LLM Agent Consistency: {consistency_rate:.1f}%")
print(f"  Consistent predictions: {results_df['consistent'].sum()}/{len(results_df)}")
print(f"  Inconsistent predictions: {(~results_df['consistent']).sum()}/{len(results_df)}")
print(f"\n  Baseline Comparison:")
print(f"    Rule-Based:           100.0%")
print(f"    Logistic Regression:  100.0%")
print(f"    Random Forest:        100.0%")
print(f"    LLM Agent:            {consistency_rate:.1f}%")
print(f"\n  Consistency Gap:      {100 - consistency_rate:.1f} percentage points")

print("\n✅ Saved: outputs/metrics_llm_consistency.json")

# ══════════════════════════════════════════════════════════════
# ANALYSIS
# ══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("CLINICAL INTERPRETATION")
print("="*60)

if consistency_rate >= 90:
    print("  HIGH consistency (≥90%): LLM agent is reasonably reliable")
    print("  for clinical use, though not perfect.")
elif consistency_rate >= 70:
    print("  MODERATE consistency (70-90%): LLM agent shows some")
    print("  variability. Acceptable for screening, but not for")
    print("  final diagnosis without human review.")
else:
    print("  LOW consistency (<70%): LLM agent is too unreliable")
    print("  for clinical deployment without significant guardrails.")

print("\n  Key Finding:")
print("  This consistency gap represents the fundamental trade-off")
print("  of agentic AI systems: they gain reasoning ability but")
print("  sacrifice the determinism of traditional ML models.")

print("\n" + "="*60)
print("DAY 12-13 COMPLETE ✅")
print("="*60)
print("Next → Run day14_final_comparison.py to generate final report")