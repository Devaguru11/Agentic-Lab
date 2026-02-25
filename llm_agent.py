# =============================================================
# DAY 8-11: Agentic LLM System with Groq
# AI Benchmarking Project — Week 2
# =============================================================
# This script builds a ReAct-style LLM agent that:
#   1. Reads patient clinical features
#   2. Reasons step-by-step about diabetes risk
#   3. Uses tool calling to check clinical thresholds
#   4. Outputs final prediction with explanation
# =============================================================

import pandas as pd
import numpy as np
import json
import os
import time
from groq import Groq
from typing import Dict, List, Tuple

# ══════════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════════

# Initialize Groq client
# IMPORTANT: Replace with your actual Groq API key
GROQ_API_KEY = "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"  # ← PUT YOUR KEY HERE
client = Groq(api_key=GROQ_API_KEY)

os.makedirs('data', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

print("✅ Groq client initialized")

# ══════════════════════════════════════════════════════════════
# TOOL DEFINITIONS
# ══════════════════════════════════════════════════════════════

# Define tools the LLM can use to check clinical thresholds
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_glucose_threshold",
            "description": "Check if patient's glucose level exceeds clinical diabetes thresholds (WHO guidelines: ≥140 mg/dL = impaired glucose tolerance, ≥126 mg/dL = pre-diabetic)",
            "parameters": {
                "type": "object",
                "properties": {
                    "glucose": {
                        "type": "number",
                        "description": "Patient's glucose level in mg/dL"
                    }
                },
                "required": ["glucose"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_bmi_category",
            "description": "Check patient's BMI category according to WHO classification (≥30 = Obese, 25-29.9 = Overweight, 18.5-24.9 = Normal)",
            "parameters": {
                "type": "object",
                "properties": {
                    "bmi": {
                        "type": "number",
                        "description": "Patient's Body Mass Index (BMI)"
                    }
                },
                "required": ["bmi"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_age_risk",
            "description": "Check if patient is in high-risk age group for diabetes (ADA guidelines: ≥45 years = increased risk)",
            "parameters": {
                "type": "object",
                "properties": {
                    "age": {
                        "type": "number",
                        "description": "Patient's age in years"
                    }
                },
                "required": ["age"]
            }
        }
    }
]

# ══════════════════════════════════════════════════════════════
# TOOL IMPLEMENTATIONS
# ══════════════════════════════════════════════════════════════

def check_glucose_threshold(glucose: float) -> dict:
    """Check glucose against diabetes thresholds."""
    if glucose >= 140:
        return {
            "status": "HIGH_RISK",
            "category": "Impaired glucose tolerance",
            "risk_level": "high",
            "message": f"Glucose {glucose} mg/dL is ≥140, indicating impaired glucose tolerance (WHO criterion)"
        }
    elif glucose >= 126:
        return {
            "status": "ELEVATED_RISK",
            "category": "Pre-diabetic range",
            "risk_level": "moderate",
            "message": f"Glucose {glucose} mg/dL is ≥126, in pre-diabetic range (ADA criterion)"
        }
    else:
        return {
            "status": "NORMAL",
            "category": "Normal glucose",
            "risk_level": "low",
            "message": f"Glucose {glucose} mg/dL is within normal range"
        }

def check_bmi_category(bmi: float) -> dict:
    """Classify BMI according to WHO standards."""
    if bmi >= 30:
        return {
            "category": "Obese",
            "risk_level": "high",
            "message": f"BMI {bmi:.1f} is ≥30 (Obese — significant diabetes risk factor)"
        }
    elif bmi >= 25:
        return {
            "category": "Overweight",
            "risk_level": "moderate",
            "message": f"BMI {bmi:.1f} is 25-29.9 (Overweight — moderate diabetes risk)"
        }
    else:
        return {
            "category": "Normal",
            "risk_level": "low",
            "message": f"BMI {bmi:.1f} is <25 (Normal weight)"
        }

def check_age_risk(age: int) -> dict:
    """Check if age is in high-risk group."""
    if age >= 45:
        return {
            "risk_level": "high",
            "message": f"Age {age} is ≥45 years (ADA high-risk age group)"
        }
    else:
        return {
            "risk_level": "low",
            "message": f"Age {age} is <45 years (lower risk age group)"
        }

# Tool dispatcher
TOOL_FUNCTIONS = {
    "check_glucose_threshold": check_glucose_threshold,
    "check_bmi_category": check_bmi_category,
    "check_age_risk": check_age_risk
}

# ══════════════════════════════════════════════════════════════
# LLM AGENT
# ══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a clinical decision support AI assisting with diabetes risk assessment. 

Your task:
1. Analyze the patient's clinical features
2. Use the available tools to check medical thresholds
3. Reason step-by-step about diabetes risk factors
4. Provide a final prediction: DIABETIC or NON_DIABETIC
5. Explain your reasoning clearly

Guidelines:
- Call tools to verify clinical thresholds (don't guess values)
- Consider multiple risk factors together
- High glucose (≥140) is the strongest predictor
- Obesity (BMI ≥30), age ≥45, and high pregnancy count increase risk
- Be conservative: when in doubt, flag as higher risk (patient safety first)

Output format:
After your analysis, end with:
FINAL_PREDICTION: [DIABETIC or NON_DIABETIC]
CONFIDENCE: [0.0 to 1.0]
"""

def run_agent(patient_data: dict, max_turns: int = 5) -> dict:
    """
    Run the LLM agent with tool calling on a single patient.
    
    Returns:
        dict with keys: prediction, confidence, reasoning, tool_calls, time_taken
    """
    start_time = time.time()
    
    # Format patient data as natural language
    patient_text = f"""Patient Profile:
- Pregnancies: {patient_data['Pregnancies']}
- Glucose: {patient_data['Glucose']} mg/dL
- Blood Pressure: {patient_data['BloodPressure']} mmHg
- Skin Thickness: {patient_data['SkinThickness']} mm
- Insulin: {patient_data['Insulin']} mu U/ml
- BMI: {patient_data['BMI']}
- Diabetes Pedigree Function: {patient_data['DiabetesPedigreeFunction']} (family history score)
- Age: {patient_data['Age']} years

Question: Based on this patient's clinical profile, assess their diabetes risk. Use the available tools to check thresholds, then provide your final prediction."""
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": patient_text}
    ]
    
    tool_calls_log = []
    reasoning_log = []
    
    # ReAct loop: LLM can call tools multiple times
    for turn in range(max_turns):
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                max_tokens=1000,
                temperature=0.3  # Lower temp for more consistent medical decisions
            )
            
            assistant_message = response.choices[0].message
            
            # If LLM wants to call tools
            if assistant_message.tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": assistant_message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in assistant_message.tool_calls
                    ]
                })
                
                # Execute each tool call
                for tool_call in assistant_message.tool_calls:
                    func_name = tool_call.function.name
                    func_args = json.loads(tool_call.function.arguments)
                    
                    # Call the actual tool function
                    result = TOOL_FUNCTIONS[func_name](**func_args)
                    
                    tool_calls_log.append({
                        "tool": func_name,
                        "args": func_args,
                        "result": result
                    })
                    
                    # Send tool result back to LLM
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result)
                    })
            
            # If LLM gives final answer (no more tool calls)
            else:
                reasoning = assistant_message.content
                reasoning_log.append(reasoning)
                
                # Parse final prediction
                prediction, confidence = parse_prediction(reasoning)
                
                time_taken = time.time() - start_time
                
                return {
                    "prediction": prediction,
                    "confidence": confidence,
                    "reasoning": "\n\n".join(reasoning_log),
                    "tool_calls": tool_calls_log,
                    "time_taken": time_taken,
                    "turns": turn + 1
                }
        
        except Exception as e:
            print(f"⚠️  Error on turn {turn}: {e}")
            # Return conservative default on error
            return {
                "prediction": "DIABETIC",  # Conservative: flag as risk
                "confidence": 0.5,
                "reasoning": f"Error occurred: {str(e)}",
                "tool_calls": tool_calls_log,
                "time_taken": time.time() - start_time,
                "turns": turn + 1,
                "error": str(e)
            }
    
    # Max turns reached without final answer
    return {
        "prediction": "DIABETIC",  # Conservative default
        "confidence": 0.5,
        "reasoning": "Max turns reached without final decision",
        "tool_calls": tool_calls_log,
        "time_taken": time.time() - start_time,
        "turns": max_turns
    }

def parse_prediction(text: str) -> Tuple[str, float]:
    """
    Extract prediction and confidence from LLM output.
    
    Expected format:
        FINAL_PREDICTION: DIABETIC
        CONFIDENCE: 0.85
    """
    prediction = "DIABETIC"  # Default conservative
    confidence = 0.5
    
    lines = text.upper().split('\n')
    for line in lines:
        if 'FINAL_PREDICTION' in line:
            if 'NON_DIABETIC' in line or 'NON-DIABETIC' in line or 'NOT DIABETIC' in line:
                prediction = "NON_DIABETIC"
            elif 'DIABETIC' in line:
                prediction = "DIABETIC"
        
        if 'CONFIDENCE' in line:
            # Extract number between 0 and 1
            try:
                parts = line.split(':')
                if len(parts) > 1:
                    conf_str = parts[1].strip().replace('%', '')
                    conf_val = float(conf_str)
                    # If given as percentage, convert to 0-1 range
                    if conf_val > 1:
                        conf_val = conf_val / 100.0
                    confidence = max(0.0, min(1.0, conf_val))
            except:
                pass
    
    return prediction, confidence

# ══════════════════════════════════════════════════════════════
# RUN ON TEST SET
# ══════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*60)
    print("WEEK 2: LLM AGENT EVALUATION")
    print("="*60)
    
    # Load clean dataset from Week 1
    df_clean = pd.read_csv('data/pima_diabetes_clean.csv')
    print(f"\n✅ Loaded {len(df_clean)} patients")
    
    # Use same test split as Week 1 (last 25%)
    test_size = int(len(df_clean) * 0.25)
    df_test = df_clean.iloc[-test_size:].reset_index(drop=True)
    
    # For Week 2, evaluate on 50 patients (to save API calls and time)
    # You can increase this to 100 or the full test set if desired
    N_PATIENTS = 50
    df_sample = df_test.sample(n=min(N_PATIENTS, len(df_test)), random_state=42)
    
    print(f"📊 Running agent on {len(df_sample)} patients...")
    print("   (This will take ~2-3 minutes)\n")
    
    results = []
    
    for idx, row in df_sample.iterrows():
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
        
        print(f"  Patient {len(results)+1}/{len(df_sample)}: ", end='')
        
        # Run agent
        result = run_agent(patient_data)
        
        # Convert prediction to binary
        pred_binary = 1 if result['prediction'] == 'DIABETIC' else 0
        
        result['true_label'] = int(true_label)
        result['pred_binary'] = pred_binary
        result['correct'] = (pred_binary == true_label)
        
        results.append(result)
        
        status = "✓" if result['correct'] else "✗"
        print(f"{status} [{result['prediction']}] (true: {'DIABETIC' if true_label == 1 else 'NON_DIABETIC'}) — {result['time_taken']:.2f}s")
    
    # Save detailed results
    results_df = pd.DataFrame(results)
    results_df.to_json('outputs/llm_agent_detailed_results.json', orient='records', indent=2)
    print(f"\n✅ Saved: outputs/llm_agent_detailed_results.json")
    
    # Compute metrics
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                  f1_score, confusion_matrix, roc_auc_score)
    
    y_true = results_df['true_label'].values
    y_pred = results_df['pred_binary'].values
    y_prob = results_df['confidence'].values
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    accuracy    = accuracy_score(y_true, y_pred)
    precision   = precision_score(y_true, y_pred, zero_division=0)
    recall      = recall_score(y_true, y_pred)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1          = f1_score(y_true, y_pred, zero_division=0)
    
    # AUC-ROC using confidence scores
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0.5
    
    metrics = {
        "system": "LLM Agent (Groq Llama 3.3 70B)",
        "model": "llama-3.3-70b-versatile",
        "n_patients_evaluated": len(results),
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
        "avg_time_per_patient": round(results_df['time_taken'].mean(), 4),
        "avg_confidence": round(results_df['confidence'].mean(), 4),
        "avg_tool_calls": round(results_df['tool_calls'].apply(len).mean(), 2),
        "error_severity_notes": {
            "false_negatives_count": int(fn),
            "clinical_risk": "HIGH — missed diabetics may go untreated",
            "false_positives_count": int(fp),
            "clinical_risk_fp": "LOW — triggers unnecessary follow-up"
        }
    }
    
    with open('outputs/metrics_llm_agent.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"  Accuracy     : {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"  Precision    : {precision:.4f}")
    print(f"  Sensitivity  : {recall:.4f}")
    print(f"  Specificity  : {specificity:.4f}")
    print(f"  F1-Score     : {f1:.4f}")
    print(f"  AUC-ROC      : {auc:.4f}")
    print(f"\n  True Positives  : {tp}")
    print(f"  True Negatives  : {tn}")
    print(f"  False Positives : {fp}")
    print(f"  False Negatives : {fn}")
    print(f"\n  Avg Time/Patient: {results_df['time_taken'].mean():.2f}s")
    print(f"  Avg Confidence  : {results_df['confidence'].mean():.3f}")
    print(f"  Avg Tool Calls  : {results_df['tool_calls'].apply(len).mean():.1f}")
    
    print("\n✅ Saved: outputs/metrics_llm_agent.json")
    print("\n" + "="*60)
    print("DAY 8-11 COMPLETE ✅")
    print("="*60)
    print("Next → Run day12_13_consistency_test.py")

if __name__ == "__main__":
    main()