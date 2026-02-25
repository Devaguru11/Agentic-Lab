# Week 3 — Agentic LLM System Code
## AI Benchmarking Research Project

This folder contains all the code for Week 3.

---

## 📋 Prerequisites

Before running these scripts, make sure you have:
1. ✅ Completed Week 2 (you should have `data/pima_diabetes_clean.csv`)
2. ✅ Groq API key (from console.groq.com)
3. ✅ Python packages installed

---

## 🔧 Setup

### Step 1: Install Required Packages
```bash
pip install groq pandas numpy scikit-learn
```

### Step 2: Add Your API Key
Open `llm_agent.py` and replace this line:
```python
GROQ_API_KEY = "YOUR_GROQ_API_KEY_HERE"
```
with your actual Groq API key:
```python
GROQ_API_KEY = "gsk_..."  # Your key here
```

---

## 🚀 Execution Order

Run the scripts in this exact order:

### **Build and Run LLM Agent**
```bash
python3 llm_agent.py
```
**What it does:**
- Loads your clean PIMA dataset
- Runs 50 patients through the LLM agent
- Each patient goes through a ReAct reasoning loop with tool calling
- Computes all metrics (accuracy, sensitivity, etc.)
- Saves: `outputs/metrics_llm_agent.json`

**Expected time:** 2-3 minutes  
**Expected output:** ~70-80% accuracy

---

### **Consistency Test**
```bash
python3 consistency_test.py
```
**What it does:**
- Takes 20 patients
- Runs each patient through the LLM agent 3 times
- Measures how often the LLM gives the SAME answer
- This is your KEY RESEARCH FINDING

**Expected time:** 3-4 minutes  
**Expected output:** 70-90% consistency (vs 100% for traditional ML)

---

### **Final Comparison**
```bash
python3 final_comparison.py
```
**What it does:**
- Loads metrics from all 4 systems (Week 2 + Week 3)
- Creates master comparison table
- Identifies winners for each metric
- Generates final findings summary

**Expected output:** `final_comparison_all_systems.csv`

---

## 📊 Output Files

After running all scripts, you'll have:

```
outputs/
├── metrics_llm_agent.json              ← LLM performance metrics
├── llm_agent_detailed_results.json     ← Per-patient predictions
├── metrics_llm_consistency.json        ← Consistency test results
├── llm_consistency_results.csv         ← Raw consistency data
├── final_comparison_all_systems.csv    ← Master table (all 4 systems)
└── final_findings_summary.json         ← Key findings for paper
```

---

## 🎯 What Makes This Code Special

### 1. **ReAct Prompting with Tool Use**
The LLM doesn't just predict — it:
- Observes patient features
- **Thinks** step-by-step about risk factors
- **Acts** by calling clinical threshold tools
- Outputs final prediction with reasoning

### 2. **Clinical Tool Calling**
Three tools available to the LLM:
- `check_glucose_threshold` — WHO/ADA glucose criteria
- `check_bmi_category` — WHO obesity classification
- `check_age_risk` — ADA age-based risk groups

### 3. **Conservative Error Handling**
If the LLM crashes or times out:
- Default prediction: **DIABETIC** (conservative — safer to flag risk)
- This prioritizes patient safety over accuracy

---

## 🔬 Expected Results

| Metric | Rule-Based | Logistic Reg | Random Forest | **LLM Agent** |
|--------|------------|--------------|---------------|---------------|
| **Accuracy** | 61.9% | 72.9% | 75.5% | **~70-80%** |
| **Sensitivity** | 93.3% | 52.2% | 55.2% | **~60-75%** |
| **Specificity** | 45.0% | 84.0% | 86.4% | **~70-85%** |
| **Consistency** | 100% | 100% | 100% | **~70-90%** ← KEY FINDING |

The LLM will be **competitive but not dominant** — and that's perfect for your paper!

---

## 💡 Tips for Success

1. **API Rate Limits:** If you hit rate limits, the code has built-in 0.5s delays
2. **Cost:** Total cost for 50 patients + consistency test ≈ **$1-2**
3. **Time:** Total runtime ≈ **5-10 minutes** for all scripts
4. **Reproducibility:** Set `temperature=0.3` for more consistency (already done)

---

## 🆘 Troubleshooting

### "ModuleNotFoundError: No module named 'groq'"
```bash
pip install groq
```

### "Error code: 400 - Invalid API key"
Check that you pasted your API key correctly in `llm_agent.py`

### "FileNotFoundError: data/pima_diabetes_clean.csv"
Run Week 2 code first — you need the cleaned dataset

### "Rate limit exceeded"
Wait 1 minute and try again, or reduce `N_PATIENTS` to 30 in the code

---

## 🎓 What Happens Next

After running all Week 3 code, come back to me and say:
**"Week 3 is done — generate final report"**

I'll create your complete research document combining:
- Week 2 results
- Week 3 results  
- Side-by-side comparison of all 4 systems
- Discussion of the consistency trade-off
- Conclusion and clinical recommendations
