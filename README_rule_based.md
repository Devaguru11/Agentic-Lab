# Rule-Based Model — README

## Overview

This README documents the rule-based diabetes prediction script and its outputs. The rule-based logic is implemented in [Rule_based.py](Rule_based.py) and produces evaluation metrics saved to [outputs/metrics_rule_based.json](outputs/metrics_rule_based.json).

## Files

- [Rule_based.py](Rule_based.py): Rule-based prediction script.
- [data/pima_diabetes_clean.csv](data/pima_diabetes_clean.csv): Cleaned dataset used by the script.
- [outputs/metrics_rule_based.json](outputs/metrics_rule_based.json): Evaluation metrics produced after running the script.

## Requirements

This project uses Python 3.8+ and standard data libraries. Typical environment setup:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # if you maintain one; otherwise install pandas/scikit-learn
pip install pandas scikit-learn
```

## Usage

Run the rule-based script from the project root:

```
python Rule_based.py
```

On success the script will write evaluation metrics to `outputs/metrics_rule_based.json` and may print a short summary to stdout.

## Example output

Below is the sample content from `outputs/metrics_rule_based.json` produced during the latest run:

```json
{
  "system": "Rule-Based Expert System",
  "accuracy": 0.6185,
  "precision": 0.4762,
  "recall_sensitivity": 0.9328,
  "specificity": 0.45,
  "f1_score": 0.6305,
  "auc_roc": 0.7949,
  "true_positives": 250,
  "true_negatives": 225,
  "false_positives": 275,
  "false_negatives": 18,
  "error_severity_notes": {
    "false_negatives_count": 18,
    "clinical_risk": "HIGH — missed diabetics may go untreated",
    "false_positives_count": 275,
    "clinical_risk_fp": "LOW — triggers unnecessary follow-up but not harmful"
  }
}
```

## Notes

- The rule-based system shows high sensitivity but relatively low precision; tune or combine with statistical models as needed.
- If you want, I can update this README with example command-line options or include a small demo notebook.
