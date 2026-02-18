# AI Benchmarking — Project README

## Overview

This repository contains lightweight experiments and scripts for benchmarking models on the PIMA Indians Diabetes dataset. It includes a dataset setup and EDA step and a simple rule-based expert system for baseline evaluation. The project is being updated over time.

## Repository structure

- `Setup_eda.py` — dataset download, cleaning, and EDA (produces cleaned CSV and plot images in `outputs/`).
- `Rule_based.py` — rule-based diabetes prediction logic and evaluation (writes `outputs/metrics_rule_based.json`).
- `data/` — raw and cleaned datasets (`pima_diabetes.csv`, `pima_diabetes_clean.csv`).
- `outputs/` — generated images and metrics (plots, `metrics_rule_based.json`).
- `README_Setup.md` — details for the dataset setup and EDA step.
- `README_rule_based.md` — details for the rule-based model and sample metrics.

## Requirements

Python 3.8+ and common data science libraries. Example packages:

- `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `requests`

If you maintain a `requirements.txt`, install with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or install minimal packages directly:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn requests
```

## Quick start — run the two main steps

1) Dataset setup & EDA

```bash
python Setup_eda.py
```

This will ensure `data/pima_diabetes.csv` exists (download if needed), write `data/pima_diabetes_clean.csv`, and save EDA plots to `outputs/`.

2) Rule-based baseline

```bash
python Rule_based.py
```

Outputs: `outputs/metrics_rule_based.json` with evaluation metrics (accuracy, precision, recall, AUC, TP/TN/FP/FN counts).

## Notes & next steps

- The repository is intended to be expanded: add model baselines, notebooks, and CI as needed.
- I will keep this README and the per-module READMEs (`README_Setup.md`, `README_rule_based.md`) updated over time.

## Contributing

Open an issue or submit a pull request with suggested changes. If you want, I can add a `requirements.txt` and example notebook next.
