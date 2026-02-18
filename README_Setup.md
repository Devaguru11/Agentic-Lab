 # Dataset Setup & EDA — README

## Overview

This document describes the dataset setup and exploratory data analysis (EDA) step used in the AI benchmarking project. The workflow downloads (if needed), cleans, and saves the PIMA Indians Diabetes dataset and produces a small set of EDA visualizations.

## Files

- [Setup_eda.py](Setup_eda.py): Script that downloads the raw dataset (if missing), performs cleaning (replace medically-invalid zeros with medians), and saves EDA figures.
- [data/pima_diabetes.csv](data/pima_diabetes.csv): Raw dataset (downloaded or provided).
- [data/pima_diabetes_clean.csv](data/pima_diabetes_clean.csv): Cleaned dataset produced by the script.
- [outputs/](outputs/): Contains generated figures such as `01_class_distribution.png`, `02_feature_distributions.png`, and `03_correlation_heatmap.png`.

## Requirements

Requires Python 3.8+ and common data libraries. Example installation:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # if present
pip install pandas numpy matplotlib seaborn requests
```

## Usage

Run the setup and EDA script from the project root:

```bash
python Setup_eda.py
```

On success the script will ensure `data/pima_diabetes.csv` exists (download if necessary), write `data/pima_diabetes_clean.csv`, and save plots to the `outputs/` directory.

## Example outputs

- `data/pima_diabetes_clean.csv` — cleaned CSV.
- `outputs/01_class_distribution.png` — target class counts.
- `outputs/02_feature_distributions.png` — feature histograms / KDEs.
- `outputs/03_correlation_heatmap.png` — feature correlation heatmap.

## Notes

- The `outputs/` directory is often excluded from version control. To include generated images on GitHub, add a placeholder file (for example, `outputs/.gitkeep`) or remove the folder from `.gitignore`.
- If the script references a different filename for the setup script (for example `day1_2_setup_eda.py`), update the command above to match the actual file in the repository.
