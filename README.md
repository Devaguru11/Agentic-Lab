# AI Benchmarking — Day 1–2: Dataset Setup & EDA

Project summary
- Purpose: Collect and clean the PIMA Indians Diabetes dataset, run basic exploratory data analysis (EDA), and save visual outputs for benchmarking and modelling.
- What was done: Downloaded the dataset, replaced medically-invalid zeros with column medians, saved a cleaned CSV, and generated three EDA plots.

Quick links
- Main script: day1_2_setup_eda.py
- Raw data: data/pima_diabetes.csv
- Cleaned data: data/pima_diabetes_clean.csv
- Generated outputs: outputs/ (01_class_distribution.png, 02_feature_distributions.png, 03_correlation_heatmap.png)

How it works
- Download: Script downloads the dataset from a public URL if data/pima_diabetes.csv is missing.
- Cleaning: Replaces zeros in Glucose, BloodPressure, SkinThickness, Insulin, BMI with column medians to produce data/pima_diabetes_clean.csv.
- Visuals: Saves three plots to outputs/.

Dependencies
- pandas, numpy, matplotlib, seaborn, requests

Install
- pip install -r requirements.txt
  or
- pip install pandas numpy matplotlib seaborn requests

Run
- python3 day1_2_setup_eda.py

Notes about outputs/
- outputs/ is ignored by default. To show it on GitHub, add outputs/.gitkeep or remove outputs/ from .gitignore.
