
<table border="0">
 <tr>
    <td><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/University_of_Prishtina_logo.svg/1200px-University_of_Prishtina_logo.svg.png" width="150" alt="University Logo" /></td>
    <td>
      <p>University of Pristina</p>
      <p>Faculty of Electrical and Computer Engineering</p>
      <p>Computer and Software Engineering - Master Program</p>
	<p>Course: Data Preparation and Visualization</p>
      <p>Professor: Dr. Sc. Mërgim Hoti</p>
    </td>
 </tr>
</table>


# Social Media vs Productivity – ETL Data Preprocessing Pipeline

This project implements a **modular ETL (Extract, Transform, Load)** pipeline for analyzing how **social media usage** affects **productivity and stress levels**.
It is designed for **academic coursework and real-world data engineering**, applying professional preprocessing, cleaning, and feature engineering techniques.

---

## Overview

### Objective

To transform raw behavioral data into a structured, enriched dataset that supports **machine learning**, **statistical analysis**, and **visual analytics**.

### Workflow Summary

1. Extract raw data and profile it.
2. Validate and clean with logical quality checks.
3. Impute missing values using dependency relationships.
4. Encode and discretize categorical and numerical variables.
5. Engineer new behavioral and psychological indicators.
6. Reduce redundancy via correlation-based feature selection.
7. Export a clean, ready-to-analyze dataset.

---

## ETL Pipeline Structure

```text
etl/
├─ aggregation.py                 # Job-based productivity aggregation
├─ advanced_imputation.py        # Smart contextual imputation
├─ apply_binarization.py         # Binary feature creation
├─ apply_discretization.py       # Continuous variable binning
├─ column_names.py               # Centralized schema definitions
├─ data_quality.py               # Logical and consistency checks
├─ data_sampling.py              # Random / stratified sampling
├─ data_type_definition.py       # Explicit dtype enforcement
├─ dependency_map.py             # Imputation dependency rules
├─ feature_reduction_enhanced.py # Duplicate + correlation-based reduction
├─ features.py                   # Feature engineering metrics
├─ extract.py                    # Load and profile raw data
├─ selection.py                  # Advanced correlation/variance feature selector
├─ load.py                       # Save processed dataset
└─ main.py                       # Pipeline orchestration entry point
```

---

## Requirements

Install the required Python packages before running the project:

pip install pandas numpy scikit-learn

---

## How to Run

1. Place your raw dataset (e.g., social_media_vs_productivity.csv) inside the data/ folder.Example path:
   data/social_media_vs_productivity.csv
2. Open a terminal or command prompt in the project directory.
3. Run the ETL pipeline using:
   python etl/main.py
4. The processed dataset will be saved automatically to:
   data/processed_dataset.csv

---

Authors:
        Besarta Kurtaj,
        Enis Hoxha,
        Shefket Bylygbashi
Year: 2025
