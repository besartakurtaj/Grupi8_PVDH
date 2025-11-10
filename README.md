
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


# Social Media vs Productivity - ETL Data Preprocessing Pipeline

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
├─ aggregation.py
├─ binarization.py
├─ column_names.py
├─ data_quality.py
├─ data_sampling.py
├─ data_type_definition.py
├─ dependency_map.py
├─ discretization.py
├─ duplicates.py
├─ extract.py
├─ feature_reduction_enhanced.py
├─ features.py
├─ load.py
├─ main.py
├─ missingValues.py
├─ protected_cols.py
├─ selection.py
├─ transform.py
```

---

## Dataset

**Source (raw):** `data/Social_media_and_productivity.csv`  
**Processed (output):** `data/processed_data.csv`

### What these files are
- **`social_media_vs_productivity.csv`** - the **original** dataset exactly as provided.
- **`processed_data.csv`** - the **transformed** dataset produced by our ETL pipeline.

---

### Results

- After running the ETL pipeline, the raw dataset on social media usage and productivity was successfully transformed into a clean and structured dataset.
- The processed data now meets high standards of data quality for further analytics.

---

### Outcomes


- Removed duplicate entries and invalid records. Missing values were imputed based on logical and statistical relationships among features.  
- Ensured consistent data types, verified expected value ranges and fixed outliers.  
- Created new indicators to quantify user behavior and focus.  
- Converted categorical variables into numerical form using label encoding. Continuous variables were discretized into interpretable intervals.  
- Reduced redundant or highly correlated features, keeping only the most informative variables.  
- Aggregated behavior metrics by user groups and balanced the dataset for analysis.

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
