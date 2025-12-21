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
## Phase 1 – Data Extraction, Validation, and Core Preparation

Phase 1 focuses on loading raw data, enforcing data consistency, performing initial validation, and preparing a clean, well-typed dataset ready for further preprocessing and feature engineering.

### Objectives

* Load raw CSV data
* Enforce consistent data types
* Detect and remove duplicates
* Assess data quality and logical inconsistencies
* Perform controlled sampling
* Prepare the dataset for downstream transformations

### Files and Responsibilities
#### extract.py

* Main entry point for Phase 1.
* Reads the raw CSV file
* Applies data type definitions
* Assesses data quality and logical constraints
* Reports missing values, duplicates, and descriptive statistics
* Removes duplicate rows
* Performs stratified sampling (default: 50%)
* Returns a cleaned pandas DataFrame

This script orchestrates most Phase 1 components and produces the dataset used in later phases.

#### data_type_definition.py

* Data type enforcement layer.
* Defines a centralized TYPE_MAPPING for all expected columns
* Converts columns to appropriate pandas dtypes (Int64, float64, category)
* 
<img width="440" height="506" alt="image" src="https://github.com/user-attachments/assets/e0c03619-ace8-4a3d-a2d2-aed0c81ba453" />

* Logs warnings when conversions fail
* Prints final column data types for verification

<img width="582" height="537" alt="image" src="https://github.com/user-attachments/assets/7b96444a-b36b-4b75-87ff-c106c7dea57c" />

* Ensures schema consistency across the pipeline.

#### duplicates.py

* Duplicate handling utility.
* Removes duplicate rows from the dataset

<img width="722" height="93" alt="image" src="https://github.com/user-attachments/assets/a99a7f32-fa81-444a-a0fe-cb5269085ea7" />

* Supports optional column subsets and retention strategy
* Reports how many rows were removed
* Used during extraction to ensure data uniqueness.

#### data_quality.py

* Logical data validation module.
* Performs rule-based checks (e.g. age range, work hours, stress score bounds)
* Detects logically invalid or extreme values
* Produces a structured report of detected issues
* This step does not remove data but provides transparency and diagnostics.

#### data_sampling.py

* Sampling strategy implementation.
Supports:

* Simple random sampling
* Stratified sampling (by job_type when available)
* Configurable sampling fraction and random seed
* Used in Phase 1 to reduce dataset size while preserving representativeness.

#### column_names.py

* Column formatting utility.
* Converts snake_case column names into Title Case with spaces
* Can be run as a standalone script for CSV transformation
* Optional helper for presentation and reporting purposes.
  
#### main.py

* End-to-end pipeline entry point.
* Defines input and output file paths
* Executes the full ETL flow:

  - `extract_data()` – raw data ingestion and validation
  - `transform_data()` – cleaning, feature engineering, and reduction
  - `load_data()` – saving the final dataset

This script is the recommended way to run Phase 1 from start to finish.

#### transform.py

* Core transformation engine for Phase 1.
* Performs dependency-aware missing value imputation
* Applies binarization and categorical encoding
* Adds aggregated and derived features
* Applies discretization to numeric variables
* Reduces dimensionality using correlation and duplication analysis
* Formats column names for readability
* This module consolidates all preprocessing logic.

#### missingValues.py

<img width="537" height="483" alt="image" src="https://github.com/user-attachments/assets/e7c08e82-9b68-4e8f-8983-6aeb7dad83d8" />

* Advanced missing value imputation.
* Uses a dependency map to impute missing values contextually
* Searches for similar rows based on related features
* Falls back to global medians when no matches are found
* Preserves integer semantics where required
* Provides smarter imputation than simple mean/median strategies.

#### features.py

* Feature engineering module.
* Creates interaction and ratio-based features such as:
* Stress–sleep ratio
* Insomnia pressure
* Distraction load
* Work-to-social ratio
* Burnout rate
* Handles division-by-zero safely
* Enhances analytical expressiveness of the dataset.

### feature_reduction_enhanced.py

* Explainable dimensionality reduction.
* Removes exact duplicate columns
* Removes highly correlated features
* Preserves protected columns
* Logs the reason for every feature removal
* Performs final row deduplication
* Ensures a compact, non-redundant feature space.

#### protected_cols.py

* Feature protection configuration.
* Defines columns that must never be dropped during reduction
* Used by the enhanced feature reduction step
* Safeguards critical engineered features.

#### load.py

* Data loading and persistence layer.
* Writes the final processed DataFrame to CSV
* Ensures consistent formatting and no index column
* Marks the completion of Phase 1.

**Dataset: social_media_vs_productivity.csv**

* Raw input dataset.
* Serves as the source data for Phase 1
* Contains demographic, behavioral, productivity, and wellbeing variables

#### Phase 1 Output

The output of Phase 1 is:

* A fully processed and cleaned dataset
* Advanced feature engineering applied
* Missing values intelligently imputed
* Redundant features removed with explanations
* Columns formatted for analysis
* Dataset saved as a CSV file ready for modeling or analysis

## Phase 2 – Exploratory Data Analysis & Outlier Handling

Phase 2 extends the pipeline with in-depth statistical exploration, visual analytics, and robust outlier detection. The goal of this phase is to understand, validate, and stress-test the dataset produced in Phase 1 before any modeling decisions are made.

### Objectives

* Provide a comprehensive statistical and visual overview of the dataset
* Analyze demographic balance and workforce composition
* Identify behavioral and wellbeing risk patterns
* Detect anomalous observations using complementary statistical techniques
* Produce clean, traceable datasets for downstream modeling

### Files and Responsibilities

### outliers_detection.py

* Robust multivariate outlier detection and auditing module.
* This script applies two independent statistical methods to numeric features:
* Interquartile Range (IQR) — robust to skewed distributions
* Z‑Score (|z| > 3) — sensitive to extreme deviations
  
**Outlier Analysis Outputs:**

* Outlier counts per column (IQR method)

<img width="2965" height="1762" alt="outlier_counts_iqr" src="https://github.com/user-attachments/assets/66b546f7-f0ac-48f4-b324-90a70a59d97c" />

* Outlier counts per column (Z‑Score method)

<img width="2965" height="1762" alt="outlier_counts_zscore" src="https://github.com/user-attachments/assets/60c1dbce-7d32-40ac-a9cb-1ad7cab71fa8" />

* Global outlier flag per row (is_outlier)

* Pie‑chart visualization of clean vs anomalous data

<img width="1656" height="1771" alt="outlier_pie_chart" src="https://github.com/user-attachments/assets/da702688-300e-4512-8146-2940bf729327" />

* Outliers are flagged, logged, and preserved for auditability, not silently removed.

**Generated Datasets:**

**dataset_with_outliers_flag.csv**
Full dataset including the is_outlier indicator column.

**removed_outliers_log.csv**
Complete log of all rows identified as anomalous.

**cleaned_dataset.csv**
Final dataset with outliers removed, used for analysis and modeling.


#### exploratory_analysis.py

* Advanced Exploratory Data Analysis (EDA) and visualization engine.
* This script generates both numerical summaries and high‑quality analytical visualizations:
* Statistical Analysis
* Dataset structure and schema inspection
* Summary statistics for numeric and categorical features
* Missing value diagnostics
* Correlation matrix computation
* Identification of strongest variable relationships
* Key Visual Outputs
* Top 10 Strongest Correlations

<img width="4158" height="2662" alt="top_correlations" src="https://github.com/user-attachments/assets/0d12d4d6-dd20-4de2-be80-c19781154397" />

* Highlights the most influential relationships, including:
* Actual vs perceived productivity
* Social media usage vs distraction metrics
* Stress‑sleep interactions
* Risk Factor Prevalence

<img width="3557" height="2362" alt="risk_prevalence" src="https://github.com/user-attachments/assets/a49159c1-e6f7-4a90-805d-13978146d6a4" />

**Quantifies the proportion of the population affected by:**
* Excessive notifications
* Burnout risk
* High stress

<img width="4628" height="2154" alt="stress_impact" src="https://github.com/user-attachments/assets/3ae331c9-2a3b-486a-bc83-20de36a21a1f" />

* Low sleep
* Social media addiction
* A 50% threshold is used to flag critical public‑health‑level risks.
* Stress Impact on Work Performance
* Dual scatter analysis showing:

**Stress vs productivity (colored by job satisfaction)**

* Stress vs job satisfaction (colored by productivity)
* Includes trend lines and correlation coefficients for interpretability.
* Dataset Demographics Overview

**Visual breakdown of:**
* Gender distribution
* Job type distribution
* Job optimism categories

All figures are exported to analysis/output/ for reporting and presentation use.

### Phase 2 Output

**Phase 2 delivers:**

* A statistically validated and visually interpreted dataset
* Clear insights into behavioral, productivity, and wellbeing patterns
* Transparent identification of extreme or anomalous observations
* A clean, modeling‑ready dataset with full traceability

## Phase 3 - Data visualization
This phase focuses on visualizing the dataset based on data types to better understand patterns, distributions, and relationships.
Interactive dashboards were created to allow users to explore categorical, binary, and numerical variables dynamically.

The goal of this phase is to transform raw data into clear, interpretable visual insights that support analysis and decision-making.

Visualize different types of data appropriately

* Enable interactive selection of variables
* Identify distributions, proportions, and imbalances
* Improve data understanding before modeling or analysis

### Interactive Dashboard (Tableau Public)

🔗 **Tableau Public Dashboard:**  
https://public.tableau.com/views/Grupi8-SocialMediavsProductivity-PVDH-Master/Dashboard1?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link

### Categorical Data Visualization

Chart Type: Pie Charts

Purpose: Show the proportion of categories within a selected variable.

Available Categories:

* Gender (F, M, O)
* Job Type (Education, Finance, Health, etc.)
* Job Optimism (Optimistic, Neutral, Pessimistic)

<img width="863" height="643" alt="image" src="https://github.com/user-attachments/assets/c993461b-a9be-486b-b75d-ef03db95e330" />

### Binary Data Visualization

Chart Type: Bar Chart

Purpose: Display the count of binary outcomes (0 / 1).

Available Binary Variables:

* High Stress
* Social Addicted
* Low Sleep
* Too Many Notifications
* Burnout Risk
  
<img width="577" height="497" alt="image" src="https://github.com/user-attachments/assets/6aa11e7b-f5a6-497c-a277-832b168446c4" />

### Numerical Data Visualization

Chart Type: Histogram

Purpose: Analyze the distribution of continuous numerical variables.

Available Numerical Variables:

* Sleep Hours
* Work Hours Per Day
* Daily Social Media Time
* Stress Level
* Actual Productivity Score
* Job Satisfaction Score

<img width="1198" height="572" alt="image" src="https://github.com/user-attachments/assets/9cb5a5b6-067d-44b1-ba96-a220ede8de50" />


## Multidimensional Data Visualizations

This section presents both **interactive** and **static multidimensional visualizations** derived from the dataset containing demographic, behavioral, productivity, stress, sleep, and social media usage indicators.  
The objective is to uncover relationships across multiple dimensions while maintaining interpretability and analytical clarity.
### Interactive Multidimensional Visualizations
#### Impact of Stress Level on Insomnia and Sleep Quality

**Chart Type:** Line Combination  

**Description:**  
This static visualization analyzes the relationship between Stress Level, Insomnia Pressure, and Sleep Hours. Users can filter the data using Gender and Job Type controls.

**Purpose:**  
To highlight the direct impact of stress on sleep-related outcomes in a fixed and descriptive manner.

<img width="857" height="661" alt="image" src="https://github.com/user-attachments/assets/4747ab0b-c0bf-4f5d-a41d-8bca3b59f102" />

#### Productivity and Notification Frequency Across Work Break Levels

**Chart Type:** Line Combination
**Description:**  
This visualization compares Actual Productivity Score and Number of Notifications across different levels of Breaks During Work. Interactive filters allow dynamic comparison without altering the chart structure.

**Purpose:**  
To explore how break frequency influences productivity and digital distraction.

<img width="660" height="558" alt="image" src="https://github.com/user-attachments/assets/68b7cbed-db57-42c6-92d4-9cd25848977b" />

#### Burnout Risk by Stress Level and Sleep Hours

**Chart Type:** Colored Side-by-Side Bar Chart 

**Description:**  
This visualization explores how Burnout Risk varies across different Stress Levels, with Sleep Hours (binned) represented by color. Global filters for Gender and Job Type are applied.

**Purpose:**  
To analyze burnout patterns using multiple dimensions in an interactive environment.

<img width="790" height="516" alt="image" src="https://github.com/user-attachments/assets/7ba5a27b-57b4-4d0b-8635-3dc55557eef5" />

### Static Multidimensional Visualizations
#### Social Media Platform Preferences by Job Type

**Chart Type:** Stacked / Grouped Bar Chart  

**Description:**  
This interactive visualization shows preferences for different social media platforms across various job types. Users can filter the data using Gender and Job Type controls.

**Purpose:**  
To enable exploratory analysis of platform usage patterns across professional groups.

<img width="683" height="510" alt="image" src="https://github.com/user-attachments/assets/830a002e-7997-48ca-a9e8-825960b0b069" />

#### Comparison of Average Age and Job Satisfaction Across Genders

**Chart Type:** Stacked / Grouped Bar Chart  

**Description:**  
This chart compares Average Age and Average Job Satisfaction Score across genders using a dual-axis approach. Interactive filters allow dynamic comparison without altering the chart structure.

**Purpose:**  
To examine demographic differences in job satisfaction while maintaining visual clarity.
<img width="537" height="578" alt="image" src="https://github.com/user-attachments/assets/0b3161c1-ba1f-49d6-b0e7-8d6c6f25436c" />

---

Authors:
        Besarta Kurtaj,
        Enis Hoxha,
        Shefket Bylygbashi
Year: 2025
