
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

Load raw CSV data

Enforce consistent data types

Detect and remove duplicates

Assess data quality and logical inconsistencies

Perform controlled sampling

Prepare the dataset for downstream transformations

### Files and Responsibilities
#### extract.py

Main entry point for Phase 1.

Reads the raw CSV file

Applies data type definitions

Assesses data quality and logical constraints

Reports missing values, duplicates, and descriptive statistics

Removes duplicate rows

Performs stratified sampling (default: 50%)

Returns a cleaned pandas DataFrame

This script orchestrates most Phase 1 components and produces the dataset used in later phases.

#### data_type_definition.py

Data type enforcement layer.

Defines a centralized TYPE_MAPPING for all expected columns

Converts columns to appropriate pandas dtypes (Int64, float64, category)

Logs warnings when conversions fail

Prints final column data types for verification

Ensures schema consistency across the pipeline.

#### duplicates.py

Duplicate handling utility.

Removes duplicate rows from the dataset

Supports optional column subsets and retention strategy

Reports how many rows were removed

Used during extraction to ensure data uniqueness.

#### data_quality.py

Logical data validation module.

Performs rule-based checks (e.g. age range, work hours, stress score bounds)

Detects logically invalid or extreme values

Produces a structured report of detected issues

This step does not remove data but provides transparency and diagnostics.

#### data_sampling.py

Sampling strategy implementation.

Supports:

Simple random sampling

Stratified sampling (by job_type when available)

Configurable sampling fraction and random seed

Used in Phase 1 to reduce dataset size while preserving representativeness.

#### column_names.py

Column formatting utility.

Converts snake_case column names into Title Case with spaces

Can be run as a standalone script for CSV transformation

Optional helper for presentation and reporting purposes.

Additional Phase 1 Components (ETL Completion)

Phase 1 also includes the full ETL (Extract–Transform–Load) flow, covering advanced cleaning, feature creation, dimensionality reduction, and dataset persistence.

#### main.py

End-to-end pipeline entry point.

Defines input and output file paths

Executes the full ETL flow:

extract_data() – raw data ingestion and validation

transform_data() – cleaning, feature engineering, and reduction

load_data() – saving the final dataset

This script is the recommended way to run Phase 1 from start to finish.

#### transform.py

Core transformation engine for Phase 1.

Performs dependency-aware missing value imputation

Applies binarization and categorical encoding

Adds aggregated and derived features

Applies discretization to numeric variables

Reduces dimensionality using correlation and duplication analysis

Formats column names for readability

This module consolidates all preprocessing logic.

#### missingValues.py

Advanced missing value imputation.

Uses a dependency map to impute missing values contextually

Searches for similar rows based on related features

Falls back to global medians when no matches are found

Preserves integer semantics where required

Provides smarter imputation than simple mean/median strategies.

#### features.py

Feature engineering module.

Creates interaction and ratio-based features such as:

Stress–sleep ratio

Insomnia pressure

Distraction load

Work-to-social ratio

Burnout rate

Handles division-by-zero safely

Enhances analytical expressiveness of the dataset.

feature_reduction_enhanced.py

Explainable dimensionality reduction.

Removes exact duplicate columns

Removes highly correlated features

Preserves protected columns

Logs the reason for every feature removal

Performs final row deduplication

Ensures a compact, non-redundant feature space.

#### protected_cols.py

Feature protection configuration.

Defines columns that must never be dropped during reduction

Used by the enhanced feature reduction step

Safeguards critical engineered features.

#### load.py

Data loading and persistence layer.

Writes the final processed DataFrame to CSV

Ensures consistent formatting and no index column

Marks the completion of Phase 1.

Dataset: social_media_vs_productivity.csv

Raw input dataset.

Serves as the source data for Phase 1

Contains demographic, behavioral, productivity, and wellbeing variables

#### Phase 1 Output

The output of Phase 1 is:

A fully processed and cleaned dataset

Advanced feature engineering applied

Missing values intelligently imputed

Redundant features removed with explanations

Columns formatted for analysis

Dataset saved as a CSV file ready for modeling or analysis

