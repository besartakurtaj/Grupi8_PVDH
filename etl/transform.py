"""
End-to-end data transformation to prepare the dataset for analysis.
"""

import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import OneHotEncoder
from dependency_map import dependency_map
from missingValues import advanced_imputation
from binarization import apply_binarization
from aggregation import add_aggregated
from features import create_features
from discretization import apply_discretization
from column_names import titlecase_columns
from selection import select_feature_subset
from feature_reduction_enhanced import reduce_dimensions_enhanced
from protected_cols import protected_cols

def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    
    #Missing Value Imputation
    df = advanced_imputation(df, dependency_map)
    
    # Binarization
    df = apply_binarization(df)

    # Aggregation
    df = add_aggregated(df)

    #Krijimi i vetive
    df = create_features(df)

    # #Selektimi i vetive
    # df = select_features(df)

    #Discretization
    df = apply_discretization(df, column="age", n_bins=5, strategy="uniform")

    #Dimension Reduction
    df = reduce_dimensions_enhanced(df, protected_cols=protected_cols, corr_threshold=0.98)

    #Column names to uppercase 
    df = titlecase_columns(df)

    #Selektimi i vetive
    df = select_feature_subset(
        df,
        keep_always=["Gender", "Age", "Job Type","Daily Social Media Time", "Social Platform Preference"],
        corr_threshold=0.95,
    )
    return df 