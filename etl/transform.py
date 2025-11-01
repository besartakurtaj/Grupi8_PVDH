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
    df = apply_discretization(df, column="daily_social_media_time", n_bins=4, strategy="uniform")

    #Column names to uppercase 
    df = titlecase_columns(df)
    return df 