
import pandas as pd
import numpy as np
from feature_engine.outliers import ArbitraryOutlierCapper

def load_dataset(file_path: str) -> pd.DataFrame:
    """
    load csv file into dataframe.

    args:
        file_path: Path of csv file.
    returns:
        df(pd.DataFrame): Dataframe of csv file.
    """

    df = pd.read_csv(file_path)

    return df

def convert_dtypes(df: pd.DataFrame, columns: list, dtype) -> pd.DataFrame:
    """
    Convert the datatype of features into desire dtypes.

    args:
        df(pd.DataFrame): Dataframe of dataset.
        feature(str): Feature of dataframe.
        dtype: Desired datatype of feature to convert.
    returns:
        df(pd.DataFrame): Dataframe with new datatype of that feature.
    """

    for feature in columns:
        df[feature] = df[feature].astype(dtype)

    return df

def handle_missing_value(df):

    if df.isnull().sum().sum() == 0:
        return df

    missing_features = df.columns[df.isna().any()].tolist()

    for feature in missing_features:
        df[f"{feature}_missing"] = df[feature].isna().astype(int)
        if df[feature].dtypes in [int, float]:
            df[feature] = df[feature].fillna(df[feature].mean())
        else:
            df[feature] = df[feature].fillna("missing")

    return df

def IQR_boundries(df, feature):
    """
    This function is to find the lower boundries and upper boundries of features in dataframe.

    args:
        df(pd.DataFrame): Dataframe in which those feature are in.
        feature(str): Feature name on which you want to find boundires.
    return:
        lower_boundries, upper_boundrie : Value of boundries of given feature.
    """

    q1 = df[feature].quantile(0.25)
    q3 = df[feature].quantile(0.75)

    IQR = q3 - q1

    lower_boundry = q1 - 1.5 * IQR
    upper_boundry = q3 + 1.5 * IQR

    return lower_boundry, upper_boundry

def remove_outliers(df):
    lower_boundry_dict = {}
    upper_boundry_dict = {}
    num_features = df.select_dtypes(include="number").columns.tolist()

    for feature in num_features:
        lower_boundry_, upper_boundry_ = IQR_boundries(df, feature)
        lower_boundry_dict[feature] = lower_boundry_
        upper_boundry_dict[feature] = upper_boundry_

    capper = ArbitraryOutlierCapper(
        min_capping_dict=lower_boundry_dict,
        max_capping_dict=upper_boundry_dict
    )

    df_capped = capper.fit_transform(df)

    return df_capped