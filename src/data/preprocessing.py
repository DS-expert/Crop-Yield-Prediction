
import pandas as pd
import numpy as np

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