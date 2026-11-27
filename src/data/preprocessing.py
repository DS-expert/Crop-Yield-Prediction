
import pandas as pd
import numpy as np
from feature_engine.outliers import ArbitraryOutlierCapper
from src.utils.config import load_config
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

config = load_config()

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

def split_dataset(df, target=config["data"]["target_feature"]):
    X = df.drop(columns=["target"])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=config["data"]["test_size"], 
                                                        random_state=config["data"]["random_state"])
    return X_train, X_test, y_train, y_test

def validate_set(train_set, train_label):
    """
    In this function, we split the train set into validate set

    args:
        train_set: training input feature of dataset.
        train_label: training label of dataset
    
    return:
        X_train, X_val, y_train, y_val : training and Validation set of feature and labels
    """

    X_train, X_val, y_train, y_val = train_test_split(train_set, train_label, 
                                                      test_size=config["data"]["test_size"],
                                                      random_state=config["data"]["random_state"])
    return X_train, X_val, y_train, y_val

def target_encoder(train_set: pd.DataFrame, test_set:pd.DataFrame, train_label: pd.DataFrame, features: list, smoothing=10, min_samples_leaf=20):
    """
    In this function, it convert the categorical feature with average value of target labels.

    args:
        train_set(pd.DataFrame): Dataframe of training features.
        test_set(pd.DataFrame): Dataframe of test features.
        train_label(pd.Series): Series of training labels.
        feature: Given feature on which to perform target_encoder.
        smoothing(int): Prevent overfitting on high cardinality or rare categoircal features.
        min_samples_lead(int): Prevent overfitting on low frequency or rate categorical features.
    returns:
        train_set, test_set: train and test set with encoded of given features.
    """

    encoder = TargetEncoder(
        cols=features,
        smoothing=smoothing,
        min_samples_leaf=min_samples_leaf
    )

    encoded_features = []
    for feature in features:
        encoded_features.append(f"{feature}_encoded")

    train_set[encoded_features] = encoder.fit_transform(train_set[features], train_label)
    test_set[encoded_features] = encoder.transform(test_set[features])

    train_set_ = train_set.drop(columns=features)
    test_set_ = test_set.drop(columns=features)

    return train_set_, test_set_

def preprocessor(train_set, test_set, num_features: list, cat_features: list):
    """
    In this preprocessor, It will apply scaling and encoding on features.
    args:
        train_set: Train dataset to apply preprocesser
        test_set: test dataset for transform the preprocesser
    returns:
        train_set, test_set: Return preprocessed datasets
    """

    transformer = ColumnTransformer(
        transformers=[
            ("Scaling", StandardScaler(), num_features),
            ("Encoding", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features)
        ], remainder="passthrough"
    )

    train_set_preprocessed = transformer.fit_transform(train_set)
    test_set_preprocessed = transformer.transform(test_set)

    return train_set_preprocessed, test_set_preprocessed

def save_preprocessed_set(train_set, train_label, test_set, test_label, val_set=None, val_label=None):
    if val_set and val_label:
        joblib.dump(val_set, config["paths"]["X_val_path"])
        joblib.dump(val_label, config["paths"]["y_val_path"])

    joblib.dump(train_label, config["paths"]["y_train_path"])
    joblib.dump(train_set, config["paths"]["X_train_path"])
    joblib.dump(test_set, config["paths"]["X_test_path"])
    joblib.dump(test_label, config["paths"]["y_test_path"])