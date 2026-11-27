from src.utils.config import load_config
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib

config = load_config()

def train_RandomForestRegressor(train_set, train_label):
    """
    Train Random Forest Regressor model on preprocesed dataset

    args:
        train_set: Preprocessed dataset for training on model.
        train_label: Training label data for model training.
    returns:
        model: Trained model on data.
    """

    forest_regressor = RandomForestRegressor()

    forest_regressor.fit(train_set, train_label)

    return forest_regressor

def train_GradientBoostingRegressor(train_set, train_label):
    """
    Train GradientBoosting Regressor model on preprocessed datsaet.

    args:
        train_set: Training set for model trainig.
        train_label: Training label for model training.
    returns:
        model: Trained model on data
    """

    gradientboost_reg = GradientBoostingRegressor()
    gradientboost_reg.fit(train_set, train_label)

    return gradientboost_reg

def save_model(model, file_path=config["paths"]["model_path"]):
    """
    Save model for deployment

    args:
        model: Trained model on preprocessed data
        file_path: File path need to save the model in the place.
    return:
        save the model in that located place.
    """

    joblib.dump(model, file_path)