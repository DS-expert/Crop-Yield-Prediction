from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, root_mean_squared_error

def evaluate_model(model, test_features, test_labels):
    """
    Evaluate the model

    args:
        model: Trained model
        test_set: Test set for prediction of model.
        test_labels: test label for evaluation of model.
    return:
        dict: r2score, mae, mse, rmse
    """
    y_pred = model.predict(test_features)

    r2_score_model = r2_score(test_labels, y_pred)
    mae_model = mean_absolute_error(test_labels, y_pred=y_pred)
    mse_model = mean_squared_error(test_labels, y_pred)
    rmse_model = root_mean_squared_error(test_labels, y_pred=y_pred)

    return {
        "r2 score": r2_score_model,
        "Mean Absolute Error ": mae_model,
        "Mean Squarred Error ": mse_model,
        "Root Mean Squared Error ": rmse_model
    }
