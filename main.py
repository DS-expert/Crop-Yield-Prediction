from src.data.preprocessing import load_dataset, convert_dtypes, remove_outliers, split_dataset, validate_set, target_encoder, preprocessor, save_preprocessed_set
from src.utils.config import load_config
from src.model.model import train_RandomForestRegressor, save_model
from src.model.evaluation import evaluate_model
def main():

    config = load_config()

    df = load_dataset(config["paths"]["raw_data"])

    df_ = convert_dtypes(df, columns=["Area"], dtype=int)

    df_without_outliers = remove_outliers(df_)

    X_train, X_test, y_train, y_test = split_dataset(df_without_outliers, )

    X_train_, X_test_ = target_encoder(X_train, X_test, train_label=y_train, features=["Crop"])

    cat_features = X_train_.select_dtypes(exclude=["number"]).columns.tolist()
    num_features = X_train_.select_dtypes(include=["number"]).columns.tolist()

    num_features.remove("Crop_encoded")

    X_train_encoded, X_test_encoded = preprocessor(X_train_, X_test_, num_features=num_features, cat_features=cat_features)

    X_train_encoded, X_val, y_train, y_val = validate_set(X_train_encoded, y_train)

    save_preprocessed_set(X_train_encoded, y_train, X_test_encoded, y_test, val_set=X_val, val_label=y_val)

    model = train_RandomForestRegressor(X_train_encoded, y_train)
    results = evaluate_model(model, X_test_encoded, y_test)
    print(f"Model : {model}")
    for name, result in results.items():
        print(f"{name} : {result}")
    save_model(model)

if __name__  == "__main__":
    main()


