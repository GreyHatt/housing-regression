import os
import pandas as pd
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Tuple

def load_datasets():
    """
    Load the California housing dataset.
    """
    data = fetch_california_housing()
    return data

def train_and_split(data) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset into training and testing sets.
    """
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test

def evaluate_model(y_true, y_pred) -> dict:
    """
    Evaluate the model's performance metrics.
    """
    return {
        "mean_absolute_error": mean_absolute_error(y_true, y_pred),
        "mean_squared_error": mean_squared_error(y_true, y_pred),
        "r2_score": r2_score(y_true, y_pred),
    }

def save_object(object, path) -> None:
    """
    Save the trained model to a file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(object, path)
    print(f"Model saved to {path}")

def load_model(path):
    """
    Load a trained model from a file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")
    model = joblib.load(path)
    print(f"Model loaded from {path}")
    return model

def save_params(method, coef, intercept, params_path):
    """
    Save the unquantized parameters to a file.
    """
    if "unquantize" in method:
        params = {"coef": coef, "intercept": intercept}
    elif "quantize" in method:
        params = {
            "coef_q": coef["q"],
            "coef_scale": coef["scale"],
            "coef_min": coef["min"],
            "intercept_q": intercept["q"],
            "intercept_scale": intercept["scale"],
            "intercept_min": intercept["min"],
        }
    save_object(params, params_path)
    print(f"{method} parameters saved to {params_path}")