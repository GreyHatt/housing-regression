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


def save_model(model, path) -> None:
    """
    Save the trained model to a file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to {path}")
