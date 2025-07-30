from sklearn.linear_model import LinearRegression
from src.utils import (
    load_datasets,
    train_and_split,
    save_model,
    evaluate_model,
)
from src.config import train_model_path

def train_model(data):
    """
    Train a model using the provided dataset.
    """
    X_train, X_test, y_train, y_test = train_and_split(data)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred)
    return model, metrics

if __name__ == "__main__":
    data = load_datasets()
    model, metrics = train_model(data)
    print(metrics)
    save_model(model, train_model_path)