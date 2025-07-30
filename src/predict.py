from src.utils import (
    load_model,
    load_datasets,
    train_and_split,
    evaluate_model,
)
from src.config import train_model_path

def predict(X_test, y_test):
    """
    Predict the target values using the loaded model and evaluate its performance.
    """
    y_pred = model.predict(X_test)
    print("Sample Outputs")
    for i in range(5):
        print(f"Predicted: {y_pred[i]:.4f}, Actual: {y_test[i]:.4f}")
    metrics = evaluate_model(y_test, y_pred)
    return metrics

if __name__ == "__main__":
    model = load_model(train_model_path)
    data = load_datasets()
    _, X_test, _, y_test = train_and_split(data)
    metrics = predict(X_test, y_test)
    print(f"Evaluation Metrics: {metrics}")