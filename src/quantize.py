import numpy as np
from src.utils import (
    load_model, 
    save_params,
    load_datasets,
    train_and_split,
    evaluate_model,
)
from src.config import (
    train_model_path,
    unquant_params_path,
    quant_params_path,
)

def extract_parameters(model):
    """
    Extract coef and intercept from the model.
    """
    coef = np.array(model.coef_)
    intercept = np.array(model.intercept_)
    return coef, intercept

def quantize(weights, n_bits=16):
    """
    Quantize the model parameters to a uint8.
    """
    levels = 2 ** n_bits -1
    min_val = np.min(weights)
    max_val = np.max(weights)
    scale = (max_val - min_val) / levels if max_val > min_val else 1.0
    quantized = np.round((weights - min_val) / scale).astype(np.uint16)
    return {
        "q": quantized,
        "scale": scale,
        "min": min_val,
    }

def dequantize(weights):
    """
    Dequantize the model parameters from a uint8.
    """
    return weights["q"].astype(np.float32) * weights["scale"] + weights["min"]

def inference(coef, intercept):
    """
    Perform inference using the dequantized model parameters.
    """
    data = load_datasets()
    X_train, X_test, y_train, y_test = train_and_split(data)
    print(f"shapes: {coef.shape} , intercept {intercept.shape}")
    y_pred = np.dot(X_test, coef) + intercept
    metrics = evaluate_model(y_test, y_pred)
    return metrics

def run(coef, intercept):
    """
    Run the quantization process on the model parameters.
    """
    coef_q = quantize(coef)
    intercept_q = quantize(intercept)
    save_params("quantize", coef_q, intercept_q, quant_params_path)
    coef_dq = dequantize(coef_q)
    intercept_dq = dequantize(intercept_q)
    return inference(coef_dq, intercept_dq)

if __name__ == "__main__":
    model = load_model(train_model_path)
    coef, intercept = extract_parameters(model)
    save_params("unquantize", coef, intercept, unquant_params_path)
    metrics = run(coef, intercept)
    print("Quantization metrics:", metrics)
