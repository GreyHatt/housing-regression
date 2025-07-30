import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from sklearn.linear_model import LinearRegression
from src.train import train_model
from src.utils import load_datasets

def test_load_datasets():
    data = load_datasets()
    assert 'data' in data
    assert 'target' in data
    assert len(data.data) > 0
    assert len(data.target) > 0

def test_model_creation():
    data = load_datasets()
    model, _ = train_model(data)
    assert isinstance(model, LinearRegression)

def test_model_is_trained():
    data = load_datasets()
    model, _ = train_model(data)
    assert hasattr(model, 'coef_')
    assert model.coef_ is not None

def test_model_r2():
    data = load_datasets()
    _, metrics = train_model(data)
    r2_score = metrics["r2_score"]
    assert r2_score > 0.5