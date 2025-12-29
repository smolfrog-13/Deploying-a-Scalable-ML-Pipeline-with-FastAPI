import pytest
import numpy as np
from ml.model import compute_model_metrics, train_model, inference
from sklearn.ensemble import RandomForestClassifier

def test_compute_model_metrics():
    """
    Test compute model metrics returns correct values
    """
    y_true = np.array([1,0,1,0])
    y_pred = np.array([1,0,1,0])

    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    assert recall == 1.0
    assert precision == 1.0
    assert fbeta == 1.0


def test_train_model():
    """
    Test trained model is a RandomForestClassifier instance
    """
    X_train = np.array([[0,1],[1,0],[0,0],[1,1]])
    y_train = np.array([0,1,0,1])

    model = train_model(X_train, y_train)

    assert isinstance(model, RandomForestClassifier), "Expected RandomForestClassifier"
 

def test_inference():
    """
    Test inference returns valid predicted values
    """

    X_train = np.array([[0,0],[1,1]])
    y_train = np.array([0,1])
    X_test = np.array([[0,1],[1,0]])

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train) 
    
    preds = inference(model, X_test)

    assert set(preds).issubset({1,0}), "Expected values are 1 and 0"
