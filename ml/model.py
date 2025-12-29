# ml/model.py

import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score
import numpy as np

from ml.data import process_data

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def compute_model_metrics(y, preds):
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta

def inference(model, X):
    return model.predict(X)

def save_model(model, path):
    try:
        with open(path, "wb") as f:
            pickle.dump(model, f)
    except Exception as e:
        print(f"Error saving model: {e}")

def load_model(path):
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")

def performance_on_categorical_slice(
    data, column_name, slice_value, categorical_features, label, encoder, lb, model
):
    """
    Computes precision, recall, and F1 for a slice of data.
    """
    input_data = data[data[column_name] == slice_value]
    if input_data.empty:
        return 0, 0, 0
    
    X_slice, y_slice, _, _ = process_data(
        input_data,
        categorical_features=categorical_features,
        label=label,
        encoder=encoder,
        lb=lb,
        training=False
    )

    preds = inference(model, X_slice)
    precision, recall, fbeta = compute_model_metrics(y_slice, preds)
    return precision, recall, fbeta

