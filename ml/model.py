import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer


def process_data(
    X,
    categorical_features=None,
    label=None,
    training=True,
    encoder=None,
    lb=None,
):
    if categorical_features is None:
        categorical_features = []

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(columns=categorical_features + ([label] if label else []))

    if training:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        X_cat = encoder.fit_transform(X_categorical)
    else:
        X_cat = encoder.transform(X_categorical)

    X_processed = np.concatenate([X_cat, X_continuous.values], axis=1)

    if label is not None:
        y = X[label].values
        if training:
            lb = LabelBinarizer()
            y = lb.fit_transform(y).ravel()
        else:
            y = lb.transform(y).ravel()
    else:
        y = np.array([])

    return X_processed, y, encoder, lb
