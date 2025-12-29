# ml/data.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer

def process_data(
    X,
    categorical_features=[],
    label=None,
    training=True,
    encoder=None,
    lb=None,
):
    """
    Processes the data used in the machine learning pipeline.
    Encodes categorical features and the label.

    Inputs
    ------
    X : pd.DataFrame
        Data to process
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in X. If None, return only X_processed
    training : bool
        Whether this is training mode (fit encoder/label binarizer)
    encoder : sklearn.preprocessing.OneHotEncoder
        Optional, pre-fitted encoder
    lb : sklearn.preprocessing.LabelBinarizer
        Optional, pre-fitted label binarizer

    Returns
    -------
    X_processed : np.array
        Processed feature data
    y : np.array
        Processed labels (None if label=None)
    encoder : OneHotEncoder
        Fitted OneHotEncoder
    lb : LabelBinarizer
        Fitted LabelBinarizer
    """
    X_processed = X.copy()
    
    if label:
        y = X_processed[label].values
        X_processed = X_processed.drop([label], axis=1)
    else:
        y = np.array([])

    if training:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        X_cat = encoder.fit_transform(X_processed[categorical_features])
        lb = LabelBinarizer()
        y = lb.fit_transform(y).ravel() if label else y
    else:
        X_cat = encoder.transform(X_processed[categorical_features])
        y = lb.transform(y).ravel() if label else y

    X_processed = np.hstack([X_processed.drop(columns=categorical_features).values, X_cat])

    return X_processed, y, encoder, lb
