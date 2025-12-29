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
    """
    Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label.
    categorical_features : list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in X. If None, y will be returned as None.
    training : bool
        Indicator if training mode or inference mode.
    encoder : OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X_processed : np.array
        Processed feature data.
    y : np.array or None
        Processed labels if label is provided, otherwise None.
    encoder : OneHotEncoder
        Trained encoder.
    lb : LabelBinarizer
        Trained label binarizer.
    """

    if categorical_features is None:
        categorical_features = []

    X = X.copy()

    # Split label if provided
    if label is not None:
        y = X.pop(label).values
    else:
        y = None

    # Categorical features
    X_categorical = X[categorical_features].values
    X_continuous = X.drop(columns=categorical_features).values

    if training:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        X_categorical = encoder.fit_transform(X_categorical)

        if y is not None:
            lb = LabelBinarizer()
            y = lb.fit_transform(y).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)

        if y is not None:
            y = lb.transform(y).ravel()

    X_processed = np.concatenate([X_categorical, X_continuous], axis=1)

    return X_processed, y, encoder, lb
