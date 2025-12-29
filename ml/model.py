import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score

# Make sure this import is at the top-level
from ml.data import process_data


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.

    Returns
    -------
    model : RandomForestClassifier
        Trained machine learning model.
    """
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        True labels.
    preds : np.array
        Predicted labels.

    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """
    Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.

    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)


def save_model(model, path):
    """Serializes model to a file."""
    try:
        with open(path, 'wb') as f:
            pickle.dump(model, f)
    except Exception as e:
        print(f"Error saving model: {e}")


def load_model(path):
    """Loads pickle file from `path` and returns it."""
    try:
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")


def performance_on_categorical_slice(
    data, column_name, slice_value, categorical_features, label, encoder, lb, model
):
    """
    Computes model metrics for a slice of the data defined by `column_name==slice_value`.

    Inputs
    ------
    data : pd.DataFrame
    column_name : str
    slice_value : str, int, float
    categorical_features : list
    label : str
    encoder : OneHotEncoder
    lb : LabelBinarizer
    model : RandomForestClassifier

    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    input_data = data[data[column_name] == slice_value]
    if input_data.empty:
        return 0.0, 0.0, 0.0

    # Call process_data from the top-level import
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
