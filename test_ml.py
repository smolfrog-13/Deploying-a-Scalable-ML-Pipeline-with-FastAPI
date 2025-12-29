import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, compute_model_metrics, inference
from ml.data import process_data 
# TODO: add necessary import


#Create sample data for testing
@pytest.fixture
def sample_data():
    data={
        "age" : [25, 38, 28, 44],
        "workclass": ["Private", "Self-emp-not-inc", "Local-gov", "Private"],
        "education": ["Bachelors", "HS-grad", "Assoc-voc", "Some-college"],
        "marital-status": ["Never-married", "Married-civ-spouse", "Divorced", "Separated"], 
        "occupation": ["Tech-support", "Craft-repair", "Exec-managerial", "Sales"], 
        "relationship": ["Not-in-family", "Husband", "Unmarried", "Own-child"],
        "race":["White", "Black", "Asian-Pac-Islander", "White"],
        "sex": ["Male", "Female", "Female", "Male"],
        "native-country":["United-States", "United-States", "India", "United-States"],
        "salary":[">50K", "<=50K", ">50K", "<=50K"]

    }
    return pd.DataFrame(data)

# TODO: implement the first test. Change the function name and input as needed
def test_compute_model_metrics():
    """
    # Test if compute model metrics returns metric values
    """
    # Your code here
    y_true = np.array([1, 0, 1, 0])
    y_preds = np.array([1, 0, 1, 1])
    precision, recall, fbeta = compute_model_metrics(y_true, y_preds)
    assert np.isclose(precision, 0.6667, atol=0.01), "The precision is not meeting expectations"
    assert np.isclose(recall, 1.0, atol=0.01), "The recall is not meeting expectations"
    assert np.isclose(fbeta, 0.8, atol=0.01), "The F1 score is not meeting expectations"


# TODO: implement the second test. Change the function name and input as needed
def test_inference(sample_data):
    """
    # Test if inference returns predictions of the correct length.
    """
    # Your code here
    cat_features = [
        "workclass", "education", "marital-status", "occupation", 
        "relationship", "race", "sex", "native-country"
    ]
    X, y, encoder, lb = process_data(
        sample_data, 
        categorical_features=cat_features,
        label="salary",
        training=True
    )
    model = train_model(X, y)
    preds = inference(model, X)
    assert len(preds) == len(y)



# TODO: implement the third test. Change the function name and input as needed
def test_train_model(sample_data):
    """
    # Test if the train_model trains a model
    """
    # Your code here
    cat_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]
    X, y, _, _ = process_data(
        sample_data,
        categorical_features=cat_features,
        label="salary",
        training=True
    )
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier), "model does not match expectations"
