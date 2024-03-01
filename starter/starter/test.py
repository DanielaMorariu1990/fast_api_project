import pandas as pd
import pickle
from ml.data import process_data
from ml.model import inference
from score_model import get_general_model_scores
import pytest


@pytest.fixture
def data():
    data = pd.read_csv("../data/census.csv")
    columns_new = [col[0] for col in data.columns.str.split()]
    data.columns = columns_new
    return data


@pytest.fixture
def model():
    with open("../model/trainedmodel.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    return model


@pytest.fixture
def encoder():
    with open("../model/enocder.pkl", "rb") as encoder_file:
        encoder = pickle.load(encoder_file)
    return encoder


@pytest.fixture
def lb():
    with open("../model/lb.pkl", "rb") as lb_file:
        lb = pickle.load(lb_file)
    return lb


@pytest.fixture
def cat_features():
    return [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]


def test_input_shape(data):
    assert data.shape[1] == 15


def test_final_metrics(data, encoder, lb, cat_features, model):
    X_test, y_test, encoder, lb = process_data(
        data,
        categorical_features=cat_features,
        encoder=encoder,
        lb=lb,
        label="salary",
        training=False,
    )
    precision, recall, fbeta = get_general_model_scores(
        model,
        X_test,
        y_test,
        filename="test_file.txt",
        cv=1,
    )
    assert (
        (precision > 0.75) & (recall > 0.6) & (fbeta > 0.6)
    ), f"For precision we have value {precision}, for recall we have a value {recall} and for fbeta we have a value {fbeta}."


def test_final_metrics(data, encoder, lb, cat_features, model):
    X_test, y_test, encoder, lb = process_data(
        data,
        categorical_features=cat_features,
        encoder=encoder,
        lb=lb,
        label="salary",
        training=False,
    )
    precision, recall, fbeta = get_general_model_scores(
        model,
        X_test,
        y_test,
        filename="test_file.txt",
        cv=1,
    )
    assert (
        (precision > 0.75) & (recall > 0.65) & (fbeta > 0.65)
    ), f"For precision we have value {precision}, for recall we have a value {recall} and for fbeta we have a value {fbeta}."


def test_inference(model, data, encoder, lb, cat_features):
    X_test, y_test, encoder, lb = process_data(
        data,
        categorical_features=cat_features,
        encoder=encoder,
        lb=lb,
        label="salary",
        training=False,
    )
    pred = inference(model, X_test)
    assert (
        len(pred) == data.shape[0]
    ), f"Data is shaped {data.shape[0]}, wheras pred is ahped {len(pred)}"
