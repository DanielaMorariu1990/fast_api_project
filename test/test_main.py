from main import app
import json
from fastapi.testclient import TestClient
import pytest
from fastapi.encoders import jsonable_encoder

client = TestClient(app)


def test_post_1():
    input_dict = {
        "age": 52,
        "workclass": "Self-emp-inc",
        "fnlgt": 287927,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Wife",
        "race": "White",
        "sex": "Female",
        "capital-gain": 15024,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }
    response = client.post("/predict", json=input_dict)
    assert response.status_code == 200
    print(response.json())
    assert response.json() == {
        "prediction": "[1]"
    }, f"Returned {response.json()} instead of 0."


def test_post_2():
    input_dict = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }
    response = client.post("/predict", json=input_dict)
    assert response.status_code == 200
    print(response.json())
    assert response.json() == {
        "prediction": "[0]"
    }, f"Returned {response.json()} instead of 0."


def test_get():
    r = client.get("/")
    assert r.status_code == 200, f"Status code {r.status_code} returned instead of 200"
    assert r.json() == [
        "Hello new user! This app predicts income based on demographic features."
    ]
