# Put the code for your API here.
from fastapi import FastAPI, Depends
from pydantic import BaseModel, Field, ConfigDict
from src.ml.data import process_data
from src.ml.model import inference
from typing import Dict, List, Optional
import pandas as pd
import pickle
import json


class DataSource(BaseModel):

    age: int = Field(alias="age")
    workclass: str = Field(alias="workclass")
    fnlgt: int = Field(alias="fnlgt")
    education: str = Field(alias="education")
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str = Field(alias="occupation")
    relationship: str = Field(alias="relationship")
    race: str = Field(alias="race")
    sex: str = Field(alias="sex")
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")

    class Config:
        allow_population_by_field_name = True

        schema_extra = {
            "example": {
                "age": 58,
                "workclass": " Private",
                "fnlgt": 151910,
                "education": " HS-grad",
                "education-num": 9,
                "marital-status": " Widowed",
                "occupation": " Adm-clerical",
                "relationship": " Unmarried",
                "race": " White",
                "sex": " Female",
                "capital-gain": 0,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": " United-States",
            }
        }


app = FastAPI()


@app.get("/")
async def say_hello():
    return {"Hello new user! This app predicts income based on demographic features."}


# Use POST action to send data to the server
@app.post("/predict")
async def exercise_function(data: DataSource):
    # Reading the input data
    age = data.age
    workclass = data.workclass
    fnlgt = data.fnlgt
    education = data.education
    education_num = data.education_num
    marital_status = data.marital_status
    occupation = data.occupation
    relationship = data.relationship
    race = data.race
    sex = data.sex
    capital_gain = data.capital_gain
    capital_loss = data.capital_loss
    hours_per_week = data.hours_per_week
    native_country = data.native_country

    raw_data = pd.DataFrame(
        [
            {
                "age": age,
                "workclass": workclass,
                "fnlgt": fnlgt,
                "education": education,
                "education-num": education_num,
                "marital-status": marital_status,
                "occupation": occupation,
                "relationship": relationship,
                "race": race,
                "sex": sex,
                "capital-gain": capital_gain,
                "capital-loss": capital_loss,
                "hours-per-week": hours_per_week,
                "native-country": native_country,
            }
        ]
    )

    try:
        with open("./model/trainedmodel.pkl", "rb") as model_file:
            model = pickle.load(model_file)
    except:
        "No trained model found."

    try:
        with open("./model/encoder.pkl", "rb") as encoder_file:
            encoder = pickle.load(encoder_file)
    except:
        "No fitted encoder found."

    try:
        with open("./model/lb.pkl", "rb") as lb_file:
            lb = pickle.load(lb_file)
    except:
        "No fitted binary label found."

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X_test, y_test, encoder, lb = process_data(
        raw_data,
        categorical_features=cat_features,
        encoder=encoder,
        lb=lb,
        label=None,
        training=False,
    )
    pred = inference(model, X_test)
    response_object = {"prediction": json.dumps(pred.tolist())}
    return response_object
