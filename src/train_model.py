# Script to train machine learning model.

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing._encoders import OneHotEncoder
from sklearn.preprocessing._label import LabelBinarizer
import pandas as pd
import pickle
from ml.data import process_data
from ml.model import (
    train_model,
    get_general_model_scores,
)

import os
import logging


# Add the necessary imports for the starter code.
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

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

# Add code to load in the data.
logger.info("Reading in data...")
data = pd.read_csv(os.path.join(os.getcwd(), "data/census.csv"), skipinitialspace=True)
logger.info("Cleaning data...")
columns_new = [col[0] for col in data.columns.str.split()]
data.columns = columns_new


# Optional enhancement, use K-fold cross validation instead of a train-test split.
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=41)


cv = 1
for train_index, test_index in skf.split(data.drop(columns=["salary"]), data["salary"]):
    logger.info(f"We are in {cv}:")

    X_train, y_train, encoder, lb = process_data(
        data.iloc[train_index],
        categorical_features=cat_features,
        label="salary",
        training=True,
    )

    # Proces the test data with the process_data function.
    X_test, y_test, encoder, lb = process_data(
        data.iloc[test_index],
        categorical_features=cat_features,
        encoder=encoder,
        lb=lb,
        label="salary",
        training=False,
    )
    # Train and save a model.
    model = train_model(X_train, y_train)
    logger.info("Calculating and logging precision, recall, fbeta.")
    precision, recall, fbeta = get_general_model_scores(
        model,
        X_test,
        y_test,
        filename=os.path.join(os.getcwd(), "model/test_scores_from_training.txt"),
        cv=cv,
    )
    cv += 1


logger.info("Save trained model.")
with open(os.path.join(os.getcwd(), "model/trainedmodel.pkl"), "wb") as file:
    pickle.dump(model, file)

logger.info("save fitted encoder and binary encoder.")
with open(os.path.join(os.getcwd(), "model/encoder.pkl"), "wb") as file:
    pickle.dump(encoder, file)

with open(os.path.join(os.getcwd(), "model/lb.pkl"), "wb") as file:
    pickle.dump(lb, file)
