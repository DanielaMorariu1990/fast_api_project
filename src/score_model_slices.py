import os
from ml.model import inference, compute_model_metrics
from ml.data import process_data
import logging
import pickle
import pandas as pd


# Add the necessary imports for the starter code.
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def get_general_model_scores(model, X_test, y_test, filename, cv):
    y_pred = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
    with open(f"{filename}", "a") as file:
        file.write(f"{cv}:\n")
        file.write(f"precision:{precision}\n")
        file.write(f"recall:{recall}\n")
        file.write(f"fbeta:{fbeta}\n")
    return precision, recall, fbeta


def score_model_on_slices(
    model,
    cat_feature_slice,
    cat_features,
    data,
    encoder,
    lb,
    filename=os.path.join(os.getcwd(), "model/results_categorical_feature.txt"),
):
    for feature in cat_feature_slice:
        all_values = list(dict.fromkeys(data[feature].values))
        for value in all_values:
            X_test, y_test, encoder, lb = process_data(
                data[data[feature] == value],
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
                filename=filename,
                cv=f"For feature: {feature} and value:{value}",
            )


# Add code to load in the data.
logger.info("Reading in data...")
data = pd.read_csv(os.path.join(os.getcwd(), "data/census.csv"), skipinitialspace=True)

logger.info("Cleaning data...")
columns_new = [col[0] for col in data.columns.str.split()]
data.columns = columns_new

logger.info("Loading trained model.")
with open(os.path.join(os.getcwd(), "model/trainedmodel.pkl"), "rb") as model_file:
    model = pickle.load(model_file)

logger.info("Loading encoder and binary encoder.")
with open(os.path.join(os.getcwd(), "model/encoder.pkl"), "rb") as encoder_file:
    encoder = pickle.load(encoder_file)

with open(os.path.join(os.getcwd(), "model/lb.pkl"), "rb") as lb_file:
    lb = pickle.load(lb_file)

logger.info("Preparing data")
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

score_model_on_slices(
    model,
    cat_feature_slice=["race", "education"],
    cat_features=cat_features,
    data=data,
    encoder=encoder,
    lb=lb,
    filename=os.path.join(os.getcwd(), "model/results_categorical_feature.txt"),
)
