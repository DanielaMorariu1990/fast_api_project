import os
from ml.model import score_model_on_slices
import logging
import pickle
import pandas as pd


# Add the necessary imports for the starter code.
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


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
