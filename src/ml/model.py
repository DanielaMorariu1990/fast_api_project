from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import GradientBoostingClassifier
from ml.data import process_data
import os


# Optional: implement hyperparameter tuning.
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
    model
        Trained machine learning model.
    """
    gbcl = GradientBoostingClassifier(
        loss="exponential",
        learning_rate=0.3,
        n_estimators=200,
        subsample=0.7,
        max_depth=5,
        max_features="sqrt",
        random_state=41,
    )

    gbcl.fit(X_train, y_train)

    return gbcl


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
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
    """Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    pred = model.predict(X)
    return pred


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
