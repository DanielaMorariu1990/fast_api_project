from ml.model import train_model, compute_model_metrics, inference
from ml.data import process_data


# Score model
def get_general_model_scores(model, X_test, y_test, filename, cv):
    y_pred = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
    with open(f"../model/{filename}", "a") as file:
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
    filename="results_categorical_feature.txt",
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
                cv=(feature, value),
            )
