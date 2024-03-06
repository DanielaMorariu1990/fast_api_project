import requests
import json

data = {
    "age": 50,
    "workclass": " Self-emp-not-inc",
    "fnlgt": 83311,
    "education": " Bachelors",
    "education-num": 13,
    "marital-status": " Married-civ-spouse",
    "occupation": " Exec-managerial",
    "relationship": " Husband",
    "race": " White",
    "sex": " Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 13,
    "native-country": " United-States",
}
response = requests.post(
    "https://fast-api-project-vzj8.onrender.com/predict/", data=json.dumps(data)
)

print(response.status_code)
print(response.json())
