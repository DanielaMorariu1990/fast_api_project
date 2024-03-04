# Deploy ML Model using FastApi

Link to project: https://github.com/DanielaMorariu1990/fast_api_project
Link to Render: https://fast-api-project-vzj8.onrender.com

I built a Gradient Boosting Classifier to predict the Income level of the population from the data set census (https://archive.ics.uci.edu/dataset/20/census+income).
The prediction is binary and should only indicate if the salary is above or below 50k. 

I created an app using FastAPI. This app contains a get and a post method. The get method greets the user and the post method, predicts the income level, based on demographic data that the user needs to "post". 

This app was then deployed on Render. 

I included 2 test files, which are part of the github action CI pipeline. One test file called test_train.py tests the training of the model, anothet test file test_main.py test the API creation.

I also included the follwoing test files, with model results. These can be found in the folder "model":
   - results_categorical_features.txt: contain results on data slices (user can pass any categorical features it chooses to investogate, currently it contains the feature "race" and "education")
   - text_scores_from_training.txt: contains the test scores on the training set, on all 5 validation data splits (I used 5 fold stratifed cross-validation)
   - test_file.txt: contains overal model results

The folder screenshots contains the requested deployment screen shots.

I also included a file called "live_post_request.py" which should make some prediction on the render deployed webiste. However, as Render doesn't allow this using the free tier anymore (since 2022), I coudn't test it. 

Howver, I did test the deployed webiste using the Swagger Interface, as one can see in the print screens. 