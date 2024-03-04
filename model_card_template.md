# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- This model is created as part of the Udacity Nano Degree MlOps Engineer
- Created on 04/03/2024
- Version 1.0
- I used a Random Gradient Boosting Classifier from scikit-learn in this project

## Intended Use
- This project is intended to show an example of deploying a ML Pipeline in Production using FastAPI and Render.
- The predition is binary: I predict if the income is above or below 50k based on demographic data.

## Training Data
- I use the census data set as a training data set. 
- Extraction was done by Barry Becker from the 1994 Census database.
- I use cross-validation for train-test split, with stratification, which should ensure the same proportion of the classes as the whole data set.
- I use cv=5, which mean that I always use 80% of data for training and 20% of data for testing.

## Evaluation Data
- I have used 20% of the dataset for evaluation purposes, using k-fold stratified cross-validation (foldes=5).

## Metrics
- I have used the folowing evaluation metrics: fbeta_score, precision_score and recall_score.
- The model's overall performance is:
    - precision:0.8150971599402093
    - recall:0.695447009310037
    - fbeta:0.7505333425091184

## Ethical Considerations
- Validation on slices of the data shows that people with lower education (finished 9th grade or below) are under represented in the data set, and have therfore a low recall score. The preciosn score is close or equal to 1, which mean that the model is reliable when/if the category is detected. 

## Caveats and Recommendations
- The performance of the model can be improved using either more advanced classifiers.
- I could also use over-sampling to balance some under represented categories in the adta set. That would improve the test scores for those categories.  