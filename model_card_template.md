# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Model Type: RandomForestClassifier Framework: Scikit-learn

## Intended Use
The purpose of the model is to predict whether a person's income goes over $50k based on factors such as gender, education, and occupation.

## Training Data
The data used was the Census income dataset using categorical and continous features. Training set: 80% of data Testing set: 20% of the data

## Evaluation Data
To perform the evaluation, I used metrics such as precision recall and f1 score.

## Metrics
_Please include the metrics used and your model's performance on those metrics._
Precision: 0.7419 | Recall: 0.6384 | F1: 0.6863

## Ethical Considerations
The model may have bias towards certain groups of people due to dataset imbalances. The data also contains sensitive information which must be handled carefully when applying to other datasets.

## Caveats and Recommendations
It is recommended to further evaluate the model using additional fairness metrics. The model should not be used for critical decisions. Based on the slice_output, there must be optimization to include more groups that are unrepresented.
