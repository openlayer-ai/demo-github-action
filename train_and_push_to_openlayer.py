"""Simple script to train a churn classifier model and push it to Openlayer."""

import copy
import os
import sys

import openlayer
import pandas as pd
import yaml
from openlayer.tasks import TaskType
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder

OPENLAYER_API_KEY = os.environ["OPENLAYER_API_KEY"]
COMMIT_MSG = os.environ.get("GITHUB_COMMIT_MESSAGE", "Commit from Openlayer Python SDK")
PROJECT_NAME = "Churn Prediction"

# ---------------------------- Preparing the data ---------------------------- #

train_df = pd.read_csv("./churn_train.csv")
val_df = pd.read_csv("./churn_val.csv")

feature_names = [
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "AggregateRate",
    "Year",
]
label_column_name = "Exited"

x_train = train_df[feature_names]
y_train = train_df[label_column_name]

x_val = val_df[feature_names]
y_val = val_df[label_column_name]


def data_encode_one_hot(df, encoders):
    """Encodes categorical features using one-hot encoding."""
    df = df.copy(True)
    df.reset_index(drop=True, inplace=True)  # Causes NaNs otherwise
    for feature, enc in encoders.items():
        enc_df = pd.DataFrame(
            enc.transform(df[[feature]]).toarray(),
            columns=enc.get_feature_names([feature]),
        )
        df = df.join(enc_df)
        df = df.drop(columns=feature)
    return df


def create_encoder_dict(df, categorical_feature_names):
    """Creates encoders for each of the categorical features.
    The predict function will need these encoders.
    """

    encoders = {}
    for feature in categorical_feature_names:
        enc = OneHotEncoder(handle_unknown="ignore")
        enc.fit(df[[feature]])
        encoders[feature] = enc
    return encoders


encoders = create_encoder_dict(x_train, ["Geography", "Gender"])

x_train_one_hot = data_encode_one_hot(x_train, encoders)
x_val_one_hot = data_encode_one_hot(x_val, encoders)

# Imputation with the training set's mean to replace NaNs
x_train_one_hot_imputed = x_train_one_hot.fillna(
    x_train_one_hot.mean(numeric_only=True)
)
x_val_one_hot_imputed = x_val_one_hot.fillna(x_train_one_hot.mean(numeric_only=True))

# ---------------------------- Training the model ---------------------------- #

sklearn_model = GradientBoostingClassifier(random_state=1300)
sklearn_model.fit(x_train_one_hot_imputed, y_train)

print(classification_report(y_val, sklearn_model.predict(x_val_one_hot_imputed)))

# --------------------------- Pushing to Openlayer --------------------------- #

client = openlayer.OpenlayerClient(OPENLAYER_API_KEY)

project = client.create_or_load_project(
    name="Churn Prediction",
    task_type=TaskType.TabularClassification,
    description="Evaluation of ML approaches to predict churn",
)


# Adding the column with the labels
training_set = x_train.copy(deep=True)
training_set["Exited"] = y_train.values
validation_set = x_val.copy(deep=True)
validation_set["Exited"] = y_val.values

# Adding the column with the predictions (since we'll also upload a model later)
training_set["predictions"] = sklearn_model.predict_proba(
    x_train_one_hot_imputed
).tolist()
validation_set["predictions"] = sklearn_model.predict_proba(
    x_val_one_hot_imputed
).tolist()

# Some variables that will go into the `dataset_config.yaml` file
categorical_feature_names = ["Gender", "Geography"]
class_names = ["Retained", "Exited"]
column_names = list(training_set.columns)
feature_names = list(x_val.columns)
label_column_name = "Exited"
prediction_scores_column_name = "predictions"


# Note the camelCase for the dict's keys
training_dataset_config = {
    "categoricalFeatureNames": categorical_feature_names,
    "classNames": class_names,
    "columnNames": column_names,
    "featureNames": feature_names,
    "label": "training",
    "labelColumnName": label_column_name,
    "predictionScoresColumnName": prediction_scores_column_name,
}

with open("training_dataset_config.yaml", "w", encoding="utf-8") as dataset_config_file:
    yaml.dump(training_dataset_config, dataset_config_file, default_flow_style=False)

validation_dataset_config = copy.deepcopy(training_dataset_config)

# The only field that changes is the `label`, from "training" -> "validation"
validation_dataset_config["label"] = "validation"

with open(
    "validation_dataset_config.yaml", "w", encoding="utf-8"
) as dataset_config_file:
    yaml.dump(validation_dataset_config, dataset_config_file, default_flow_style=False)


# Training set
project.add_dataframe(
    dataset_df=training_set,
    dataset_config_file_path="training_dataset_config.yaml",
)

# Validation set
project.add_dataframe(
    dataset_df=validation_set,
    dataset_config_file_path="validation_dataset_config.yaml",
)

print(project.status())


model_config = {
    "name": PROJECT_NAME,
    "architectureType": "sklearn",
    "metadata": {  # Can add anything here, as long as it is a dict
        "model_type": "Gradient Boosting Classifier",
        "regularization": "None",
        "encoder_used": "One Hot",
        "imputation": "Imputed with the training set's mean",
    },
    "classNames": class_names,
    "featureNames": feature_names,
    "categoricalFeatureNames": categorical_feature_names,
}

with open("model_config.yaml", "w", encoding="utf-8") as model_config_file:
    yaml.dump(model_config, model_config_file, default_flow_style=False)

project.add_model(
    model_config_file_path="model_config.yaml",
)

project.commit(COMMIT_MSG)

print(project.status())

version = project.push()

version.wait_for_completion(timeout=600)
version.print_goal_report()

if version.failing_goal_count > 0:
    print("Failing pipeline due to failing goals.")
    sys.exit(1)
