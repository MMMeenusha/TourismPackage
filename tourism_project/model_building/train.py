
# Data manipulation
import os
import pandas as pd
import joblib
import mlflow
import xgboost as xgb

# Preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# Model training and evaluation
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# MLflow configuration
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-training-experiment")

# Load training and testing data
Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest = pd.read_csv("ytest.csv").squeeze()

# Feature definitions
numeric_features = [
    "CustomerID", "Age", "CityTier", "DurationOfPitch",
    "NumberOfPersonVisiting", "NumberOfFollowups", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore", "PreferredPropertyStar",
    "OwnCar", "NumberOfChildrenVisiting", "MonthlyIncome"
]

categorical_features = [
    "TypeofContact", "Occupation", "Gender",
    "MaritalStatus", "ProductPitched", "Designation"
]

# Validate that the required columns exist
required_features = numeric_features + categorical_features

missing_columns = set(required_features) - set(Xtrain.columns)
if missing_columns:
    raise ValueError(f"Missing columns in training data: {missing_columns}")

missing_test_columns = set(required_features) - set(Xtest.columns)
if missing_test_columns:
    raise ValueError(f"Missing columns in testing data: {missing_test_columns}")

# Ensure numeric columns are numeric
for column in numeric_features:
    Xtrain[column] = pd.to_numeric(Xtrain[column], errors="raise")
    Xtest[column] = pd.to_numeric(Xtest[column], errors="raise")

# Handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Preprocessing pipeline
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)

# XGBoost model
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    eval_metric="logloss"
)

# Hyperparameter grid
param_grid = {
    "xgbclassifier__n_estimators": [50, 100],
    "xgbclassifier__max_depth": [2, 3],
    "xgbclassifier__learning_rate": [0.05, 0.1]
}

# Complete preprocessing + model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Train and evaluate
with mlflow.start_run():

    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="recall",
        n_jobs=-1,
        refit=True
    )

    grid_search.fit(Xtrain, ytrain)

    # Get the best fitted pipeline
    best_model = grid_search.best_estimator_

    print("Best parameters:", grid_search.best_params_)
    print("Best model steps:", best_model.named_steps.keys())

    # Confirm that the XGBoost estimator is fitted
    best_model.named_steps["xgbclassifier"].get_booster()

    # Predictions
    classification_threshold = 0.5

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    # Evaluation
    train_report = classification_report(
        ytrain, y_pred_train, output_dict=True, zero_division=0
    )

    test_report = classification_report(
        ytest, y_pred_test, output_dict=True, zero_division=0
    )

    print("\nTest Classification Report:")
    print(classification_report(ytest, y_pred_test, zero_division=0))

    # MLflow logging
    mlflow.log_params(grid_search.best_params_)

    mlflow.log_metrics({
        "train_accuracy": train_report["accuracy"],
        "train_precision": train_report["1"]["precision"],
        "train_recall": train_report["1"]["recall"],
        "train_f1_score": train_report["1"]["f1-score"],
        "test_accuracy": test_report["accuracy"],
        "test_precision": test_report["1"]["precision"],
        "test_recall": test_report["1"]["recall"],
        "test_f1_score": test_report["1"]["f1-score"]
    })

    # Save the complete fitted pipeline
    model_path = "tourism_project/deployment/tourism_package_v1_0.joblib"

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    joblib.dump(best_model, model_path)

    # Verify the saved model can be loaded and predicts
    loaded_model = joblib.load(model_path)
    loaded_model.predict(Xtest.head(5))

    print("\nModel saved and verified successfully!")
    print("Model type:", type(loaded_model))
    print("Model steps:", loaded_model.named_steps.keys())
    print("Model path:", model_path)

    # Log model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
