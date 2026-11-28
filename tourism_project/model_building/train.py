
# Data manipulation
import os
import numpy as np
import pandas as pd
import joblib
import mlflow
import xgboost as xgb

# Preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline, Pipeline

# Model training and evaluation
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# MLflow configuration (the workflow starts a local MLflow server on port 5000 before this script)
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-training-experiment")

# 1) Load the train/test splits produced by the data-prep job (downloaded as a workflow artifact)
Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest = pd.read_csv("ytest.csv").squeeze()

# 2) Feature definitions (CustomerID was removed in prep.py)
numeric_features = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "PreferredPropertyStar", "OwnCar", "NumberOfChildrenVisiting", "MonthlyIncome"
]
categorical_features = [
    "TypeofContact", "Occupation", "Gender",
    "MaritalStatus", "ProductPitched", "Designation"
]

# Validate that the required columns exist
required_features = numeric_features + categorical_features
for name, frame in [("training", Xtrain), ("testing", Xtest)]:
    missing_columns = set(required_features) - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing columns in {name} data: {missing_columns}")

# Ensure numeric columns are numeric
for column in numeric_features:
    Xtrain[column] = pd.to_numeric(Xtrain[column], errors="raise")
    Xtest[column] = pd.to_numeric(Xtest[column], errors="raise")

# 3) Handle class imbalance (negatives / positives)
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# 4) Preprocessing + model in ONE pipeline
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)

xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    eval_metric="logloss"
)

param_grid = {
    "xgbclassifier__n_estimators": [50, 100],
    "xgbclassifier__max_depth": [2, 3],
    "xgbclassifier__learning_rate": [0.05, 0.1]
}

model_pipeline = make_pipeline(preprocessor, xgb_model)

# 5) Tune, evaluate, log and save
with mlflow.start_run(run_name="xgboost_gridsearch"):

    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="recall",      # missing a likely buyer is the costly mistake
        n_jobs=-1,
        refit=True             # refit the best parameters on the full training set
    )
    grid_search.fit(Xtrain, ytrain)

    # Log EVERY tuned parameter combination (one nested run each) with its CV score
    results = grid_search.cv_results_
    for params, mean_score, std_score in zip(
        results["params"], results["mean_test_score"], results["std_test_score"]
    ):
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("mean_cv_recall", mean_score)
            mlflow.log_metric("std_cv_recall", std_score)

    # Best fitted pipeline
    best_model = grid_search.best_estimator_
    print("Best parameters:", grid_search.best_params_)

    # Predictions and evaluation
    classification_threshold = 0.5
    y_pred_train = (best_model.predict_proba(Xtrain)[:, 1] >= classification_threshold).astype(int)
    y_pred_test = (best_model.predict_proba(Xtest)[:, 1] >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True, zero_division=0)
    test_report = classification_report(ytest, y_pred_test, output_dict=True, zero_division=0)

    print("\nTest Classification Report:")
    print(classification_report(ytest, y_pred_test, zero_division=0))

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

    # 6) Save the complete FITTED pipeline so the workflow can commit it to the repository
    model_path = "tourism_project/deployment/tourism_package_v1_0.joblib"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    assert isinstance(best_model, Pipeline), f"Expected Pipeline but got {type(best_model)}"
    best_model.named_steps["xgbclassifier"].get_booster()      # raises NotFittedError if unfitted

    joblib.dump(best_model, model_path)

    # 7) Reload and verify, exactly like the Streamlit app will do
    saved_model = joblib.load(model_path)
    assert isinstance(saved_model, Pipeline)
    saved_model.named_steps["xgbclassifier"].get_booster()
    assert np.allclose(
        saved_model.predict_proba(Xtest)[:, 1], best_model.predict_proba(Xtest)[:, 1]
    ), "Reloaded model gives different predictions"

    print("Saved model type:", type(saved_model).__name__)
    print("Pipeline steps:", list(saved_model.named_steps))
    print("SUCCESS: fitted pipeline saved and verified at", model_path)
