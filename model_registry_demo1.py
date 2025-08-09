from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn
import dagshub

# Initialize DagsHub repo for MLflow tracking
dagshub.init(repo_owner='AbhaySingh71', repo_name='Advance-MLflow', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/AbhaySingh71/Advance-MLflow.mlflow")

# Set experiment
mlflow.set_experiment('mlflow_model_registry')

with mlflow.start_run() as run:
    # Create dataset
    X, y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    params = {"max_depth": 2, "random_state": 42}
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)

    # Log parameters and metrics
    mlflow.log_params(params)
    y_pred = model.predict(X_test)
    mlflow.log_metrics({"mse": mean_squared_error(y_test, y_pred)})

    # Log & register model (old API style)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="sklearn-model",  # must use this for DagsHub
        input_example=X_train,
        registered_model_name="sk-learn-random-forest-reg-model"
    )

# Verify model is registered
client = mlflow.MlflowClient()
latest_versions = client.get_latest_versions("sk-learn-random-forest-reg-model", stages=["None"])
for v in latest_versions:
    print(f"Model Name: {v.name}, Version: {v.version}, Run ID: {v.run_id}")
