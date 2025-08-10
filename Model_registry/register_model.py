
# client demo

from mlflow.tracking import MlflowClient
import mlflow
import dagshub

dagshub.init(repo_owner='AbhaySingh71', repo_name='Advance-MLflow', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/AbhaySingh71/Advance-MLflow.mlflow")
# Initialize the MLflow Client
client = MlflowClient()

# Replace with the run_id of the run where the model was logged
run_id = "526785ab6be84602af7a2e3ad399e8d4"

# Replace with the path to the logged model within the run
model_path = "mlflow-artifacts:/33c81dba4317470abc20f43cae190d24/526785ab6be84602af7a2e3ad399e8d4/artifacts/random_forest"

# Construct the model URI
model_uri = f"runs:/{run_id}/{model_path}"

# Register the model in the model registry
model_name = "diabetes-rf"
result = mlflow.register_model(model_uri, model_name)

import time
time.sleep(5)

# Add a description to the registered model version
client.update_model_version(
    name=model_name,
    version=result.version,
    description="This is a RandomForest model trained to predict diabetes outcomes based on Pima Indians Diabetes Dataset."
)

client.set_model_version_tag(
    name=model_name,
    version=result.version,
    key="experiment",
    value="diabetes prediction"
)

client.set_model_version_tag(
    name=model_name,
    version=result.version,
    key="day",
    value="sat"
)
print(f"Model registered with name: {model_name} and version: {result.version}")
print(f"Added tags to model {model_name} version {result.version}")

# Get and print the registered model information
registered_model = client.get_registered_model(model_name)
print("Registered Model Information:")
print(f"Name: {registered_model.name}")
print(f"Creation Timestamp: {registered_model.creation_timestamp}")
print(f"Last Updated Timestamp: {registered_model.last_updated_timestamp}")
print(f"Description: {registered_model.description}")



#gpt code to register model
"""
from mlflow.tracking import MlflowClient
import mlflow
import dagshub
import time

# DagsHub connection
dagshub.init(repo_owner='AbhaySingh71', repo_name='Advance-MLflow', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/AbhaySingh71/Advance-MLflow.mlflow")

client = MlflowClient()

# Replace with the run_id of the best model you logged
run_id = "526785ab6be84602af7a2e3ad399e8d4"

# This should match the artifact path you used when logging the model
artifact_path = "random_forest"

# Correct model URI
model_uri = f"runs:/{run_id}/{artifact_path}"

# Register model
model_name = "diabetes-rf"
result = mlflow.register_model(model_uri, model_name)

# Wait to ensure registration completes
time.sleep(5)

# Add description
client.update_model_version(
    name=model_name,
    version=result.version,
    description="RandomForest model trained on Pima Indians Diabetes Dataset."
)

# Add tags
client.set_model_version_tag(
    name=model_name,
    version=result.version,
    key="experiment",
    value="diabetes prediction"
)
client.set_model_version_tag(
    name=model_name,
    version=result.version,
    key="day",
    value="sat"
)

print(f"Model registered: {model_name}, version: {result.version}")

# Fetch registered model info
registered_model = client.get_registered_model(model_name)
print("Registered Model Information:")
print(f"Name: {registered_model.name}")
print(f"Creation Timestamp: {registered_model.creation_timestamp}")
print(f"Last Updated Timestamp: {registered_model.last_updated_timestamp}")
print(f"Description: {registered_model.description}")

"""

