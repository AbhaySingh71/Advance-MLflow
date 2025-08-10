import mlflow
import dagshub

model_name = "diabetes-rf"
version_to_promote = 3  # keep as int for readability

# Connect to DagsHub
dagshub.init(repo_owner='AbhaySingh71', repo_name='Advance-MLflow', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/AbhaySingh71/Advance-MLflow.mlflow")

# Initialize MLflow client
client = mlflow.MlflowClient()

# Assign alias (casting version to string to avoid TypeError)
client.set_registered_model_alias(model_name, "champion", str(version_to_promote))

print(f"✅ Alias 'champion' has been assigned to version {version_to_promote} of '{model_name}'.")
