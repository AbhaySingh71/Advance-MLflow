from mlflow.tracking import MlflowClient
import dagshub
import mlflow

# Connect to DagsHub MLflow
dagshub.init(repo_owner='AbhaySingh71', repo_name='Advance-MLflow', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/AbhaySingh71/Advance-MLflow.mlflow")

client = MlflowClient()

model_name = "diabetes-rf"
new_champion_version = 3  # The new model you want to promote

# --- Step 1: Find current champion (if any) ---
try:
    current_champion = client.get_model_version_by_alias(model_name, "champion")
    old_champion_version = int(current_champion.version)

    if old_champion_version != new_champion_version:
        # Remove 'champion' alias from old champion
        client.delete_registered_model_alias(model_name, "champion")

        # Assign 'archived' alias to old champion
        client.set_registered_model_alias(model_name, "archived", old_champion_version)
        print(f"Old champion (v{old_champion_version}) archived.")

except Exception:
    print("No existing champion found — promoting new model directly.")

# --- Step 2: Promote new model to champion ---
client.set_registered_model_alias(model_name, "champion", new_champion_version)
print(f"Model version {new_champion_version} promoted to 'champion'.")
