import mlflow
import mlflow.pyfunc
import numpy as np
import os

# === 1. Set your DagsHub credentials ===
# Replace with your actual token
os.environ["MLFLOW_TRACKING_USERNAME"] = "AbhaySingh71"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "your-token-here"  # Optional, if you have a token

# === 2. Point MLflow to your DagsHub MLflow tracking URI ===
mlflow.set_tracking_uri("https://dagshub.com/AbhaySingh71/Advance-MLflow.mlflow")

# === 3. Model info ===
model_name = "diabetes-rf"
model_alias = "champion"  # Using the alias instead of version

# === 4. Load model ===
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}@{model_alias}")

# === 5. Prepare input data ===
data = np.array([1, 85, 66, 29, 0, 26.6, 0.351, 31]).reshape(1, -1)

# === 6. Run inference ===
pred = model.predict(data)

print("✅ Prediction:", pred)
