import mlflow
import mlflow.sklearn
from mlflow.sklearn import save_model
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub
import os

# Initialize DagsHub repo for MLflow tracking
dagshub.init(repo_owner='AbhaySingh71', repo_name='Advance-MLflow', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/AbhaySingh71/Advance-MLflow.mlflow")

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Parameters
max_depth = 10
n_estimators = 100

# Apply mlflow

# mlflow autologging feature
mlflow.autolog()

mlflow.set_experiment('mlflow_autolog_feature')

with mlflow.start_run():
    # Train model
    rf = RandomForestClassifier(max_depth=max_depth , n_estimators=n_estimators)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")

    # Log artifacts
    mlflow.log_artifact("confusion_matrix.png")
    #mlflow.log_artifact(__file__)  # Log current script


    # Tags
    mlflow.set_tag('author','Abhay Singh')
    mlflow.set_tag('model','Random Forest Classifier')

    print('accuracy:', accuracy)
