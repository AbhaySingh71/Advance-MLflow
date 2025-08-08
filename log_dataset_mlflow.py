import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub
import pandas as pd
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

# Set experiment
mlflow.set_experiment('log_dataset_mlflow')

# Parameters
max_depth = 10

with mlflow.start_run():
    # Train model
    dt = DecisionTreeClassifier(max_depth=max_depth)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Log params and metrics
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_metric('accuracy', accuracy)

    # Plot confusion matrix
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix_decision_tree.png")

    # Log artifacts
    mlflow.log_artifact("confusion_matrix_decision_tree.png")
    mlflow.log_artifact(__file__)  # Log current script

    # Merge train and test into DataFrames
    train_df = pd.DataFrame(X_train, columns=iris.feature_names)
    train_df['variety'] = y_train

    test_df = pd.DataFrame(X_test, columns=iris.feature_names)
    test_df['variety'] = y_test

    # Convert to MLflow dataset type
    train_dataset = mlflow.data.from_pandas(train_df)
    test_dataset = mlflow.data.from_pandas(test_df)

    # Log datasets
    mlflow.log_input(train_dataset, "train_data")
    mlflow.log_input(test_dataset, "validation_data")

    # Tags
    mlflow.set_tag('author', 'Abhay Singh')
    mlflow.set_tag('model', 'decision tree with dataset logging')

    print('accuracy:', accuracy)
