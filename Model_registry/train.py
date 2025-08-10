from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
import dagshub
import pandas as pd

# Load dataset
df = pd.read_csv('https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv')

# DagsHub + MLflow tracking setup
dagshub.init(repo_owner='AbhaySingh71', repo_name='Advance-MLflow', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/AbhaySingh71/Advance-MLflow.mlflow")
mlflow.set_experiment('random_forest_hp')

# Features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']   

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest
rf = RandomForestClassifier(random_state=42)

# Hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20]
}

# Grid Search
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, verbose=2, n_jobs=-1)

# MLflow run
with mlflow.start_run(description="Random Forest Hyperparameter Tuning") as run:
    # Fit grid search
    grid_search.fit(X_train, y_train)

    # Best parameters & score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # Log only the best params and metrics
    mlflow.log_params(best_params)
    mlflow.log_metric("best_accuracy", best_score)

    # Train data logging
    train_df = X_train.copy()
    train_df['Outcome'] = y_train
    train_input = mlflow.data.from_pandas(train_df)
    mlflow.log_input(train_input, "Training Data")

    # Test data logging
    test_df = X_test.copy()
    test_df['Outcome'] = y_test
    test_input = mlflow.data.from_pandas(test_df)
    mlflow.log_input(test_input, "Validation Data")    

    # Log the source code file (this script)
    mlflow.log_artifact(__file__)

    # Model signature
    signature = mlflow.models.infer_signature(X_train, grid_search.best_estimator_.predict(X_train))

    # Log the best model
    mlflow.sklearn.log_model(grid_search.best_estimator_, "random_forest", signature=signature)

    # Tags
    mlflow.set_tag('author', 'Abhay Singh')
    mlflow.set_tag('model_type', 'RandomForestClassifier')
    mlflow.set_tag('tuned_with', 'GridSearchCV')

    print("Best Parameters:", best_params)
    print("Best Accuracy:", best_score)
