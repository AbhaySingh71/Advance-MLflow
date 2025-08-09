from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
import dagshub
import pandas as pd
from sklearn.model_selection import train_test_split

#load dataset
df = pd.read_csv('https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv')

dagshub.init(repo_owner='AbhaySingh71', repo_name='Advance-MLflow', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/AbhaySingh71/Advance-MLflow.mlflow")

mlflow.set_experiment('random_forest_hp')

#split data intp features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']   

# splitting the dataset into training and testing sets
X_train , X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# creating the random forest classifier
rf = RandomForestClassifier(random_state=42)

# defining the paramerter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200,300],
    'max_depth': [None, 10, 20, 30]
}

#applying grid search for hyperparameter tuning
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, verbose=2, n_jobs=-1)



with mlflow.start_run(description="Random Forest Hyperparameter Tuning") as parent:
    # fitting the model
    grid_search.fit(X_train, y_train)

    # log all the child runs
    for i in range (len(grid_search.cv_results_['params'])):

        print(i)
        with mlflow.start_run(nested=True) as child:

            mlflow.log_params(grid_search.cv_results_['params'][i])
            mlflow.log_metric("accuracy", grid_search.cv_results_['mean_test_score'][i])

    # Display the best parameters and score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # params
    mlflow.log_params(best_params)

    # metrics
    mlflow.log_metric("best_accuracy", best_score)

    # Train data input logging
    train_df = X_train.copy()
    train_df['Outcome'] = y_train

    train_df = mlflow.data.from_pandas(train_df)

    mlflow.log_input(train_df, "Training Data")
 
    # Test data input logging
    test_df = X_test.copy()
    test_df['Outcome'] = y_test

    test_df = mlflow.data.from_pandas(test_df)

    mlflow.log_input(test_df, "validation Data")    

    # log the source code of the script
    mlflow.log_artifact(__file__)

    #model 
    # Infer model signature
    print(X_train.head())
    signature = mlflow.models.infer_signature(X_train, grid_search.best_estimator_.predict(X_train))

    mlflow.sklearn.log_model(grid_search.best_estimator_ , "random_forest", signature=signature)

    #tags
    mlflow.set_tag('author', 'Abhay Singh')

    print(best_params)
    print(best_score)




