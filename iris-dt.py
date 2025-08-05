import mlflow
import mlflow.sklearn
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


mlflow.set_tracking_uri("http://127.0.0.1:5000")


# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameters for the Random Forest model
max_depth = 5

# set experiment
mlflow.set_experiment('iris_dt')

# apply mlflow

with mlflow.start_run(experiment_id='677906403814493186', run_name='Abhay_boss'):

    dt = DecisionTreeClassifier(max_depth=max_depth)

    dt.fit(X_train, y_train)

    y_pred = dt.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric('accuracy', accuracy)

    mlflow.log_param('max_depth', max_depth)

    cm = confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True,fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confuison Matrix')

    plt.savefig("Confuison Matrix.png")
    
    # log the the confusion matrix
    mlflow.log_artifact("Confuison Matrix.png")

    # log the file
    mlflow.log_artifact(__file__)

    # Enable complete experiment tracking with one line
    mlflow.sklearn.autolog()

    # log the model
    mlflow.sklearn.log_model(dt, name="Decision_Tree_Model")
    from mlflow.models.signature import infer_signature

    signature = infer_signature(X_test, y_pred)
    mlflow.sklearn.log_model(dt, name="Decision_Tree_Model", input_example=X_test[:1], signature=signature)

    mlflow.set_tag('author','Abhay singh')
    mlflow.set_tag('model','Decision Tree')


    print('Accuracy', accuracy)
    