import argparse
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, required=True)
parser.add_argument("--l1_ratio", type=float, required=True)
args = parser.parse_args()

# Start MLflow run
with mlflow.start_run():
    mlflow.log_param("alpha", args.alpha)
    mlflow.log_param("l1_ratio", args.l1_ratio)

    # Dummy data (in real case, load your dataset)
    np.random.seed(42)
    X = np.random.rand(100, 5)
    y = np.random.rand(100)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = ElasticNet(alpha=args.alpha, l1_ratio=args.l1_ratio, random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    # Log metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)

    # Log model
    mlflow.sklearn.log_model(model, "model")
