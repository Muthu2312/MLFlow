import argparse
import mlflow
import mlflow.sklearn
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error

parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, required=True)
parser.add_argument("--l1_ratio", type=float, required=True)
args = parser.parse_args()

mlflow.start_run()

mlflow.log_param("alpha", args.alpha)
mlflow.log_param("l1_ratio", args.l1_ratio)

model = ElasticNet(alpha=args.alpha, l1_ratio=args.l1_ratio)
y_true = [1, 2, 3]
y_pred = [1, 2.1, 3.1]
mse = mean_squared_error(y_true, y_pred)

mlflow.log_metric("mse", mse)

mlflow.sklearn.log_model(model, "model")

mlflow.end_run()
