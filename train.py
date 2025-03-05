import argparse
import mlflow

parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, required=True)
parser.add_argument("--l1_ratio", type=float, required=True)
args = parser.parse_args()

mlflow.start_run()
mlflow.log_param("alpha", args.alpha)
mlflow.log_param("l1_ratio", args.l1_ratio)
mlflow.end_run()
