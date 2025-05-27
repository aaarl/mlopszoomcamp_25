#!/usr/bin/env python3
"""
Single-shot training script for a baseline RandomForestRegressor.
Logs everything to MLflow for quick experiment tracking.
"""

import os
import pickle
from pathlib import Path
import click
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# --------------------------------------------------------------------------- #
# MLflow setup
# --------------------------------------------------------------------------- #
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("week2_experiment")
mlflow.sklearn.autolog()


def load_pickle(filename: os.PathLike):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


# --------------------------------------------------------------------------- #
# CLI entry-point
# --------------------------------------------------------------------------- #
@click.command()
@click.option(
    "--data_path",
    default="./output",
    show_default=True,
    help="Folder with preprocess_data output (train/val pickle).",
)
def run_train(data_path: str):
    """
    Train a RandomForest with fixed hyper-params and report validation RMSE.
    Useful as a baseline before HPO.
    """
    data_path = Path(data_path)
    X_train, y_train = load_pickle(data_path / "train.pkl")
    X_val, y_val = load_pickle(data_path / "val.pkl")

    with mlflow.start_run():
        rf = RandomForestRegressor(max_depth=10, random_state=0, n_jobs=-1)
        rf.fit(X_train, y_train)

        val_rmse = mean_squared_error(y_val, rf.predict(X_val), squared=False)
        print(f"Validation RMSE: {val_rmse:.3f}")
        mlflow.log_metric("val_rmse", val_rmse)


if __name__ == "__main__":
    run_train()
