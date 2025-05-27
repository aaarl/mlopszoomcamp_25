#!/usr/bin/env python3
"""
Pull the best hyper-opt RF configs, re-train on full data, evaluate,
and register the champion model in MLflow.
"""

import os
import pickle
from pathlib import Path
import click
import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# --------------------------------------------------------------------------- #
# Global config
# --------------------------------------------------------------------------- #
HPO_EXPERIMENT_NAME = "random-forest-hyperopt"
EXPERIMENT_NAME = "random-forest-best-models"
RF_PARAMS = [
    "max_depth",
    "n_estimators",
    "min_samples_split",
    "min_samples_leaf",
    "random_state",
]

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.sklearn.autolog()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def load_pickle(filename: os.PathLike):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def train_and_log_model(data_path: Path, params: dict):
    """
    Re-train a RandomForestRegressor with *params* and log to MLflow.  
    Validation & test RMSE are logged for model selection.
    """
    X_train, y_train = load_pickle(data_path / "train.pkl")
    X_val, y_val = load_pickle(data_path / "val.pkl")
    X_test, y_test = load_pickle(data_path / "test.pkl")

    with mlflow.start_run(nested=True):
        # Cast hyper-opt strings → ints
        clean_params = {k: int(params[k]) for k in RF_PARAMS}
        # Ensure full CPU usage unless provided
        clean_params.setdefault("n_jobs", -1)

        rf = RandomForestRegressor(**clean_params)
        rf.fit(X_train, y_train)

        mlflow.log_metrics(
            {
                "val_rmse": mean_squared_error(
                    y_val, rf.predict(X_val), squared=False
                ),
                "test_rmse": mean_squared_error(
                    y_test, rf.predict(X_test), squared=False
                ),
            }
        )
        # The fitted model artefact is auto-logged by mlflow.sklearn.autolog()


# --------------------------------------------------------------------------- #
# CLI entry-point
# --------------------------------------------------------------------------- #
@click.command()
@click.option(
    "--data_path",
    default="./output",
    show_default=True,
    help="Folder with preprocess_data output (train/val/test pickle).",
)
@click.option(
    "--top_n",
    default=5,
    show_default=True,
    type=int,
    help="Number of top hyper-opt runs to re-evaluate.",
)
def run_register_model(data_path: str, top_n: int):
    """
    Select *top_n* hyper-opt runs with best RMSE, retrain on full data,
    then register the model with the lowest *test_rmse*.
    """
    data_path = Path(data_path)
    client = MlflowClient()

    # ----------------------------------------------------------------------- #
    # 1) Pull the best hyper-opt runs
    # ----------------------------------------------------------------------- #
    hpo_exp = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    best_hpo_runs = client.search_runs(
        hpo_exp.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.rmse ASC"],
    )

    # Re-train each candidate on train data & evaluate
    for run in best_hpo_runs:
        train_and_log_model(data_path, run.data.params)

    # ----------------------------------------------------------------------- #
    # 2) Pick the champion model (lowest test_rmse) and register it
    # ----------------------------------------------------------------------- #
    best_exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    champion = client.search_runs(
        [best_exp.experiment_id],
        run_view_type=ViewType.ACTIVE_ONLY,
        order_by=["metrics.test_rmse ASC"],
        max_results=1,
    )[0]

    model_uri = f"runs:/{champion.info.run_id}/model"
    mlflow.register_model(model_uri, name="best-random-forest-model")


if __name__ == "__main__":
    run_register_model()
