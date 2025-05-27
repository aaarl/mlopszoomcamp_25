#!/usr/bin/env python3
"""
Hyper-parameter search for a RandomForestRegressor on the NYC-Taxi dataset
using Hyperopt + MLflow.

Run from the shell, e.g.
    python rf_hyperopt.py --data_path ./output --num_trials 20
"""

import os
import pickle
from pathlib import Path  # slightly safer than os.path for paths
import click
import mlflow
import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# --------------------------------------------------------------------------- #
# MLflow configuration – adjust to your MLflow server if needed
# --------------------------------------------------------------------------- #
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("random-forest-hyperopt")


def load_pickle(filename: os.PathLike):
    """
    Convenience wrapper for loading pickled objects.

    Parameters
    ----------
    filename : str | Path
        File path to a pickle file.

    Returns
    -------
    Any
        The deserialised Python object.
    """
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    show_default=True,
    help="Folder that contains train.pkl and val.pkl exported by the data prep step.",
)
@click.option(
    "--num_trials",
    default=15,
    show_default=True,
    help="Number of hyper-parameter configurations to evaluate.",
)
def run_optimization(data_path: str, num_trials: int) -> None:
    """
    Runs a Hyperopt search over RandomForestRegressor parameters and logs
    everything to MLflow.

    Notes
    -----
    * Expects two pickle files, *train.pkl* and *val.pkl*, each containing
      `(X, y)` tuples created during preprocessing.
    * Each trial is logged as a separate MLflow run with the parameter set
      and the resulting RMSE.
    """

    # ----------------------------------------------------------------------- #
    # Load data
    # ----------------------------------------------------------------------- #
    data_path = Path(data_path)
    X_train, y_train = load_pickle(data_path / "train.pkl")
    X_val, y_val = load_pickle(data_path / "val.pkl")

    # ----------------------------------------------------------------------- #
    # Objective function handed to Hyperopt
    # ----------------------------------------------------------------------- #
    def objective(params):
        """
        Train a RandomForestRegressor with **params**, compute validation RMSE
        and log the run to MLflow. Hyperopt minimises this RMSE.
        """
        # Each Hyperopt evaluation gets its own (nested) MLflow run
        with mlflow.start_run(nested=True):
            # Important optimisation parameter: use all CPU cores
            params.setdefault("n_jobs", -1)

            rf = RandomForestRegressor(**params)
            rf.fit(X_train, y_train)

            y_pred = rf.predict(X_val)
            rmse = mean_squared_error(y_val, y_pred, squared=False)

            # ----------- MLflow logging ----------------------------------- #
            mlflow.log_params(params)
            mlflow.log_metric("rmse", rmse)

            # Hyperopt expects a dict with 'loss' - the quantity to minimise
            return {"loss": rmse, "status": STATUS_OK}

    # ----------------------------------------------------------------------- #
    # Search space definition
    # Note: values are cast to int via `scope.int(...)`
    # ----------------------------------------------------------------------- #
    search_space = {
        "max_depth": scope.int(hp.quniform("max_depth", 1, 20, 1)),
        "n_estimators": scope.int(hp.quniform("n_estimators", 10, 150, 1)),
        "min_samples_split": scope.int(hp.quniform("min_samples_split", 2, 10, 1)),
        "min_samples_leaf": scope.int(hp.quniform("min_samples_leaf", 1, 4, 1)),
        "random_state": 42,
    }

    # Ensure reproducibility of the search itself
    rstate = np.random.default_rng(42)

    # ----------------------------------------------------------------------- #
    # Kick off the optimisation
    # ----------------------------------------------------------------------- #
    fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=num_trials,
        trials=Trials(),
        rstate=rstate,
    )


if __name__ == "__main__":
    run_optimization()
