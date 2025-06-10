import argparse
import os
from pathlib import Path
import pickle
from typing import Tuple, List

import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

NUMERICAL: List[str] = []
CATEGORICAL: List[str] = ["PULocationID", "DOLocationID"]
TARGET = "duration"


# Data
def read_dataframe(path: Path) -> pd.DataFrame:
    """Reads parquet into a DataFrame."""
    return pd.read_parquet(path)


def add_duration(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[TARGET] = (df.tpep_dropoff_datetime - df.tpep_pickup_datetime).dt.total_seconds() / 60
    return df[(df[TARGET] >= 1) & (df[TARGET] <= 60)]


def prep_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    df[CATEGORICAL] = df[CATEGORICAL].astype(str)
    y = df[TARGET].values
    X = df[CATEGORICAL].to_dict(orient="records")
    return X, y


# Training
def train_model(
    X_train: List[dict], y_train, X_val: List[dict], y_val
) -> Tuple[LinearRegression, DictVectorizer, float]:
    dv = DictVectorizer()
    X_train_enc = dv.fit_transform(X_train)
    X_val_enc = dv.transform(X_val)

    model = LinearRegression()
    model.fit(X_train_enc, y_train)

    y_pred = model.predict(X_val_enc)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    return model, dv, rmse


# MLFlow run
def run_experiment(
    data_path: Path,
    experiment_name: str = "nyc-taxi-experiment",
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    mlflow.set_tracking_uri(f"file://{data_path / 'mlruns'}")
    mlflow.set_experiment(experiment_name)
    mlflow.autolog(log_models=False)  # wir loggen das Modell von Hand, s. u.

    df = add_duration(read_dataframe(data_path))
    X, y = prep_features(df)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    with mlflow.start_run(run_name="linear-regression-dictvec") as run:
        model, dv, rmse = train_model(X_train, y_train, X_val, y_val)

        # ----- explizite Artefakte
        mlflow.log_metric("rmse", rmse)
        mlflow.sklearn.log_model(model, "model")

        dv_path = Path("dv.pkl")
        with dv_path.open("wb") as f:
            pickle.dump(dv, f)
        mlflow.log_artifact(str(dv_path), artifact_path="dv")

        mlflow.set_tags({"developer": os.getenv("USER", "unknown"), "framework": "sklearn"})


# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NYC taxi duration model.")
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Pfad zur Parquet-Datei (z. B. ../data/yellow_tripdata_2023-03.parquet)",
    )
    parser.add_argument("--experiment-name", default="nyc-taxi-experiment")
    args = parser.parse_args()

    run_experiment(data_path=args.data_path, experiment_name=args.experiment_name)
