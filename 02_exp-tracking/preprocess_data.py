#!/usr/bin/env python3
"""
Pre-process raw NYC Taxi parquet files into train/val/test pickle artefacts
and a DictVectorizer for feature engineering.

Example
-------
python preprocess_data.py \
    --raw_data_path ./data/raw \
    --dest_path ./output
"""

import os
import pickle
from pathlib import Path
import click
import pandas as pd
from sklearn.feature_extraction import DictVectorizer


# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #
def dump_pickle(obj, filename: os.PathLike) -> None:
    """Pickle *obj* to *filename* (overwrites if file exists)."""
    with open(filename, "wb") as f_out:
        pickle.dump(obj, f_out)


def read_dataframe(filename: os.PathLike) -> pd.DataFrame:
    """
    Load a parquet file and filter trips outside the 1–60 min duration window.

    Returns
    -------
    DataFrame
        With an added 'duration' column (minutes, float) and
        PULocationID/DOLocationID cast to *str* for downstream hashing.
    """
    df = pd.read_parquet(filename)

    # Duration in minutes
    df["duration"] = (
        df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    # Keep only “reasonable” trips
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    # Ensure categorical columns are strings
    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)

    return df


def preprocess(
    df: pd.DataFrame, dv: DictVectorizer, *, fit_dv: bool = False
):
    """
    Create PU_DO combos and vectorise.  Returns (X_sparse, dv).
    """
    df["PU_DO"] = df["PULocationID"] + "_" + df["DOLocationID"]

    categorical = ["PU_DO"]
    numerical = ["trip_distance"]

    dicts = df[categorical + numerical].to_dict(orient="records")
    X = dv.fit_transform(dicts) if fit_dv else dv.transform(dicts)
    return X, dv


# --------------------------------------------------------------------------- #
# CLI entry-point
# --------------------------------------------------------------------------- #
@click.command()
@click.option(
    "--raw_data_path",
    required=True,
    help="Folder with raw parquet files (one month per file).",
)
@click.option(
    "--dest_path",
    required=True,
    help="Folder where processed pickle files are written.",
)
def run_data_prep(raw_data_path: str, dest_path: str, dataset: str = "green"):
    """
    Expect parquet files called '<dataset>_tripdata_2023-0{1,2,3}.parquet'.
    Saves dv.pkl, train.pkl, val.pkl, test.pkl into *dest_path*.
    """
    raw_path = Path(raw_data_path)
    dest_path = Path(dest_path)
    dest_path.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------------- #
    # Load three months of data
    # ----------------------------------------------------------------------- #
    df_train = read_dataframe(raw_path / f"{dataset}_tripdata_2023-01.parquet")
    df_val = read_dataframe(raw_path / f"{dataset}_tripdata_2023-02.parquet")
    df_test = read_dataframe(raw_path / f"{dataset}_tripdata_2023-03.parquet")

    # ----------------------------------------------------------------------- #
    # Split target / features
    # ----------------------------------------------------------------------- #
    target = "duration"
    y_train, y_val, y_test = (
        df_train[target].values,
        df_val[target].values,
        df_test[target].values,
    )

    # DictVectorizer → sparse feature matrices
    dv = DictVectorizer()
    X_train, dv = preprocess(df_train, dv, fit_dv=True)
    X_val, _ = preprocess(df_val, dv)
    X_test, _ = preprocess(df_test, dv)

    # ----------------------------------------------------------------------- #
    # Persist artefacts
    # ----------------------------------------------------------------------- #
    dump_pickle(dv, dest_path / "dv.pkl")
    dump_pickle((X_train, y_train), dest_path / "train.pkl")
    dump_pickle((X_val, y_val), dest_path / "val.pkl")
    dump_pickle((X_test, y_test), dest_path / "test.pkl")


if __name__ == "__main__":
    run_data_prep()
