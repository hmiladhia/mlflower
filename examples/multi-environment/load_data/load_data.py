from __future__ import annotations

import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import typer
from sklearn import datasets
from sklearn.model_selection import train_test_split


def export(output_dir: str | Path, filename: str, data: np.array) -> None:
    path = os.path.join(output_dir, filename)
    pd.DataFrame(data).to_csv(path, index=False)


def main(output_directory: str, test_size: float = 0.3, seed: int = 42) -> None:
    print("# Load data")
    digits = datasets.load_digits()

    print("# Prepare data")
    n_samples = len(digits.images)
    images = digits.images.reshape((n_samples, -1))
    target = digits.target.reshape((n_samples, -1))

    data = np.hstack((images, target), casting="safe")

    print("# Train Test split")
    train_data, test_data = train_test_split(
        data, test_size=test_size, random_state=seed
    )

    print("# Export data")
    export(output_directory, "train_data.csv", train_data)
    export(output_directory, "test_data.csv", test_data)


if __name__ == "__main__":
    with mlflow.start_run():
        typer.run(main)
