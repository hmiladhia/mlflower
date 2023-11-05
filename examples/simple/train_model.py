from pathlib import Path

import mlflow
import pandas as pd
import typer
from sklearn.neural_network import MLPClassifier


def main(train_data: str):
    mlflow.sklearn.autolog()
    print("Load data")
    data_path = Path(train_data, "train_data.csv")
    train_df = pd.read_csv(data_path)

    print(train_df.shape)

    # Train & target
    x_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]

    print("Train Model")
    clf = MLPClassifier(hidden_layer_sizes=(50,))
    clf.fit(x_train, y_train)

    # mlflow.sklearn.log_model(clf, "my_mlp_model")
    return clf


if __name__ == "__main__":
    with mlflow.start_run():
        typer.run(main)
