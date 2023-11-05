from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import typer
from sklearn import metrics


def main(test_data: str, model_uri: str) -> None:
    mlflow.sklearn.autolog()

    print("Load test data")
    data_path = Path(test_data, "test_data.csv")
    test_df = pd.read_csv(data_path)

    x_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1]

    print("Load Model")
    loaded_model = mlflow.pyfunc.load_model(model_uri)

    print("Predict ")
    predicted = loaded_model.predict(x_test)

    print("Classification report")
    clf_report = metrics.classification_report(y_test, predicted)
    print(clf_report)

    print("Confusion Matrix")
    cm_disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    log_confusion_matrix(cm_disp)


def log_confusion_matrix(cm_disp: metrics.ConfusionMatrixDisplay) -> None:
    conf_matrix = cm_disp.confusion_matrix
    print(conf_matrix)

    mlflow.log_metric("TP", conf_matrix[0][0])
    mlflow.log_metric("TN", conf_matrix[1][1])
    mlflow.log_metric("FP", conf_matrix[0][1])
    mlflow.log_metric("FN", conf_matrix[1][0])

    fig, ax = plt.subplots()
    cm_disp.plot(ax=ax)
    mlflow.log_figure(fig, "confusion_matrix.png")


if __name__ == "__main__":
    with mlflow.start_run():
        typer.run(main)
