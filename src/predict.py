import mlflow
import pandas as pd
from preprocess import load_and_preprocess_data

TRACKING_URI = "http://127.0.0.1:5000/"
mlflow.set_tracking_uri(TRACKING_URI)

def simulate_production_traffic():
    model_name = "Credit_Card_Fraud_Model"
    stage = "Staging"
    model_uri = f"models:/{model_name}/{stage}"

    try:
        model = mlflow.xgboost.load_model(model_uri)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    _, X_test, _, y_test = load_and_preprocess_data(r"C:\Users\MIKAEL\PycharmProjects\TermProject\data\creditcard.csv")

    sample_X = X_test.sample(5, random_state=99)
    sample_y = y_test.loc[sample_X.index]

    predictions = model.predict(sample_X)

    for i, (idx, row) in enumerate(sample_X.iterrows()):
        actual = 1 if sample_y.iloc[i] == 1 else 0
        pred_label = 1 if predictions[i] == 1 else 0

        print(f"ID: {idx} | Pred: {pred_label} | Actual: {actual}")

if __name__ == "__main__":
    simulate_production_traffic()