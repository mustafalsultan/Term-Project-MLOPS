import mlflow
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, classification_report

from preprocess import load_and_preprocess_data

TRACKING_URI = "http://127.0.0.1:5000/"
mlflow.set_tracking_uri(TRACKING_URI)
client = mlflow.tracking.MlflowClient()

def evaluate_and_register_best_model():
    experiment = client.get_experiment_by_name("Credit_Card_Fraud_Detection")
    
    if not experiment:
        return

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="metrics.pr_auc > 0",
        order_by=["metrics.pr_auc DESC"],
        max_results=20
    )

    best_run = None
    best_run_id = None
    model = None

    for run in runs:
        run_id = run.info.run_id
        model_uri = f"runs:///{run_id}/model"
        try:
            model = mlflow.xgboost.load_model(model_uri)
            best_run = run
            best_run_id = run_id
            break
        except Exception:
            continue

    if not best_run or not model:
        return

    best_pr_auc = best_run.data.metrics['pr_auc']

    X_train, X_test, y_train, y_test = load_and_preprocess_data(r"C:\Users\MIKAEL\PycharmProjects\TermProject\data\creditcard.csv")

    with mlflow.start_run(run_name="Final_Evaluation_&_Registry"):
        mlflow.log_param("source_run_id", best_run_id)

        y_probs = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path, bbox_inches='tight')
        plt.close()

        precisions, recalls, _ = precision_recall_curve(y_test, y_probs)
        plt.figure(figsize=(6, 5))
        plt.plot(recalls, precisions, label=f'XGBoost (AUC = {best_pr_auc:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='best')
        pr_curve_path = "precision_recall_curve.png"
        plt.savefig(pr_curve_path, bbox_inches='tight')
        plt.close()

        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(pr_curve_path)

        report = classification_report(y_test, y_pred, output_dict=True)
        with open("classification_report.json", "w") as f:
            json.dump(report, f, indent=4)
        mlflow.log_artifact("classification_report.json")

        os.remove(cm_path)
        os.remove(pr_curve_path)
        os.remove("classification_report.json")

        model_name = "Credit_Card_Fraud_Model"
        model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

        client.transition_model_version_stage(
            name=model_name,
            version=model_details.version,
            stage="Staging"
        )

if __name__ == "__main__":
    evaluate_and_register_best_model()