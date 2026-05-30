import mlflow
import mlflow.xgboost
import xgboost as xgb
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, f1_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from preprocess import load_and_preprocess_data

TRACKING_URI = "http://127.0.0.1:5000/"
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("Credit_Card_Fraud_Detection")

X_train, X_test, y_train, y_test = load_and_preprocess_data(r"C:\Users\MIKAEL\PycharmProjects\TermProject\data\creditcard.csv")

scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

search_space = {
    'max_depth': hp.choice('max_depth', range(3, 10)),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
    'subsample': hp.uniform('subsample', 0.6, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
    'n_estimators': hp.choice('n_estimators', range(50, 200)),
    'scale_pos_weight': scale_pos_weight
}

def objective(params):
    with mlflow.start_run(nested=True):
        mlflow.log_params(params)

        model = xgb.XGBClassifier(
            **params,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42
        )
        model.fit(X_train, y_train)

        mlflow.xgboost.log_model(model, artifact_path="model")

        y_probs = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        precisions, recalls, _ = precision_recall_curve(y_test, y_probs)
        pr_auc = auc(recalls, precisions)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_metric("pr_auc", pr_auc)
        mlflow.log_metric("f1_score", f1)

        return {'loss': 1 - pr_auc, 'status': STATUS_OK}

def run_tuning_and_training():
    with mlflow.start_run(run_name="Hyperopt_XGBoost_Session") as parent_run:
        trials = Trials()

        best_params_idx = fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=5,
            trials=trials
        )

        best_loss = trials.best_trial['result']['loss']
        best_pr_auc = 1 - best_loss

        mlflow.log_metric("best_val_pr_auc", best_pr_auc)

if __name__ == "__main__":
    run_tuning_and_training()