# Credit Card Fraud Detection

This project implements a machine learning pipeline to detect fraudulent credit card transactions. It uses XGBoost for classification, Hyperopt for hyperparameter tuning, and MLflow for experiment tracking and model registry.

## Project Structure

```text
TermProject/
├── data/
│   └── creditcard.csv       # Dataset (download and place here)
├── src/
│   ├── preprocess.py        # Data loading and preprocessing
│   ├── train.py             # Model training and hyperparameter tuning
│   ├── evaluate.py          # Model evaluation, artifact logging, and registry
│   └── predict.py           # Inference script using the registered model
├── requirements.txt         # Project dependencies
└── README.md
```

## Setup

1. Create and activate a virtual environment:
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Unix/MacOS:
source .venv/bin/activate
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Place your dataset (`creditcard.csv`) in the `data/` directory.

## Usage

To view the MLflow dashboard locally, you can start the UI in a separate terminal:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### 1. Training and Tuning
Run the training script to tune hyperparameters using Hyperopt. All runs, metrics, and models are tracked in MLflow.
```bash
python src/train.py
```

### 2. Evaluation and Registration
Evaluate the best model from the training runs, generate diagnostic plots (Precision-Recall curve, Confusion Matrix), and register it to the MLflow Model Registry in the "Staging" phase.
```bash
python src/evaluate.py
```

### 3. Prediction
Simulate production traffic by fetching the model from the Staging registry and running inferences on sample data.
```bash
python src/predict.py
```