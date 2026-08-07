import os
import joblib
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from src.ingest import generate_sample_data
from src.preprocess import split_data

def train():
    data_path = generate_sample_data()
    df = pd.read_csv(data_path)

    X_train, X_test, y_train, y_test = split_data(df, target_column="target")

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    return model, acc, f1

if __name__ == "__main__":
    train()
