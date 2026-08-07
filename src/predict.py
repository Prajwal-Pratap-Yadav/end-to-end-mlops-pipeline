import joblib
import pandas as pd

def predict(input_data: dict):
    model = joblib.load("models/model.pkl")
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]
    return int(prediction)

if __name__ == "__main__":
    sample = {f"feature_{i}": 0.1 for i in range(10)}
    print(predict(sample))
