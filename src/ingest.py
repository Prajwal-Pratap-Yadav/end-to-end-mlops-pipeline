import os
import pandas as pd
from sklearn.datasets import make_classification

def generate_sample_data(output_path: str = "data/sample.csv", n_samples: int = 1000):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    X, y = make_classification(
        n_samples=n_samples,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        random_state=42
    )
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
    df["target"] = y
    df.to_csv(output_path, index=False)
    return output_path

if __name__ == "__main__":
    print(generate_sample_data())
