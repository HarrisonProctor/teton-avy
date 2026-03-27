import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

SCRIPT_DIR = Path(__file__).parent.resolve()
SPLITS_DIR = SCRIPT_DIR / ".." / "data" / "splits"

def main():
    X_train = pd.read_csv(SPLITS_DIR / "X_train.csv")
    X_test = pd.read_csv(SPLITS_DIR / "X_test.csv")
    y_train = pd.read_csv(SPLITS_DIR / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(SPLITS_DIR / "y_test.csv").squeeze("columns")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_s, y_train)

    raw_preds = model.predict(X_test_s)
    preds = np.clip(np.round(raw_preds), 1, 3).astype(int)

    acc = accuracy_score(y_test, preds)
    print(f"Model Accuracy: {acc * 100:.2f}%")

if __name__ == "__main__":
    main()
