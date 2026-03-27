import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

MAX_ITER = 500
C = 1.0

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

    model = LogisticRegression(max_iter=MAX_ITER, C=C, random_state=42, solver="lbfgs")
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)

    acc = accuracy_score(y_test, preds)
    print(f"Model Accuracy: {acc * 100:.2f}%")

if __name__ == "__main__":
    main()
