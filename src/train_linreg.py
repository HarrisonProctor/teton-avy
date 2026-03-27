import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

SCRIPT_DIR = Path(__file__).parent.resolve()
SPLITS_DIR = SCRIPT_DIR / ".." / "data" / "splits"

def main():
    try:
        X_train = pd.read_csv(SPLITS_DIR / "X_train.csv")
        X_test = pd.read_csv(SPLITS_DIR / "X_test.csv")
        y_train = pd.read_csv(SPLITS_DIR / "y_train.csv").squeeze("columns")
        y_test = pd.read_csv(SPLITS_DIR / "y_test.csv").squeeze("columns")
    except FileNotFoundError:
        print(f"Error: Frozen split CSVs not found in {SPLITS_DIR}.")
        return

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    raw_predictions = model.predict(X_test_scaled)

    # Round to nearest integer and clamp within valid class range (1-3)
    predictions = np.clip(np.round(raw_predictions), 1, 3).astype(int)

    accuracy = accuracy_score(y_test, predictions)
    print(f"Linear Regression Model Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
