import os
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

MAX_ITERATIONS = 500
REGULARIZATION_C = 1.0 # Smaller number = simpler model that resists memorizing data

SCRIPT_DIR = Path(__file__).parent.resolve()
SPLITS_DIR = SCRIPT_DIR / ".." / "data" / "splits"

def main():
    try:
        # Load the exact same frozen CSV splits we used for the Random Forest
        X_train = pd.read_csv(SPLITS_DIR / "X_train.csv")
        X_test = pd.read_csv(SPLITS_DIR / "X_test.csv")
        y_train = pd.read_csv(SPLITS_DIR / "y_train.csv").squeeze("columns")
        y_test = pd.read_csv(SPLITS_DIR / "y_test.csv").squeeze("columns")
    except FileNotFoundError:
        print(f"Error: Frozen split CSVs not found in {SPLITS_DIR}.")
        return

    # Logistic Regression requires normalized math scales (unlike Random Forests!)
    # We must mathematically standardize the columns so big numbers (Wind Speed 40mph)
    # don't accidentally overpower small numbers (Snowfall 1.5 inches).
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr_model = LogisticRegression(
        max_iter=MAX_ITERATIONS,
        C=REGULARIZATION_C,
        random_state=42,
        solver='lbfgs'
    )

    lr_model.fit(X_train_scaled, y_train)
    predictions = lr_model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, predictions)
    print(f"Logistic Regression Model Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
