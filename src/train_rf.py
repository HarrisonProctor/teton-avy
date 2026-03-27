import os
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

N_ESTIMATORS = 400
MAX_DEPTH = 8
MIN_SAMPLES_SPLIT = 10
MIN_SAMPLES_LEAF = 4

SCRIPT_DIR = Path(__file__).parent.resolve()
SPLITS_DIR = SCRIPT_DIR / ".." / "data" / "splits"

def main():
    try:
        # Load the exact frozen CSV matrices directly from disk
        X_train = pd.read_csv(SPLITS_DIR / "X_train.csv")
        X_test = pd.read_csv(SPLITS_DIR / "X_test.csv")
        y_train = pd.read_csv(SPLITS_DIR / "y_train.csv").squeeze("columns") # Squeeze back to a Pandas Series
        y_test = pd.read_csv(SPLITS_DIR / "y_test.csv").squeeze("columns")
    except FileNotFoundError:
        print(f"Error: 62% frozen split CSVs not found in {SPLITS_DIR}.")
        return

    rf_model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=47,
        class_weight=None
    )

    rf_model.fit(X_train, y_train)
    predictions = rf_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    print(f"Frozen Load Successful. Model Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
