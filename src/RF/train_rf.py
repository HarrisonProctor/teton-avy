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
    X_train = pd.read_csv(SPLITS_DIR / "X_train.csv")
    X_test = pd.read_csv(SPLITS_DIR / "X_test.csv")
    y_train = pd.read_csv(SPLITS_DIR / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(SPLITS_DIR / "y_test.csv").squeeze("columns")

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=47
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"Model Accuracy: {acc * 100:.2f}%")

if __name__ == "__main__":
    main()
