import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

N_ESTIMATORS = 400
MAX_DEPTH = 8
MIN_SAMPLES_SPLIT = 10
MIN_SAMPLES_LEAF = 4
TEST_SIZE = 0.20

SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_PATH = SCRIPT_DIR / ".." / "data" / "processed" / "teton_am_archive_clean.csv"

def load_and_preprocess_data(filepath):
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {filepath}")
        return None, None

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    df['3d_snowfall_total'] = df['recent_snowfall'].rolling(window=3).sum()
    df = df.dropna()

    X = df[['morning_temperature', '3d_snowfall_total', 'wind_speed']]
    y = df['morning_avalanche_hazard'].replace([4, 5], 3)

    return X, y

def main():
    X, y = load_and_preprocess_data(DATA_PATH)
    if X is None:
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, shuffle=True, random_state=None
    )

    rf_model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=None,
        class_weight=None
    )

    rf_model.fit(X_train, y_train)
    predictions = rf_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
