import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

script_dir = Path(__file__).parent.resolve()
splits_dir = script_dir / ".." / "data" / "splits"

X_train = pd.read_csv(splits_dir / "X_train.csv")
X_test = pd.read_csv(splits_dir / "X_test.csv")
y_train = pd.read_csv(splits_dir / "y_train.csv").squeeze("columns")
y_test = pd.read_csv(splits_dir / "y_test.csv").squeeze("columns")

class_labels = {1: "Low", 2: "Moderate", 3: "Considerable+"}
class_colors = {1: "#2ecc71", 2: "#f39c12", 3: "#e74c3c"}

rf = RandomForestClassifier(n_estimators=400, max_depth=8, min_samples_split=10, min_samples_leaf=4, random_state=42)
rf.fit(X_train, y_train)
rf_preds = pd.Series(rf.predict(X_test), index=X_test.index)

scaler = StandardScaler()
lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42, solver="lbfgs")
lr.fit(scaler.fit_transform(X_train), y_train)
lr_preds = pd.Series(lr.predict(scaler.transform(X_test)), index=X_test.index)

def plot_3d(ax, X, y, title):
    for cls in sorted(y.unique()):
        mask = y == cls
        ax.scatter(
            X.loc[mask, "morning_temperature"],
            X.loc[mask, "3d_snowfall_total"],
            X.loc[mask, "wind_speed"],
            c=class_colors[cls], label=class_labels[cls],
            alpha=0.6, edgecolors="k", linewidths=0.3, s=30
        )
    ax.set_xlabel("Morning Temp (°F)")
    ax.set_ylabel("3-Day Snowfall (in)")
    ax.set_zlabel("Wind Speed (mph)")
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=8)

datasets = [
    (y_test, "Actual Hazard Ratings"),
    (rf_preds, "Random Forest Predictions"),
    (lr_preds, "Logistic Regression Predictions"),
]

for y_data, title in datasets:
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    plot_3d(ax, X_test, y_data, title)

plt.show()
