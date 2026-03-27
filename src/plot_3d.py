import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

script_dir = Path(__file__).parent.resolve()
splits_dir = script_dir / ".." / "data" / "splits"
output_dir = script_dir / ".." / "figures"
output_dir.mkdir(exist_ok=True)

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
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42, solver="lbfgs")
lr.fit(X_train_s, y_train)
lr_preds = pd.Series(lr.predict(X_test_s), index=X_test.index)

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
    ax.set_xlabel("Morning Temp (°F)", fontsize=9)
    ax.set_ylabel("3-Day Snowfall (in)", fontsize=9)
    ax.set_zlabel("Wind Speed (mph)", fontsize=9)

    ax.view_init(elev=25, azim=135)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.legend(fontsize=8, loc="upper left")

fig1 = plt.figure(figsize=(8, 6))
ax1 = fig1.add_subplot(111, projection="3d")
plot_3d(ax1, X_test, y_test, "Actual Avalanche Hazard (Test Data)")
fig1.tight_layout()
fig1.savefig(output_dir / "actual_hazard_3d.png", dpi=200)

fig2 = plt.figure(figsize=(8, 6))
ax2 = fig2.add_subplot(111, projection="3d")
plot_3d(ax2, X_test, rf_preds, "Random Forest Predictions (Test Data)")
fig2.tight_layout()
fig2.savefig(output_dir / "rf_predictions_3d.png", dpi=200)

fig3 = plt.figure(figsize=(8, 6))
ax3 = fig3.add_subplot(111, projection="3d")
plot_3d(ax3, X_test, lr_preds, "Logistic Regression Predictions (Test Data)")
fig3.tight_layout()
fig3.savefig(output_dir / "lr_predictions_3d.png", dpi=200)

fig4, ax4 = plt.subplots(figsize=(8, 4))
importances = rf.feature_importances_
names = ["Morning Temp", "3-Day Snowfall", "Wind Speed"]
idx = np.argsort(importances)[::-1]
bars = ax4.bar([names[i] for i in idx], [importances[i] for i in idx],
               color=["#e74c3c", "#f39c12", "#2ecc71"], edgecolor="k", linewidth=0.5)
for bar, val in zip(bars, [importances[i] for i in idx]):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f"{val*100:.1f}%", ha="center", fontsize=11, fontweight="bold")
ax4.set_ylabel("Importance")

ax4.set_ylim(0, max(importances) + 0.08)
ax4.grid(True, axis="y", alpha=0.3)
fig4.tight_layout()
fig4.savefig(output_dir / "feature_importance.png", dpi=200)

print("All figures saved to /figures.")
