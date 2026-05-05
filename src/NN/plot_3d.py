import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

script_dir = Path(__file__).parent.resolve()
splits_dir = script_dir / ".." / ".." / "data" / "splits"
output_dir = script_dir / ".." / ".." / "figures"
output_dir.mkdir(exist_ok=True)

class AvalancheNN(nn.Module):
    def __init__(self, input_size=3, hidden_size=16, num_classes=3):
        super(AvalancheNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out

def plot_3d(ax, X, y, title):
    class_labels = {1: "Low", 2: "Moderate", 3: "Considerable+"}
    class_colors = {1: "#2ecc71", 2: "#f39c12", 3: "#e74c3c"}

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

    # Same exact orientation as RF plots
    ax.view_init(elev=25, azim=135)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(title, fontsize=12, pad=10)

def main():
    torch.manual_seed(47)

    X_train = pd.read_csv(splits_dir / "X_train.csv")
    X_test = pd.read_csv(splits_dir / "X_test.csv")
    y_train = pd.read_csv(splits_dir / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(splits_dir / "y_test.csv").squeeze("columns")

    # Labels for PyTorch (0-indexed)
    y_train_pt = torch.tensor(y_train.values - 1, dtype=torch.long)
    X_train_pt = torch.tensor(X_train.values, dtype=torch.float32)
    X_test_pt = torch.tensor(X_test.values, dtype=torch.float32)

    # Train the NN quickly
    train_dataset = TensorDataset(X_train_pt, y_train_pt)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    model = AvalancheNN(input_size=3, hidden_size=16, num_classes=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    epochs = 100
    print("Training NN for plots...")
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # Predict using NN
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_pt)
        _, predicted = torch.max(outputs.data, 1)
        # Convert back to 1, 2, 3 indexing to match RF data and plotting
        nn_preds = pd.Series(predicted.numpy() + 1, index=X_test.index)

    # Plot NN predictions
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    # Title makes it explicitly clear it's from NN
    plot_3d(ax, X_test, nn_preds, "Neural Network (NN) Predictions (Test Data)")
    fig.tight_layout()
    fig.savefig(output_dir / "nn_predictions_3d.png", dpi=200)
    
    print(f"NN figure saved to {output_dir.resolve()}\\nn_predictions_3d.png")

if __name__ == "__main__":
    main()
