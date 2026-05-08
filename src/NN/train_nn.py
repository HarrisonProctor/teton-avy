import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score

SCRIPT_DIR = Path(__file__).parent.resolve()
SPLITS_DIR = SCRIPT_DIR / ".." / ".." / "data" / "splits"

class AvalancheNN(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_classes=3):
        super(AvalancheNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out

def load_data():
    from sklearn.preprocessing import StandardScaler
    
    X_train_raw = pd.read_csv(SPLITS_DIR / "X_train.csv").values
    X_test_raw = pd.read_csv(SPLITS_DIR / "X_test.csv").values
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    
    # Labels are 1, 2, 3 so we subtract 1 to get 0, 1, 2 for PyTorch CrossEntropyLoss
    y_train = pd.read_csv(SPLITS_DIR / "y_train.csv").squeeze("columns").values - 1
    y_test = pd.read_csv(SPLITS_DIR / "y_test.csv").squeeze("columns").values - 1
    
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor

def main():
    # Set random seed for reproducibility
    torch.manual_seed(47)
    
    print("Loading data...")
    X_train, y_train, X_test, y_test = load_data()
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Initialize model, loss, and optimizer
    model = AvalancheNN(input_size=3, hidden_size=64, num_classes=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 200
    print(f"Training model for {epochs} epochs...")
    
    # Training Loop
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
            
    # Evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        acc = accuracy_score(y_test.numpy(), predicted.numpy())
        print(f"\nModel Accuracy: {acc * 100:.2f}%")

if __name__ == '__main__':
    main()
