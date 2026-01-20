import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

# =====================
# Device (GPU)
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print("GPU:", torch.cuda.get_device_name(0))

# =====================
# Load data
# =====================
df = pd.read_csv("data/processed/cleaned_data.csv")

X = df.drop("Churn", axis=1).values
y = df["Churn"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =====================
# Dataset
# =====================
class ChurnDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(
    ChurnDataset(X_train, y_train),
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    ChurnDataset(X_test, y_test),
    batch_size=64,
    shuffle=False
)

# =====================
# Neural Network
# =====================
class ChurnNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)

model = ChurnNN(X.shape[1]).to(device)

# =====================
# Train
# =====================
pos_weight = torch.tensor(
    (y == 0).sum() / (y == 1).sum(),
    dtype=torch.float32
).to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

epochs = 100
for epoch in range(epochs):
    model.train()
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device).unsqueeze(1)

        optimizer.zero_grad()
        preds = model(Xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

# =====================
# Evaluation
# =====================
model.eval()
y_pred = []

threshold = 0.4
with torch.no_grad():
    for Xb, _ in test_loader:
        Xb = Xb.to(device)
        preds = model(Xb)
        probs = torch.sigmoid(preds) 
        y_pred.extend((preds.cpu().numpy() > 0.5).astype(int))

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))

# =====================
# Export predictions
# =====================
with torch.no_grad():
    all_preds = model(torch.tensor(X, dtype=torch.float32).to(device))
    df["Churn_Prediction"] = (all_preds.cpu().numpy() > 0.5).astype(int)

df.to_csv("data/processed/churn_prediction.csv", index=False)
