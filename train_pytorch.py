import time
import torch
import torch.nn as nn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# ======================
# DEVICE
# ======================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# ======================
# LOAD DATA
# ======================

print("\nLoading dataset...")

df = pd.read_csv("dataset.csv", header=None)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

print("Total Samples:", len(df))

# ======================
# LABEL ENCODING
# ======================

encoder = LabelEncoder()

y = encoder.fit_transform(y)

# ======================
# TRAIN TEST SPLIT
# ======================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# ======================
# TENSORS
# ======================

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

X_train = X_train.to(device)
X_test = X_test.to(device)

y_train = y_train.to(device)
y_test = y_test.to(device)

# ======================
# MODEL
# ======================

class GestureNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(42, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 18)
        )

    def forward(self, x):
        return self.net(x)

model = GestureNet().to(device)

# ======================
# LOSS + OPTIMIZER
# ======================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# ======================
# TRAINING
# ======================

epochs = 50

print("\nTraining...\n")

start = time.time()

for epoch in range(epochs):

    model.train()

    outputs = model(X_train)

    loss = criterion(outputs, y_train)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 5 == 0:
        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss: {loss.item():.4f}"
        )

train_time = time.time() - start

# ======================
# EVALUATION
# ======================

model.eval()

with torch.no_grad():

    outputs = model(X_test)

    _, predictions = torch.max(outputs, 1)

accuracy = accuracy_score(
    y_test.cpu(),
    predictions.cpu()
)

# ======================
# RESULTS
# ======================

print("\n===================")
print("RESULTS")
print("===================")

print(f"\nAccuracy: {accuracy:.4f}")

print(f"\nTraining Time: {train_time:.2f} sec")