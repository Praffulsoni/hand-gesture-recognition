import time
import torch
import torch.nn as nn
import pandas as pd

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import( 
    accuracy_score,
    classification_report,
    confusion_matrix
    )

# =====================================================
# DEVICE
# =====================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("===================================")
print("DEVICE INFO")
print("===================================")
print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# =====================================================
# LOAD DATA
# =====================================================

print("\nLoading dataset...")

df = pd.read_csv("dataset_v4.csv", header=None)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

print("Total Samples:", len(df))

# =====================================================
# LABEL ENCODING
# =====================================================

encoder = LabelEncoder()
y = encoder.fit_transform(y)

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# =====================================================
# standard scaler
# =====================================================

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =====================================================
# TENSORS
# =====================================================

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# =====================================================
# DATALOADER
# =====================================================

BATCH_SIZE = 512

train_dataset = TensorDataset(X_train, y_train)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# =====================================================
# MODEL
# =====================================================

class GestureNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(72, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 18)

        )

    def forward(self, x):
        return self.net(x)

model = GestureNet().to(device)

# =====================================================
# LOSS + OPTIMIZER
# =====================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# =====================================================
# TRAINING
# =====================================================

EPOCHS = 100

print("\nTraining Started...\n")

start_time = time.time()

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    for batch_x, batch_y in train_loader:

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        outputs = model(batch_x)

        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    if (epoch + 1) % 5 == 0:

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Loss: {avg_loss:.4f}"
        )

training_time = time.time() - start_time

# =====================================================
# EVALUATION
# =====================================================

print("\nEvaluating...\n")

model.eval()

with torch.no_grad():

    X_test_gpu = X_test.to(device)

    outputs = model(X_test_gpu)

    _, predictions = torch.max(outputs, 1)

predictions = predictions.cpu()

accuracy = accuracy_score(
    y_test,
    predictions
)

report = classification_report(
    y_test,
    predictions
)

cm = confusion_matrix(
    y_test,
    predictions
)
# =====================================
# CLASS MAPPING
# =====================================

print("\nClass Mapping:")

for i, name in enumerate(encoder.classes_):
    print(i, "->", name)

# =====================================================
# RESULTS
# =====================================================

print("===================================")
print("RESULTS")
print("===================================")

print(f"\nAccuracy: {accuracy:.4f}")

print(f"\nTraining Time: {training_time:.2f} seconds")

print("\nClassification Report:")
print(report)

print("\nConfusion Matrix:")
print(cm)