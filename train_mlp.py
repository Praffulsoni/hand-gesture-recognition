import pandas as pd
import time
import joblib

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================
# LOAD DATASET
# ==========================

print("Loading dataset...")

df = pd.read_csv("dataset.csv", header=None)

print(f"Total Samples: {len(df)}")

# ==========================
# FEATURES & LABELS
# ==========================

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# ==========================
# TRAIN TEST SPLIT
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training Samples: {len(X_train)}")
print(f"Testing Samples: {len(X_test)}")

# ==========================
# CREATE MLP MODEL
# ==========================

mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    max_iter=100,
    random_state=42
)

# ==========================
# TRAIN MODEL
# ==========================

print("\nTraining MLP...")

train_start = time.time()

mlp.fit(X_train, y_train)

# ==========================
# SAVE MODEL
# ==========================

joblib.dump(
    mlp,
    "champion_model.pkl"
)

print("\nChampion model saved!")

train_end = time.time()

training_time = train_end - train_start

# ==========================
# PREDICTIONS
# ==========================

print("\nMaking Predictions...")

predict_start = time.time()

predictions = mlp.predict(X_test)

predict_end = time.time()

prediction_time = predict_end - predict_start

# ==========================
# EVALUATION
# ==========================

accuracy = accuracy_score(y_test, predictions)

print("\n==========================")
print("RESULTS")
print("==========================")

print(f"\nAccuracy: {accuracy:.4f}")

print(f"\nTraining Time: {training_time:.2f} seconds")

print(f"\nPrediction Time: {prediction_time:.2f} seconds")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))