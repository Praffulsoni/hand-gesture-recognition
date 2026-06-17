import pandas as pd
import time

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

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

# Uncomment this if SVM is too slow
# df = df.sample(n=30000, random_state=42)

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
# CREATE SVM MODEL
# ==========================

svm = SVC(
    kernel="rbf",
    C=10,
    gamma="scale"
)

# ==========================
# TRAIN MODEL
# ==========================

print("\nTraining SVM...")

train_start = time.time()

svm.fit(X_train, y_train)

train_end = time.time()

training_time = train_end - train_start

# ==========================
# PREDICTION
# ==========================

print("\nMaking Predictions...")

predict_start = time.time()

predictions = svm.predict(X_test)

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