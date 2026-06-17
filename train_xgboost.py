import pandas as pd
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# ==========================
# LOAD DATASET
# ==========================

print("Loading dataset...")

df = pd.read_csv("dataset.csv", header=None)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

print("Total Samples:", len(df))

# ==========================
# ENCODE LABELS
# ==========================

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# ==========================
# TRAIN TEST SPLIT
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# ==========================
# MODEL
# ==========================

model = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softmax",
    eval_metric="mlogloss",
    random_state=42
)

# ==========================
# TRAINING
# ==========================

print("\nTraining XGBoost...")

start_train = time.time()

model.fit(X_train, y_train)

end_train = time.time()

# ==========================
# PREDICTION
# ==========================

print("\nMaking Predictions...")

start_pred = time.time()

predictions = model.predict(X_test)

end_pred = time.time()

# ==========================
# RESULTS
# ==========================

accuracy = accuracy_score(y_test, predictions)

print("\n======================")
print("RESULTS")
print("======================")

print(f"\nAccuracy: {accuracy:.4f}")

print(f"\nTraining Time: {(end_train-start_train):.2f} seconds")

print(f"\nPrediction Time: {(end_pred-start_pred):.2f} seconds")

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        predictions,
        target_names=encoder.classes_
    )
)

print("\nConfusion Matrix:\n")

print(confusion_matrix(y_test, predictions))