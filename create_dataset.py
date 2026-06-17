import json
import csv
import math
import os

# Path to annotation files
DATASET_PATH = r"C:\Users\prafful\Desktop\dataset\ann_train_val"

# Output CSV
OUTPUT_CSV = "dataset.csv"

# Samples per gesture
SAMPLES_PER_CLASS = 10000


def normalize_landmarks(landmarks):
    """
    landmarks = [[x,y], [x,y], ...]
    """

    wrist_x, wrist_y = landmarks[0]

    # Center around wrist
    centered = []

    for x, y in landmarks:
        centered.append([x - wrist_x, y - wrist_y])

    # Find max distance
    max_dist = 0

    for x, y in centered:
        dist = math.sqrt(x**2 + y**2)

        if dist > max_dist:
            max_dist = dist

    # Avoid division by zero
    if max_dist == 0:
        max_dist = 1

    normalized = []

    for x, y in centered:
        normalized.append(x / max_dist)
        normalized.append(y / max_dist)

    return normalized


all_rows = []

# Loop through every gesture file
for filename in os.listdir(DATASET_PATH):

    if not filename.endswith(".json"):
        continue

    gesture_label = filename.replace(".json", "")

    print(f"Processing {gesture_label}...")

    file_path = os.path.join(DATASET_PATH, filename)

    with open(file_path, "r") as f:
        data = json.load(f)

    count = 0

    for sample_id, sample_data in data.items():

        if count >= SAMPLES_PER_CLASS:
            break

        landmarks = sample_data["landmarks"][0]

        # Ensure 21 landmarks
        if len(landmarks) != 21:
            continue

        features = normalize_landmarks(landmarks)

        features.append(gesture_label)

        all_rows.append(features)

        count += 1

print(f"\nTotal Samples Collected: {len(all_rows)}")

# Save CSV
with open(OUTPUT_CSV, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerows(all_rows)

print(f"Dataset saved as {OUTPUT_CSV}")