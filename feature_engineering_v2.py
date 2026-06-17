import pandas as pd
import math

print("Loading dataset...")

df = pd.read_csv("dataset.csv", header=None)

print(f"Samples Loaded: {len(df)}")

new_rows = []

# Fingertips
TIP_IDS = [4, 8, 12, 16, 20]

# Palm landmarks
PALM_IDS = [0, 1, 5, 9, 13, 17]

for _, row in df.iterrows():

    features = row[:-1].tolist()
    label = row.iloc[-1]

    coords = []

    # ====================================
    # Reconstruct landmarks
    # ====================================

    for i in range(0, 42, 2):
        x = features[i]
        y = features[i + 1]

        coords.append((x, y))

    engineered = []

    # ====================================
    # 1. Finger Lengths (5)
    # ====================================

    wrist_x, wrist_y = coords[0]

    for tip in TIP_IDS:

        x, y = coords[tip]

        dist = math.sqrt(
            (x - wrist_x) ** 2 +
            (y - wrist_y) ** 2
        )

        engineered.append(dist)

    # ====================================
    # 2. Fingertip Distances (10)
    # ====================================

    for i in range(len(TIP_IDS)):
        for j in range(i + 1, len(TIP_IDS)):

            x1, y1 = coords[TIP_IDS[i]]
            x2, y2 = coords[TIP_IDS[j]]

            dist = math.sqrt(
                (x1 - x2) ** 2 +
                (y1 - y2) ** 2
            )

            engineered.append(dist)

    # ====================================
    # 3. Palm Center
    # ====================================

    palm_x = 0
    palm_y = 0

    for idx in PALM_IDS:
        palm_x += coords[idx][0]
        palm_y += coords[idx][1]

    palm_x /= len(PALM_IDS)
    palm_y /= len(PALM_IDS)

    # ====================================
    # 4. Palm Center -> Fingertips (5)
    # ====================================

    for tip in TIP_IDS:

        x, y = coords[tip]

        dist = math.sqrt(
            (x - palm_x) ** 2 +
            (y - palm_y) ** 2
        )

        engineered.append(dist)

    # ====================================
    # 5. Finger Spread Distances (4)
    # ====================================

    adjacent_pairs = [
        (4, 8),
        (8, 12),
        (12, 16),
        (16, 20)
    ]

    for a, b in adjacent_pairs:

        x1, y1 = coords[a]
        x2, y2 = coords[b]

        dist = math.sqrt(
            (x1 - x2) ** 2 +
            (y1 - y2) ** 2
        )

        engineered.append(dist)

    # ====================================
    # 6. Palm Geometry (4)
    # ====================================

    # Palm Width
    x1, y1 = coords[5]
    x2, y2 = coords[17]

    palm_width = math.sqrt(
        (x1 - x2) ** 2 +
        (y1 - y2) ** 2
    )

    engineered.append(palm_width)

    # Palm Height
    x1, y1 = coords[0]
    x2, y2 = coords[9]

    palm_height = math.sqrt(
        (x1 - x2) ** 2 +
        (y1 - y2) ** 2
    )

    engineered.append(palm_height)

    # Palm Diagonal
    x1, y1 = coords[0]
    x2, y2 = coords[17]

    palm_diag = math.sqrt(
        (x1 - x2) ** 2 +
        (y1 - y2) ** 2
    )

    engineered.append(palm_diag)

    # Thumb-Pinky Span
    x1, y1 = coords[4]
    x2, y2 = coords[20]

    span = math.sqrt(
        (x1 - x2) ** 2 +
        (y1 - y2) ** 2
    )

    engineered.append(span)

    # ====================================
    # Final Row
    # ====================================

    new_row = features + engineered + [label]

    new_rows.append(new_row)

# ====================================
# Save Dataset
# ====================================

new_df = pd.DataFrame(new_rows)

new_df.to_csv(
    "dataset_v2.csv",
    index=False,
    header=False
)

print("\nDone!")
print("Saved as dataset_v2.csv")

print(f"Feature Count: {len(new_rows[0]) - 1}")