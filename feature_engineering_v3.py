import pandas as pd
import math

print("Loading dataset...")

df = pd.read_csv("dataset.csv", header=None)

print(f"Samples Loaded: {len(df)}")

new_rows = []


# =====================================
# Helper Function
# =====================================

def calculate_angle(a, b, c):
    """
    Angle ABC in degrees
    """

    ba_x = a[0] - b[0]
    ba_y = a[1] - b[1]

    bc_x = c[0] - b[0]
    bc_y = c[1] - b[1]

    dot = ba_x * bc_x + ba_y * bc_y

    mag_ba = math.sqrt(ba_x**2 + ba_y**2)
    mag_bc = math.sqrt(bc_x**2 + bc_y**2)

    if mag_ba == 0 or mag_bc == 0:
        return 0

    cos_theta = dot / (mag_ba * mag_bc)

    # Numerical safety
    cos_theta = max(-1, min(1, cos_theta))

    angle = math.degrees(math.acos(cos_theta))

    return angle


# =====================================
# Process Dataset
# =====================================

for _, row in df.iterrows():

    features = row[:-1].tolist()
    label = row.iloc[-1]

    coords = []

    # Rebuild landmarks
    for i in range(0, 42, 2):

        x = features[i]
        y = features[i + 1]

        coords.append((x, y))

    engineered = []

    # =====================================
    # 1. Finger Bend Angles (5)
    # =====================================

    finger_triplets = [
        (2, 3, 4),      # Thumb
        (6, 7, 8),      # Index
        (10, 11, 12),   # Middle
        (14, 15, 16),   # Ring
        (18, 19, 20)    # Pinky
    ]

    for a, b, c in finger_triplets:

        angle = calculate_angle(
            coords[a],
            coords[b],
            coords[c]
        )

        engineered.append(angle)

    # =====================================
    # 2. Palm Center
    # =====================================

    palm_ids = [0, 1, 5, 9, 13, 17]

    palm_x = sum(coords[i][0] for i in palm_ids) / len(palm_ids)
    palm_y = sum(coords[i][1] for i in palm_ids) / len(palm_ids)

    palm_center = (palm_x, palm_y)

    # =====================================
    # 3. Finger-to-Finger Angles (4)
    # =====================================

    tips = [4, 8, 12, 16, 20]

    vectors = []

    for tip in tips:

        vx = coords[tip][0] - palm_center[0]
        vy = coords[tip][1] - palm_center[1]

        vectors.append((vx, vy))

    for i in range(len(vectors) - 1):

        v1 = vectors[i]
        v2 = vectors[i + 1]

        dot = v1[0] * v2[0] + v1[1] * v2[1]

        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)

        if mag1 == 0 or mag2 == 0:
            angle = 0
        else:

            cos_theta = dot / (mag1 * mag2)

            cos_theta = max(-1, min(1, cos_theta))

            angle = math.degrees(
                math.acos(cos_theta)
            )

        engineered.append(angle)

    # =====================================
    # 4. Palm Orientation Angle (1)
    # =====================================

    wrist = coords[0]
    middle_mcp = coords[9]

    dx = middle_mcp[0] - wrist[0]
    dy = middle_mcp[1] - wrist[1]

    orientation = math.degrees(
        math.atan2(dy, dx)
    )

    engineered.append(orientation)

    # =====================================
    # Final Row
    # =====================================

    new_row = features + engineered + [label]

    new_rows.append(new_row)


# =====================================
# Save Dataset
# =====================================

new_df = pd.DataFrame(new_rows)

new_df.to_csv(
    "dataset_v3.csv",
    index=False,
    header=False
)

print("\nDone!")
print("Saved as dataset_v3.csv")

print(f"Feature Count: {len(new_rows[0]) - 1}")