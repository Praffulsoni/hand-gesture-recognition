# 🪐 Aura: Real-Time Hand Gesture Recognition

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.14-0097A7?logo=google&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-MLP-F7931E?logo=scikitlearn&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)
![Pygame](https://img.shields.io/badge/Pygame-2.6-2E7D32)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-91.27%25-brightgreen)

**Aura** recognises **18 static hand gestures** live from a webcam. **MediaPipe Hands** finds 21 landmarks on your hand, those landmarks are normalised into a 42-number feature vector, and a lightweight **scikit-learn MLP** classifies the gesture on every frame. The result is shown in a full-screen **Pygame** interface with translucent "glassmorphism" panels, a skeleton overlay, a stability meter and a detection history.

Five model families were benchmarked (Random Forest, SVM, XGBoost, PyTorch and an sklearn MLP). The **MLP won with 91.27% test accuracy** and predicts in milliseconds, which makes it a good fit for real-time use.

---

## 📑 Table of Contents

- [Demo Pipeline](#-demo-pipeline)
- [Supported Gestures](#-supported-gestures)
- [Features](#-features)
- [How It Works](#-how-it-works)
- [Model Benchmarks](#-model-benchmarks)
- [Feature Engineering Experiments](#-feature-engineering-experiments)
- [Engineering Challenges and Fixes](#-engineering-challenges-and-fixes)
- [Project Structure](#-project-structure)
- [Installation](#️-installation)
- [Usage](#-usage)
- [Re-training the Model](#-re-training-the-model)
- [Troubleshooting](#-troubleshooting)
- [Limitations and Future Work](#-limitations-and-future-work)
- [Tech Stack](#️-tech-stack)
- [Author](#-author)
- [Acknowledgements](#-acknowledgements)

---

## 🎬 Demo Pipeline

```
 ┌─────────────┐   ┌──────────────────┐   ┌───────────────────────┐   ┌──────────────┐
 │  Camera     │──►│ MediaPipe Hands  │──►│ Normalise landmarks   │──►│ MLP (128,64) │
 │ 1280×720    │   │ 21 (x, y) points │   │ wrist-centred, scaled │   │ 18 classes   │
 │ (MSMF)      │   └──────────────────┘   │ → 42 features         │   └──────┬───────┘
 └─────────────┘                          └───────────────────────┘          │
                                                                              ▼
 ┌──────────────────────────────────────────────┐   ┌──────────────────────────────┐
 │ Pygame full-screen UI                        │◄──│ Temporal smoothing           │
 │ skeleton · FPS · stability · history · label │   │ 12-frame majority vote       │
 └──────────────────────────────────────────────┘   └──────────────────────────────┘
```

---

## ✋ Supported Gestures

The model recognises the 18 gesture classes of the [HaGRID](https://github.com/hukenovs/hagrid) dataset:

| | | | |
| :--- | :--- | :--- | :--- |
| 🤙 `call` | 👎 `dislike` | ✊ `fist` | 🖐️ `four` |
| 👍 `like` | 🤫 `mute` | 👌 `ok` | ☝️ `one` |
| 🖐 `palm` | ✌️ `peace` | 🔻 `peace_inverted` | 🤘 `rock` |
| ✋ `stop` | 🔻 `stop_inverted` | 🤟 `three` | 🤟 `three2` |
| ✌️ `two_up` | 🔻 `two_up_inverted` | | |

`*_inverted` classes are the same hand shape pointing downwards. `three` and `three2` are two different ways of showing three fingers.

---

## ✨ Features

- 🎯 **18-class static gesture recognition** at 91.27% test accuracy
- ⚡ **Real-time inference** using a small MLP. Predicting all 18,000 test samples takes about 0.03 s.
- 📐 **Translation- and scale-invariant features**, so the hand can be anywhere in the frame and at any distance
- 🧘 **Flicker-free output** thanks to a 12-frame majority-vote buffer
- 🖥️ **Full-screen Pygame UI** with anti-aliased text, translucent cards and a mirrored, selfie-style camera view
- 🦴 **Live hand skeleton overlay** (orange joints, black bones)
- 📊 **On-screen telemetry:** engine FPS, tracking status, signal-stability meter and the last 3 detected gestures
- 📷 **External HD camera support** (e.g. Iriun / Camo phone webcam), with automatic fallback to the built-in webcam
- 🧪 **Benchmarking suite** with training scripts for 5 model families and 3 feature-engineering variants

---

## 🔍 How It Works

### 1. Landmark extraction

MediaPipe Hands returns 21 landmarks per hand (wrist, plus 4 joints for each finger). Only the `x` and `y` coordinates are used, giving **21 × 2 = 42 values**.

```
         8   12  16  20        ← fingertips
         |   |   |   |
         7   11  15  19
    4    |   |   |   |
    |    6   10  14  18
    3    |   |   |   |
     \   5---9---13--17
      2  |          /
       \ |         /
        1|        /
          \      /
           \    /
             0                  ← wrist
```

### 2. Normalisation

Raw MediaPipe coordinates depend on where the hand is on screen and how close it is to the camera. Each landmark is therefore re-expressed relative to the wrist and divided by the largest wrist-to-landmark distance:

$$\mathbf{f}_i = \frac{\mathbf{p}_i - \mathbf{p}_{\text{wrist}}}{\max_j \lVert \mathbf{p}_j - \mathbf{p}_{\text{wrist}} \rVert}$$

- **Translation invariant:** the wrist is always at `(0, 0)`.
- **Scale invariant:** the hand always fits inside a unit circle.

The **same function** is used in `create_dataset.py` (training) and `test_camera.py` (live inference). This is what makes the live predictions match the offline accuracy.

### 3. Classification

`champion_model.pkl` is a scikit-learn `MLPClassifier`:

| Layer | Size | Activation |
| :--- | :---: | :---: |
| Input | 42 | – |
| Hidden 1 | 128 | ReLU |
| Hidden 2 | 64 | ReLU |
| Output | 18 | Softmax |

Trained with the Adam optimiser for up to 100 iterations (`random_state=42`).

### 4. Temporal smoothing

Single-frame predictions can jump around because of sensor noise. The last **12 predictions** are kept in a `collections.deque`, and the UI shows the **most common** one (`collections.Counter`). The share of the buffer that agrees with it is shown as **Signal Stability** (e.g. 12/12 → 100%). When the hand leaves the frame the buffer drains one entry per frame, so the label fades out instead of disappearing instantly.

---

## 📊 Model Benchmarks

All models use the same 42-feature dataset, an **80/20 stratified train/test split** and `random_state=42`. Results are from [`results_summary.txt`](results_summary.txt) (baseline run on 18 classes × 5,000 samples = 90,000 samples).

| Model | Script | Test Accuracy | Train Time | Predict Time (test set) |
| :--- | :--- | :---: | :---: | :---: |
| Random Forest (100 trees) | `train_rf.py` | 90.18% | – | – |
| SVM (RBF, C=10) | `train_svm.py` | 90.88% | 69.39 s | 32.31 s |
| XGBoost (300 trees, depth 8) | `train_xgboost.py` | 90.93% | – | – |
| PyTorch MLP (42→128→64→18) | `train_pytorch.py` | 91.13% | – | – |
| **sklearn MLP (128, 64)** 🏆 | **`train_mlp.py`** | **91.27%** | **71.55 s** | **0.03 s** |

**Why the MLP was chosen:** it has the highest accuracy, and it predicts **about 1000× faster than the SVM** (0.03 s vs 32 s for the whole test set). For a real-time video loop, prediction speed matters as much as accuracy.

### How the champion was found

Training was an iterative study rather than a single run:

1. **Dataset scaling.** Models were first trained on 1,000 samples per class, then 2,000 and 5,000, and finally 10,000 per class. The production dataset (`dataset.csv`) has **180,000 samples** (18 × 10,000).
2. **Feature-set iterations (v1 to v4).** Each feature set was compared using loss curves and confusion matrices to find which gestures were being confused with each other (see [Feature Engineering Experiments](#-feature-engineering-experiments)).
3. **Hyperparameter tuning.** The MLP was tuned by changing the number of hidden layers, neurons per layer, activation functions and regularisation.
4. **GPU experiments.** Deeper networks were trained in PyTorch on CUDA to speed up iteration and test whether more depth helps.

---

## 🧪 Feature Engineering Experiments

Three richer feature sets were built on top of the 42 normalised coordinates to see whether hand-crafted geometry would help:

| Script | Output | Extra features added | Total |
| :--- | :--- | :--- | :---: |
| `feature_engineering_v2.py` | `dataset_v2.csv` | 5 finger lengths, 10 fingertip-to-fingertip distances, 5 palm-centre-to-tip distances, 4 adjacent-finger spreads, 4 palm geometry measures | 70 |
| `feature_engineering_v3.py` | `dataset_v3.csv` | 5 finger bend angles, 4 angles between neighbouring fingers, 1 palm orientation angle | 52 |
| `feature_engineeringV4.py` | `dataset_v4.csv` | v3 + 5 extra joint angles + 5 finger extension distances + 10 fingertip distances | 72 |

`train_pytorchV2.py` trains a deeper network (72→512→256→128→64→18, mini-batches of 512, `StandardScaler`, 100 epochs) on the v4 feature set.

**Outcome:** the extra features did not beat the plain 42-coordinate baseline by a meaningful margin, and they would make live inference more complex. The **42-feature pipeline stays the production standard.**

---

## 🛠️ Engineering Challenges and Fixes

<details>
<summary><b>1. Live predictions stuck on a single class (<code>two_up</code>)</b></summary>

**Problem:** the model scored 91% offline but in the live feed predicted `two_up` almost regardless of the gesture.

**Cause:** the training data was wrist-centred and scaled, but the live pipeline fed raw MediaPipe coordinates (absolute values between 0 and 1). The model was seeing inputs from a completely different distribution.

**Fix:** `normalize_live_landmarks()` in `test_camera.py` now applies exactly the same normalisation as `create_dataset.py`.
</details>

<details>
<summary><b>2. Flickering predictions</b></summary>

**Problem:** small frame-to-frame jitter in landmark positions made the label switch rapidly between similar gestures.

**Fix:** a 12-frame majority-vote buffer (`deque` + `Counter`). The displayed label only changes once most recent frames agree.
</details>

<details>
<summary><b>3. Limited UI with OpenCV windows</b></summary>

**Problem:** `cv2.imshow` with `cv2.putText` gives blocky text and no transparency.

**Fix:** the frame is rendered into a full-screen, hardware-accelerated, double-buffered **Pygame** window. This allows anti-aliased fonts, rounded semi-transparent panels and smooth scaling to the monitor's native resolution.
</details>

<details>
<summary><b>4. Motion blur from a low-quality laptop webcam</b></summary>

**Problem:** fast hand movements blurred, and landmark tracking dropped out.

**Fix:** a phone is used as an HD webcam (Iriun over USB-C), opened through the Windows **Media Foundation** backend (`cv2.CAP_MSMF`) at 1280×720. If that camera is not available the app falls back to the default webcam.
</details>

---

## 📂 Project Structure

```
gesture-detection/
│
├── 🚀 Application
│   ├── test_camera.py            # Live app: camera → MediaPipe → MLP → Pygame UI
│   └── champion_model.pkl        # Trained sklearn MLP (production model)
│
├── 🗃️ Data preparation
│   ├── create_dataset.py         # HaGRID JSON annotations → normalised dataset.csv
│   ├── test_parser.py            # Inspect the structure of one HaGRID JSON file
│   ├── feature_engineering_v2.py # → dataset_v2.csv (70 features)
│   ├── feature_engineering_v3.py # → dataset_v3.csv (52 features)
│   └── feature_engineeringV4.py  # → dataset_v4.csv (72 features)
│
├── 🧠 Model training and benchmarking
│   ├── train_mlp.py              # sklearn MLP, saves champion_model.pkl  🏆
│   ├── train_rf.py               # Random Forest
│   ├── train_svm.py              # SVM (RBF kernel)
│   ├── train_xgboost.py          # XGBoost
│   ├── train_pytorch.py          # PyTorch MLP on 42 features
│   ├── train_pytorchV2.py        # Deeper PyTorch MLP on 72 features (v4)
│   ├── verification.py           # Print details of the saved champion model
│   └── results_summary.txt       # Recorded benchmark results
│
├── 🔧 Hardware checks
│   ├── webcam_check.py           # Scan camera indices 0–4 and print resolutions
│   ├── cuda_check.py             # Print PyTorch / CUDA versions
│   └── gpu_check.py              # Print GPU name if CUDA is available
│
├── requirement.txt               # Dependencies for running the live app
└── README.md
```

> `*.csv` datasets are git-ignored because of their size. The trained `champion_model.pkl` **is** included, so you can run the live app without training anything.

---

## ⚙️ Installation

> **Tested with Python 3.11.9 on Windows.** The camera code uses the Windows-only `cv2.CAP_MSMF` backend (see [Troubleshooting](#-troubleshooting) for macOS / Linux).

### 1. Clone the repository

```bash
git clone https://github.com/Praffulsoni/hand-gesture-recognition.git
cd hand-gesture-recognition
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

To **run the live app**:

```bash
pip install -r requirement.txt
```

To also **re-train and benchmark** the models:

```bash
pip install pandas xgboost torch
```

> 💡 The app uses MediaPipe's `mp.solutions.hands` API. If a newer MediaPipe release no longer provides it, install the tested version: `pip install mediapipe==0.10.14`.

---

## 🚀 Usage

### 1. (Optional) Find your camera index

```bash
python webcam_check.py
```

```
=== STARTING WEB CAM SLOT SCAN ===
Index 0: WORKING | Resolution: 640x480
Index 1: WORKING | Resolution: 1280x720
Index 2: Closed/Unavailable
...
```

`test_camera.py` tries **index 1** first (an external/phone camera) and falls back to **index 0**. To use a different camera, change `chosen_index` near the top of the "HD HARDWARE ENGINE INITIALIZATION" section.

### 2. Launch Aura

```bash
python test_camera.py
```

The app opens **full screen** at your monitor's resolution. Hold one hand up in front of the camera and make any of the 18 gestures.

| UI element | Position | What it shows |
| :--- | :--- | :--- |
| **System Core** | top left | Engine FPS (capped at 60) and `TRACKING` / `IDLE` status |
| **Signal Stability** | top right | % of the last 12 frames that agree with the shown gesture |
| **Recent Detections** | bottom left | The last 3 distinct gestures |
| **Prediction banner** | bottom centre | The current smoothed gesture, or `STANDBY` |

Press **`Esc`** to exit.

### 3. (Optional) Inspect the saved model

```bash
python verification.py
```

---

## 🔁 Re-training the Model

1. **Download the HaGRID annotations** (the `ann_train_val` folder with one JSON file per gesture) from the [HaGRID repository](https://github.com/hukenovs/hagrid).

2. **Point `create_dataset.py` at it** by editing `DATASET_PATH`, and optionally change `SAMPLES_PER_CLASS` (default 10,000):

   ```python
   DATASET_PATH = r"C:\path\to\ann_train_val"
   SAMPLES_PER_CLASS = 10000
   ```

3. **Build the dataset:**

   ```bash
   python create_dataset.py        # → dataset.csv  (42 features + label per row)
   ```

4. **Train the champion model:**

   ```bash
   python train_mlp.py             # → champion_model.pkl + accuracy, report, confusion matrix
   ```

5. **(Optional) Run the other benchmarks:**

   ```bash
   python train_rf.py
   python train_svm.py
   python train_xgboost.py
   python train_pytorch.py
   ```

6. **(Optional) Try the engineered features:**

   ```bash
   python feature_engineeringV4.py # → dataset_v4.csv
   python train_pytorchV2.py
   ```

> 💡 `train_svm.py` is slow on large datasets. Uncomment the `df.sample(n=30000, ...)` line in that script to train on a subset.

---

## 🧯 Troubleshooting

| Problem | Fix |
| :--- | :--- |
| Black screen or `Camera matrix frame drop detected` | Wrong camera index or the camera is in use by another app. Run `webcam_check.py` and set `chosen_index`. |
| Not on Windows / MSMF fails | Replace `cv2.CAP_MSMF` with `cv2.CAP_ANY` (or remove the second argument) in `test_camera.py`. |
| `AttributeError: module 'mediapipe' has no attribute 'solutions'` | Install the tested version: `pip install mediapipe==0.10.14`. |
| Label stays on `STANDBY` | No hand detected. Improve lighting, move closer, and keep the whole hand in frame. |
| Prediction keeps switching | Hold the gesture steady for about half a second so the 12-frame buffer can settle. |
| Low FPS | Use a lower camera resolution, or close other apps using the CPU. |
| `FileNotFoundError: champion_model.pkl` | Run the app from inside the project folder, or re-train with `train_mlp.py`. |
| `FileNotFoundError: dataset.csv` | The CSV is not in the repo. Generate it with `create_dataset.py`. |

---

## 🧭 Limitations and Future Work

**Current limitations**

- Recognises **static** gestures only (no swipes or other motion).
- Tracks **one hand** at a time (`max_num_hands=1`).
- Uses only `x, y` coordinates, so depth (`z`) and handedness are ignored.
- Camera setup is tuned for Windows (`CAP_MSMF`).

**Ideas for the future**

- [ ] Map gestures to actions (media controls, volume, slides, mouse control)
- [ ] Dynamic gestures with an LSTM / temporal model over landmark sequences
- [ ] Two-hand support and handedness as a feature
- [ ] Use the MLP's `predict_proba` to show a real confidence score and reject low-confidence frames
- [ ] Add the `z` coordinate and rotation normalisation
- [ ] Export to ONNX / TFLite for mobile or browser use
- [ ] Command-line options for camera index, buffer size and windowed mode

---

## 🛠️ Tech Stack

| Area | Tools |
| :--- | :--- |
| Language | Python 3.11 |
| Hand tracking | MediaPipe Hands |
| Video capture | OpenCV (`cv2.CAP_MSMF`) |
| Production model | scikit-learn `MLPClassifier`, Joblib |
| Benchmark models | Random Forest, SVM, XGBoost, PyTorch (CUDA) |
| Data processing | NumPy, pandas |
| User interface | Pygame |

---

## 👨‍💻 Author

**Prafful Soni**
B.Tech IT student, interested in AI, Machine Learning and Computer Vision 🚀

GitHub: [@Praffulsoni](https://github.com/Praffulsoni)

---

## ⭐ Acknowledgements

- [HaGRID: HAnd Gesture Recognition Image Dataset](https://github.com/hukenovs/hagrid) for the gesture classes and landmark annotations
- [Google MediaPipe](https://github.com/google-ai-edge/mediapipe) for hand landmark detection
- [scikit-learn](https://scikit-learn.org/), [XGBoost](https://xgboost.ai/) and [PyTorch](https://pytorch.org/)
- [Pygame](https://www.pygame.org/) and [OpenCV](https://opencv.org/)

If you found this project useful, consider giving it a ⭐!
