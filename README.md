# 🪐 Aura: Advanced Real-Time Gesture Interface

A premium, edge-to-edge static hand gesture recognition system utilizing **MediaPipe Hands** for spatial coordinate tracking and a production-optimized **Scikit-Learn MLP Neural Network** for real-time classification. 

The core engine tracks 21 landmark configurations across **18 distinct gesture classes**, evaluating inputs against a trained model optimized from **180,000 localized feature frames** running fluidly at 60 FPS.

---

## 📈 The Engineering Evolution: From Prototype to Production

This project underwent a radical architecture migration to transition from a localized, translation-dependent machine learning prototype into a production-grade, highly aesthetic desktop interface.

### Phase 1: The ChatGPT Baseline (The Core ML Pipeline)
* **The Starting Point:** The initial implementation established a highly accurate machine learning environment. After rigorous benchmarking across multiple architectures (including Random Forest, SVM, XGBoost, and PyTorch Deep Networks), a **Scikit-Learn Multi-Layer Perceptron (MLP) Classifier** was selected as the champion model, hitting **91.27% accuracy** over 18 gesture classes.
* **The Critical Blocker:** When entering real-time inference via OpenCV, **the classifier collapsed**, predicting the `two_up` gesture class for almost every hand position. 

### Phase 2: The Gemini Takeover (Root Cause Analysis & Engineering Breakthroughs)
Upon migrating the codebase development to Gemini, we systematically diagnosed and resolved the systemic integration issues through a multi-tiered engineering sprint:

1. **The Normalization Mismatch (Data Correction):** We reverse-engineered the raw dataset generation script and discovered that while the training data was tightly bound via custom geometric operations, the live webcam pipeline was feeding un-normalized screen coordinates ($0.0$ to $1.0$). This spatial dependency meant that moving or scaling the hand instantly broke inference.
   * *The Fix:* We ported the precise mathematical transformation directly into the real-time pipeline, zero-centering all points relative to the wrist origin and scaling via the maximum Euclidean distance vector:
     $$\text{Features}_i = \frac{\mathbf{x}_i - \mathbf{x}_{\text{wrist}}}{\text{max}(\|\mathbf{x} - \mathbf{x}_{\text{wrist}}\|)}$$
     This rendered the model entirely **translation-invariant** and **scale-invariant**.

2. **Temporal Prediction Smoothing (Eliminating Jitter):**
   Even with high static accuracy, frame-by-frame coordinate twitching caused the user interface text to flicker rapidly.
   * *The Fix:* Implemented a rolling `collections.deque` queue buffer (Size: 12 frames) coupled with a `Counter` majority-voting filter. By displaying only the statistical mode of recent frames, anomalous flickering was completely dampened, producing a rock-solid UI output.

3. **Bypassing OpenCV GUI Constraints via Pygame (Premium UX Overhaul):**
   OpenCV's primitive, pixelated fonts and harsh block overlays gave the application a basic "utility-tool" appearance.
   * *The Fix:* We completely bypassed OpenCV's display layer and rebuilt the entire front-end window in **Pygame**. This allowed us to support anti-aliased typography, hardware-accelerated rendering, and elegant translucent **Glassmorphism container panels** that float naturally over the camera feed.

4. **Hardware-Level Video Optimization (LinkedIn Demo Readiness):**
   Low-resolution laptop webcams produced blurry silhouettes that degraded visual value for a professional portfolio presentation.
   * *The Fix:* Wired a high-definition iPhone camera module stream directly into Python via an **Iriun USB-C hardware layer driver**, running seamlessly through Windows Microsoft Media Foundation (`cv2.CAP_MSMF`) at a native screen-matched resolution.

---

## 📊 Comprehensive Model Benchmarking Suite

The project repository includes your full comparative experimentation history. The core 42-feature spatial coordinate vector split was evaluated across an 80/20 stratified configuration:

| Model Architecture Script | Test Accuracy | Training Run-Time | Engineering Status |
| :--- | :---: | :---: | :---: |
| Random Forest (`train_rf.py`) | 90.18% | ~69s | Evaluated |
| Support Vector Machine (`train_svm.py`) | 90.88% | ~69s | Evaluated |
| XGBoost Classifier (`train_xgboost.py`) | 90.93% | ~69s | Evaluated |
| PyTorch Deep Network (`train_pytorch.py`) | 91.13% | ~269s | Evaluated |
| **Sklearn MLP Neural Network (`train_mlp.py`)** | **91.27%** | **~75s** | **🏆 Chosen Champion** |

> **Feature Engineering Note:** Additional iterations (`feature_engineering_v2` through `v4`) attempted to inject hand-crafted parameters like explicit fingertip flex angles and spread distances. However, benchmarking proved that clean spatial landmark normalization provided the cleanest mathematical boundaries, setting the original 42 features as our production standard.

---

## 📂 Core Workspace Architecture

The repository is organized into distinct segments representing the lifecycle of the project:

* **Production Application Engine**
  * `test_camera.py`: The live production script coordinating native fullscreen Pygame canvas processing, temporal prediction smoothing filtering, and external HD camera matrix interface handshakes.
* **Model Training & Architecture Validation**
  * `train_mlp.py`: Multi-Layer Perceptron neural network trainer that generates our production-optimized weights asset (`champion_model.pkl`).
  * `train_rf.py` / `train_svm.py` / `train_xgboost.py` / `train_pytorch.py`: Full evaluation scripts used during the benchmarking phase.
* **Data Engineering & Hardware Infrastructure**
  * `create_dataset.py`: Coordinates raw annotated feature arrays into structured training sets.
  * `cuda_check.py` / `gpu_check.py`: Validation modules ensuring absolute utilization for underlying NVIDIA CUDA hardware acceleration layers.

---

## 🛠️ Local Installation & Deployment

1. Clone the repository to your workspace:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPO.git](https://github.com/YOUR_USERNAME/YOUR_REPO.git)
   cd YOUR_REPO
2. Activate your virtual environment and install the required dependencies:
    ```bash
    pip install opencv-python mediapipe joblib pygame numpy scikit-learn
3. Ensure your phone camera link (like Iriun) is active over USB-C, and run the production engine:
    ```bash
    python test_camera.py
    Press Esc at any time to smoothly terminate the fullscreen application.