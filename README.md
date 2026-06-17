# 🪐 Aura: Advanced Real-Time Gesture Interface

A premium, edge-to-edge static hand gesture recognition system utilizing **MediaPipe Hands** for spatial coordinate tracking and a production-optimized **Scikit-Learn MLP Neural Network** for real-time classification. 

The core engine tracks 21 landmark configurations across **18 distinct gesture classes**, evaluating inputs against a trained model optimized from **180,000 localized feature frames** running fluidly at 60 FPS.

---

## 📈 Engineering Evolution & Architectural Breakthroughs

The development lifecycle of this project progressed through a rigorous optimization pipeline, transitioning from a localized, translation-dependent machine learning prototype into a production-grade, high-definition desktop interface.

### Phase 1: Algorithmic Architecture Optimization & Benchmarking
The initial phase focused on building a highly accurate machine learning environment. After conducting structural benchmarking across multiple architectures—including Random Forest, Support Vector Machines (SVM), XGBoost, and PyTorch Deep Networks—a **Multi-Layer Perceptron (MLP) Classifier** was engineered as the champion model, achieving a peak validation accuracy of **91.27%** over 18 gesture classes.

### Phase 2: Resolving Real-Time Inference Obstacles (Systemic Engineering)
Transitioning the trained model weights into a live, real-time video stream exposed critical edge cases. The following multi-tiered engineering sprints were systematically implemented to resolve these runtime bottlenecks:

1. **Mathematical Transformation for Scale and Translation Invariance:**
   During initial live testing, spatial dependencies caused the model to default to a singular gesture class (`two_up`) when the hand shifted orientation. The root cause was a coordinate system mismatch: the training data was tightly bound, but the live pipeline fed raw, absolute screen positions ($0.0$ to $1.0$). 
   * *The Engineering Fix:* Designed and injected a precise spatial transformation directly into the real-time pipeline. By zero-centering all 21 landmarks relative to the wrist origin and scaling the coordinates via the maximum Euclidean distance vector, the system became entirely **translation-invariant** and **scale-invariant**:
     $$\text{Features}_i = \frac{\mathbf{x}_i - \mathbf{x}_{\text{wrist}}}{\text{max}(\|\mathbf{x} - \mathbf{x}_{\text{wrist}}\|)}$$

2. **Temporal Prediction Smoothing (Dampening Micro-Jitter):**
   Even with high static accuracy, frame-by-frame coordinate twitching from the camera sensor caused rapid flickering in the user interface.
   * *The Engineering Fix:* Implemented a temporal filtering mechanism utilizing a rolling `collections.deque` window (Size: 12 frames) coupled with a `Counter` majority-voting filter. By displaying only the statistical mode of recent predictions, anomalous flickering was completely eliminated, producing a stabilized UI output.

3. **Advanced Pygame UX Framework Overhaul:**
   Standard computer vision display libraries severely limited user interface capabilities, rendering basic, pixelated text and harsh block overlays.
   * *The Engineering Fix:* Completely bypassed legacy display pipelines and built a hardware-accelerated presentation window from the ground up in **Pygame**. This layout natively supports anti-aliased typography, smooth canvas scaling, and elegant translucent **Glassmorphism container panels** that float over the camera feed.

4. **Hardware-Level Video Optimization:**
   Low-resolution laptop webcams introduced severe motion blur and artifacting, dropping tracking precision during rapid gestures.
   * *The Engineering Fix:* Integrated a high-definition external camera module via an **Iriun USB-C hardware layer driver**, running seamlessly through the Windows Microsoft Media Foundation (`cv2.CAP_MSMF`) backend at a native screen-matched resolution ($1280 \times 720$).

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

## 🛠️ Tech Stack & Core Dependencies

* **Core Language:** Python 3.11
* **Computer Vision & Spatial Tracking:** MediaPipe Hands API, OpenCV (via Microsoft Media Foundation `cv2.CAP_MSMF`)
* **Machine Learning Framework:** Scikit-Learn (Multi-Layer Perceptron Network Architecture), Joblib
* **Graphics & User Interface Engine:** Pygame (Hardware-Accelerated Canvas Rendering)
* **Mathematical Operations:** NumPy, Math

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
   git clone https://github.com/Praffulsoni/hand-gesture-recognition.git
2. Activate your virtual environment and install the required dependencies:
    ```bash
    pip install -r requirements.txt
3. Ensure your phone camera link (like Iriun) is active over USB-C, and run the production engine:
    ```bash
    python test_camera.py
Press Esc at any time to smoothly terminate the fullscreen application.
