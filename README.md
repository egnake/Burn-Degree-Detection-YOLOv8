# 🔥 BurnSentinel: AI Burn Degree Detection System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-magenta)](https://github.com/ultralytics/ultralytics)
[![Platform](https://img.shields.io/badge/Platform-CUDA%20%7C%20Windows-orange)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()

> **An advanced Computer Vision project designed to detect, classify, and analyze skin burn injuries in real-time using Deep Learning.**

## 📖 Overview

**BurnSentinel** is an AI-powered medical assistance tool that analyzes live video feeds or static images to identify skin burns. Leveraging the **YOLOv8 (You Only Look Once)** architecture, it segments the injury area and classifies it into three medical severity levels: **1st Degree**, **2nd Degree**, and **3rd Degree**.

Beyond classification, the system acts as a "Smart First Aid Assistant," providing immediate, rule-based medical guidance specific to the detected burn type. This project explores the intersection of **Medical AI** and **Edge Computing**, optimized for high-performance inference on NVIDIA GPUs.

---

## 🚀 Key Features

* **⚡ Real-Time Inference:** Capable of processing high-FPS video feeds with low latency.
* **🧠 Multi-Class Detection:**
    * **1st Degree:** Superficial damage (Redness/Erythema).
    * **2nd Degree:** Partial thickness damage (Blistering/Bullae).
    * **3rd Degree:** Full thickness damage (Charring/Eschar).
* **🏥 Medical HUD:** Displays dynamic, color-coded bounding boxes and confidence scores.
* **❄️ Freeze & Analyze Mode:** Allows the user to pause the live feed (using `Spacebar`) to inspect the injury closely and read detailed treatment protocols.
* **🛡️ Robust AI Model:** Trained on a diverse dataset to distinguish burns from healthy skin.

---

## 🛠️ Technology Stack

| Component | Technology |
| :--- | :--- |
| **Core Language** | Python 3.10+ |
| **Neural Network** | Ultralytics YOLOv8 (PyTorch) |
| **Computer Vision** | OpenCV (cv2) |
| **Acceleration** | NVIDIA CUDA (RTX Series Optimized) |

---

## ⚙️ Installation & Setup

Follow these commands step-by-step to set up the project on your local machine.

```bash
# 1. Clone the repository
git clone [https://github.com/egnake/Burn-Degree-Detection-YOLOv8.git](https://github.com/egnake/Burn-Degree-Detection-YOLOv8.git)
cd Burn-Degree-Detection-YOLOv8

# 2. Create and Activate Virtual Environment (Windows)
python -m venv venv
.\venv\Scripts\activate

# 3. Install Standard Dependencies
pip install -r requirements.txt

# 4. Install GPU-Supported PyTorch (Critical for RTX Performance)
# (This uninstalls the CPU version and installs the CUDA version)
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
```
##🖥️ Usage
```bash
python sentinel_ai.py
```
---

## 📂 Repository Structure

```text
Burn-Degree-Detection/
│
├── dataset/             # Images and Label files (Excluded from repo)
├── runs/                # Training outputs and weights (Excluded from repo)
│
├── sentinel_ai.py       # Main Application (Inference & UI Logic)
├── train_model.py       # Training Script for Transfer Learning
├── config.yaml          # YOLO Dataset Configuration
├── requirements.txt     # Project Dependencies
└── README.md            # Documentation
```
## Medical Disclaimer
## ⚠️ CRITICAL WARNING
```text
This software is developed for educational and experimental purposes only. It utilizes artificial intelligence algorithms that are probabilistic and can make errors.

This is NOT a certified medical device.

It should NOT be used as a substitute for professional medical diagnosis or treatment.

In case of a medical emergency, always contact emergency services immediately.
```

##📜 License
```text
This project is open-source and available under the MIT License.
```
