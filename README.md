# ASL Hand Sign Recognition (A–Z) with Voice Control and Speech Output

A full **real‑time ASL translator system** that recognizes **hand signs (A–Z + space + delete + nothing)** using MediaPipe + XGBoost and supports:

✔ **Text‑to‑Speech (system speaks your translated text)**
✔ **Speech‑to‑Text voice commands (clear, speak, stop)**
✔ **Real‑time webcam detection**
✔ **Word builder + space + delete gestures**
✔ **Professional folder structure for evaluation**

This system uses **landmark‑based ML**, not CNN, meaning:

* Extremely fast on CPU
* Very small model
* No TensorFlow required
* Very high accuracy (99.18%)

---

## ⭐ Features

* ASL Alphabet Recognition (A–Z)
* Special signs: `space`, `del`, `nothing`
* Voice Output (TTS using pyttsx3)
* Voice Input Commands (STT using SpeechRecognition)
* Real‑time camera feed with hand‑landmarks
* High‑accuracy XGBoost model

---

## 📁 Project Structure

```
SignStream/
│
├── data/
│   ├── raw/                 # 0–28 folders for A–Z, space, del, nothing
│   ├── processed/           # X.npy, y.npy
│   └── labels.csv           # class mapping
│
├── models/
│   └── asl_xgboost.pkl      # trained model
│
├── src/
│   ├── extract_landmarks_asl.py
│   ├── train_xgboost_asl.py
│   └── realtime_asl_xgboost_voice.py   # FINAL APP
│
├── venv/                     # Python virtual env
│
├── README.md
└── requirements.txt
```

---

## 🎬 How It Works

### **Step 1 — Landmark Extraction**

MediaPipe extracts **21 hand landmarks** (x, y, z) → **63 features**.

### **Step 2 — Train Model**

XGBoost classifier trained on 63,676 samples → **99.18% accuracy**.

### **Step 3 — Real‑Time Recognition**

Camera → Landmarks → Model Prediction → Word Builder.

### **Step 4 — Voice Features**

* **TTS:** System speaks translated text
* **STT:** Voice commands (speak / clear / stop)

---

## ▶️ Run Real‑Time Translator

Activate venv:

```
venv\Scripts\activate
```

Run:

```
python src/realtime_asl_xgboost_voice.py
```

Press **Q** to quit.

---

## 🎮 Controls

### Keyboard

| Key | Action             |
| --- | ------------------ |
| Q   | Quit               |
| C   | Clear text         |
| S   | Speak text         |
| V   | Voice command mode |

### Voice Commands

| Command            | Action              |
| ------------------ | ------------------- |
| "speak" / "read"   | Speaks current text |
| "clear" / "delete" | Clears text         |
| "stop" / "exit"    | Quits program       |

---

## 📈 Model Performance

* Accuracy: **0.9918**
* Dataset: 63,676 samples
* Model: XGBoost, 63‑feature landmark vector



