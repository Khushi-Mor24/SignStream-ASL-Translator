# SignStream – Real-Time ASL Translator with Voice-to-Sign Animation  
---

## 📌 Overview

**SignStream** is a real-time **ASL (American Sign Language) alphabet translator** that supports **both directions of communication**:

### 🔵 **1. ASL Hand Sign → Text (Webcam)**
- Detects **A–Z alphabets**, **SPACE**, **DELETE**, **NOTHING**
- MediaPipe → 21 hand landmarks → 63 numeric features
- XGBoost classifier (trained on 63,676 landmark samples)
- Final accuracy: **99.18%**
- Uses prediction smoothing for high stability

### 🔵 **2. Voice → ASL Sign Animation**
- User presses **V** and speaks: *“hello”*
- System converts speech to text
- Displays animated ASL signs:  
  **H → E → L → L → O**
- Shown only in **bottom-right corner** (clean UI)
- Uses custom images from:
  ```
  assets/signs/A.jpg … Z.jpg
  ```

### 🔵 **3. Text-to-Speech (TTS)**
- Press **S** → System speaks the built word/sentence

### 🔵 **4. Word & Sentence Builder**
- Each stable sign is added to a word
- When “space” sign is shown:
  - Word is spell-corrected
  - Added to sentence
- “del” sign removes last letter

---

## 🌟 Key Features

| Feature | Description |
|--------|-------------|
| 🖐 ASL Alphabet Recognition | A–Z + space + delete + nothing |
| 🎙 Voice-to-Sign | Converts spoken text → sign animations |
| 🔊 Text-to-Speech | System speaks recognized text |
| ✏ Word Builder | Auto spell-corrected word creation |
| 📄 Sentence Builder | Multi-word sentence formation |
| 🎥 Real-Time Webcam | 21-point Mediapipe Hand landmarks |
| ⚡ High Accuracy ML | XGBoost classifier @ 99.18% |
| 🪟 Clean UI | Only single webcam + animation window |

---

📘 Dataset Used

This project uses the ASL Alphabet Dataset containing A–Z + Space + Delete + Nothing.

🔗 Dataset Link (Kaggle)

https://www.kaggle.com/datasets/grassknoted/asl-alphabet

Download and extract into:

data/raw/

---

## 📁 Project Structure

```
SignStream/
│
├── assets/
│   └── signs/
│       ├── A.jpg
│       ├── B.jpg
│       ├── ...
│       └── Z.jpg
│
├── data/
│   ├── raw/                 # A–Z, space, del, nothing folders (0–28)
│   ├── processed/
│   │   ├── X.npy
│   │   └── y.npy
│   └── labels.csv
│
├── models/
│   └── asl_xgboost.pkl
│
├── src/
│   ├── extract_landmarks_asl.py
│   ├── train_xgboost_asl.py
│   └── realtime_signstream.py   # FINAL APPLICATION
│
├── venv/
├── README.md
└── requirements.txt
```

---

## 🎬 How the System Works

### **1. Hand Landmark Extraction**
- MediaPipe returns **21 hand keypoints**
- Each has (x, y, z) → **63 numeric features**
- Normalized per frame

### **2. Model Prediction**
- 63-dimensional feature vector fed to XGBoost
- Outputs one class from **29 classes**
- Smoothing applied → single stable prediction

### **3. Building Words**
- New letter added only when hand becomes “nothing” (ready_for_new_letter)

### **4. Voice → Sign Animation**
- SpeechRecognition converts voice to text
- For each character:
  - Image loaded from `assets/signs/<LETTER>.jpg`
  - Displayed in **bottom-right corner**
  - Automatically transitions letter-by-letter

### **5. Text-to-Speech**
- Uses offline engine: `pyttsx3`

---

## ▶ Run Application

Activate environment:
```
venv\Scripts\activate
```

Install dependencies:
```
pip install -r requirements.txt
```

Run:
```
python src/realtime_asl_xgboost_voice.py
```

---

## 🎛 Controls

### 🖥 Keyboard Controls
| Key | Action |
|-----|--------|
| **Q** | Quit program |
| **C** | Clear text |
| **S** | Speak current text |
| **V** | Start voice-to-sign mode |

### 🎙 Voice Commands (inside V mode)
| Command | Action |
|---------|--------|
| **hello / any word** | Convert to sign animation |
| **clear** | Clear text |
| **speak** | Speak text |
| **stop / exit** | Quit voice mode |

---

## 📊 Model Performance

- **Classifier:** XGBoost  
- **Accuracy:** **0.9918 (99.18%)**  
- **Dataset:** 63,676 landmark samples  
- **Classes:** 29 (A–Z + space + del + nothing)  

---

## 🔮 Future Enhancements

- Word-level sign recognition  
- Animated sign GIF support  
- ISL mode (Indian Sign Language)  
- Sign-to-Speech continuous mode  
- Mobile app version (TFLite)

---

## 👩‍💻 Developer
**Khushi Mor**  
B.Tech CSE  
Batch 2023–2027
