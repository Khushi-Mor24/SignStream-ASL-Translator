import os
import time
from collections import deque, Counter

import cv2
import mediapipe as mp
import numpy as np
import joblib
import pandas as pd
import pyttsx3
import speech_recognition as sr
from spellchecker import SpellChecker

# ========= PATHS =========
MODEL_PATH = os.path.join("models", "asl_xgboost.pkl")
LABELS_PATH = os.path.join("data", "labels.csv")
SIGN_DIR = os.path.join("assets", "signs")  # A.jpg ... Z.jpg, space.jpg

# ========= LOAD MODEL & LABELS =========
model = joblib.load(MODEL_PATH)
labels = pd.read_csv(LABELS_PATH)
id2name = dict(zip(labels["class_id"], labels["sign_name"]))

# ========= MEDIAPIPE =========
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ========= SPELL CHECKER =========
spell = SpellChecker()

# ========= TTS ENGINE =========
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 170)  # speaking speed


def speak(text: str):
    """Speak text using pyttsx3 (offline)."""
    if not text:
        return
    tts_engine.say(text)
    tts_engine.runAndWait()


# ========= LANDMARK EXTRACTION =========
def extract_landmarks_from_frame(frame, hands):
    """
    Takes a BGR frame and MediaPipe Hands instance.
    Returns 63-dim feature vector and results object.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    if not res.multi_hand_landmarks:
        return None, res

    hand = res.multi_hand_landmarks[0]
    xs, ys, zs = [], [], []
    for lm in hand.landmark:
        xs.append(lm.x)
        ys.append(lm.y)
        zs.append(lm.z)

    xs = (np.array(xs) - np.mean(xs)) / (np.std(xs) + 1e-6)
    ys = (np.array(ys) - np.mean(ys)) / (np.std(ys) + 1e-6)
    zs = (np.array(zs) - np.mean(zs)) / (np.std(zs) + 1e-6)

    feat = np.concatenate([xs, ys, zs])
    return feat, res


# ========= SIGN IMAGES =========
def load_sign_image(letter: str):
    """Load the reference ASL image for a letter (A–Z)."""
    if not letter or not letter.isalpha():
        return None

    fname = f"{letter.upper()}.jpg"
    path = os.path.join(SIGN_DIR, fname)

    if not os.path.exists(path):
        return None

    img = cv2.imread(path)
    if img is None:
        return None

    img = cv2.resize(img, (150, 150))
    return img


# ========= VOICE → TEXT =========
def listen_voice_text(timeout=6, phrase_time_limit=4) -> str:
    """Normal voice recognition: jo bhi bolo, uska text return karega."""
    r = sr.Recognizer()
    r.energy_threshold = 350
    r.pause_threshold = 0.6

    try:
        with sr.Microphone() as source:
            print("\n🎙  Speak now... (e.g., 'hello')")
            r.adjust_for_ambient_noise(source, duration=1.2)
            audio = r.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)

        text = r.recognize_google(audio, language="en-IN")
        print("✔ Recognized voice text:", text)
        return text
    except sr.UnknownValueError:
        print("❌ Could not understand audio.")
        return ""
    except sr.WaitTimeoutError:
        print("⏱ No voice detected.")
        return ""
    except Exception as e:
        print("⚠ Voice error:", e)
        return ""


# ========= MAIN REAL-TIME LOOP =========
def main():
    # Text from hand signs
    sentence_text = ""
    current_word = ""

    # ---- NEW: prediction smoothing ----
    prediction_history = deque(maxlen=8)  # last 8 frames
    ready_for_new_letter = True           # naya letter tabhi add jab haath hat chuka ho
    nothing_counter = 0                   # kitne frames se "nothing" aa raha hai

    # Voice-to-sign animation state (same as before)
    voice_anim_text = ""
    voice_anim_index = 0
    voice_anim_last_time = 0.0
    voice_anim_delay = 0.7
    voice_anim_img = None
    voice_anim_clean_text = ""

    print("Using sign images from:", SIGN_DIR)
    if os.path.exists(SIGN_DIR):
        print("Sign files:", os.listdir(SIGN_DIR))
    else:
        print("⚠ SIGN_DIR does not exist!")

    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            H, W, _ = frame.shape

            feat, res = extract_landmarks_from_frame(frame, hands)

            # Draw hand landmarks
            if res and res.multi_hand_landmarks:
                for hand_landmarks in res.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

            # ---------- RAW PREDICTION PER FRAME ----------
            if feat is not None:
                pred = model.predict([feat])[0]
                frame_pred = id2name[int(pred)]
            else:
                frame_pred = "nothing"  # no hand visible

            prediction_history.append(frame_pred)

            # ---------- STABLE PREDICTION (SMOOTHING) ----------
            stable_sign = "nothing"
            if prediction_history:
                most_common, count = Counter(prediction_history).most_common(1)[0]
                # agar 5 ya zyada frames same sign mile hain, usko stable maan lo
                if count >= 5:
                    stable_sign = most_common

            # "nothing" frames ka count track karo
            if stable_sign == "nothing":
                nothing_counter += 1
                # agar kaafi der se nothing hai, next sign ke liye ready ho jao
                if nothing_counter >= 5:
                    ready_for_new_letter = True
            else:
                nothing_counter = 0

            # ---------- WORD / SENTENCE BUILD (ONLY ON STABLE SIGN) ----------
            if stable_sign != "nothing" and ready_for_new_letter:
                # ek sign ka ek hi event
                if stable_sign == "space":
                    if current_word:
                        corrected = spell.correction(current_word.lower())
                        if corrected is None:
                            corrected = current_word

                        if sentence_text:
                            sentence_text += " " + corrected
                        else:
                            sentence_text = corrected

                        print(f"Word finalized: {current_word} -> {corrected}")
                        current_word = ""
                elif stable_sign == "del":
                    if current_word:
                        current_word = current_word[:-1]
                    else:
                        sentence_text = sentence_text[:-1]
                else:
                    # normal alphabet
                    current_word += stable_sign

                # jab tak haath hata nahi, ready_for_new_letter = False
                ready_for_new_letter = False

            # Display text: sentence + current word
            if sentence_text and current_word:
                display_text = sentence_text + " " + current_word
            elif sentence_text:
                display_text = sentence_text
            else:
                display_text = current_word

            # ===== VOICE → SIGN ANIMATION UPDATE =====
            now = time.time()
            if voice_anim_clean_text:
                while (voice_anim_index < len(voice_anim_clean_text) and
                       not (voice_anim_clean_text[voice_anim_index].isalpha() or
                            voice_anim_clean_text[voice_anim_index] == " ")):
                    voice_anim_index += 1
                    voice_anim_last_time = now

                if voice_anim_index < len(voice_anim_clean_text):
                    if now - voice_anim_last_time >= voice_anim_delay:
                        ch = voice_anim_clean_text[voice_anim_index]
                        voice_anim_last_time = now
                        voice_anim_index += 1

                        if ch.isalpha():
                            voice_anim_img = load_sign_image(ch)
                        elif ch == " ":
                            space_path = os.path.join(SIGN_DIR, "space.jpg")
                            if os.path.exists(space_path):
                                img = cv2.imread(space_path)
                                if img is not None:
                                    voice_anim_img = cv2.resize(img, (150, 150))
                            else:
                                voice_anim_img = None
                else:
                    voice_anim_clean_text = ""
                    voice_anim_img = None

            # ===== OVERLAYS =====

            # Bottom-right: voice animation sign image
            if voice_anim_img is not None:
                vh, vw, _ = voice_anim_img.shape
                y2 = H - 10
                y1 = y2 - vh
                x2 = W - 10
                x1 = x2 - vw
                if x1 >= 0 and y1 >= 0:
                    frame[y1:y2, x1:x2] = voice_anim_img

            # Text overlays
            cv2.putText(frame, f"Prediction (raw): {frame_pred}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.putText(frame, f"Stable sign: {stable_sign}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.putText(frame, f"Current word: {current_word}", (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.putText(frame, f"Sentence: {display_text}", (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)

            if voice_anim_text:
                cv2.putText(frame, f"Voice: {voice_anim_text}", (10, 165),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)

            cv2.putText(frame, "Keys: Q=quit  C=clear  S=speak  V=voice→sign",
                        (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow("ASL Realtime: Hand Sign (smoothed) + Voice→Sign Animation", frame)
            key = cv2.waitKey(1) & 0xFF

            # ===== KEYBOARD CONTROLS =====
            if key == ord('q'):
                break
            elif key == ord('c'):
                sentence_text = ""
                current_word = ""
                voice_anim_text = ""
                voice_anim_clean_text = ""
                voice_anim_img = None
            elif key == ord('s'):
                print("Speaking:", display_text)
                speak(display_text)
            elif key == ord('v'):
                text = listen_voice_text()
                if text:
                    voice_anim_text = text
                    voice_anim_clean_text = "".join(
                        [ch.lower() for ch in text if ch.isalpha() or ch == " "]
                    )
                    voice_anim_index = 0
                    voice_anim_last_time = time.time()
                    voice_anim_img = None

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
