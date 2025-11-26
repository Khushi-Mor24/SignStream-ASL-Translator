import cv2
import os
import time
import speech_recognition as sr
import pyttsx3
import string

# Path to sign images (A.jpg, B.jpg, ..., Z.jpg, space.jpg)
SIGN_DIR = os.path.join("assets", "signs")

# TTS
engine = pyttsx3.init()
engine.setProperty("rate", 170)

def speak(text: str):
    engine.say(text)
    engine.runAndWait()

def load_sign_image(ch: str):
    """
    ch: single character, e.g. 'A', 'B', ' '
    returns: BGR image or None
    """
    if ch == " ":
        fname = "space.jpg"  # optional
    else:
        fname = f"{ch.upper()}.jpg"

    path = os.path.join(SIGN_DIR, fname)
    if not os.path.exists(path):
        print(f"[WARN] No sign image for: {ch} -> {path}")
        return None

    img = cv2.imread(path)
    return img

def listen_once(timeout=5, phrase_time_limit=5):
    """
    Use microphone to capture one sentence/prompt.
    Returns recognized text (string) or None.
    """
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            print("🎙 Say a word or sentence for Voice-to-Sign...")
            r.adjust_for_ambient_noise(source, duration=1.0)
            audio = r.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)

        text = r.recognize_google(audio, language="en-IN")
        print("You said:", text)
        return text
    except Exception as e:
        print("Voice recognition failed:", e)
        return None

def animate_sign_sequence(text: str, delay_per_char=0.8):
    """
    For each character in text, show corresponding sign image.
    Only A–Z and space are used.
    """
    filtered = ""
    for ch in text.upper():
        if ch in string.ascii_uppercase or ch == " ":
            filtered += ch

    if not filtered:
        print("No valid A–Z characters to show.")
        return

    cv2.namedWindow("Voice-to-Sign Animation", cv2.WINDOW_NORMAL)

    for ch in filtered:
        img = load_sign_image(ch)
        if img is None:
            continue

        # Overlay label
        cv2.putText(img, f"{ch}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

        cv2.imshow("Voice-to-Sign Animation", img)
        key = cv2.waitKey(int(delay_per_char * 1000)) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

def main():
    while True:
        print("\n=== Voice-to-Sign Mode ===")
        print("1. Use microphone")
        print("2. Type text manually")
        print("q. Quit")
        choice = input("Choose: ").strip().lower()

        if choice == "1":
            text = listen_once()
            if text:
                print("Animating:", text)
                speak(f"You said: {text}. Now showing ASL signs.")
                animate_sign_sequence(text)
        elif choice == "2":
            text = input("Enter a word or sentence: ").strip()
            if text:
                print("Animating:", text)
                animate_sign_sequence(text)
        elif choice == "q":
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()
