import speech_recognition as sr
import cv2
import os
import pyttsx3

# Folder where you have A.jpg, B.jpg, ..., Z.jpg, space.jpg
SIGNS_FOLDER = os.path.join("assets", "signs")

# TTS Engine
engine = pyttsx3.init()
engine.setProperty("rate", 170)


def speak(text: str):
    if not text:
        return
    engine.say(text)
    engine.runAndWait()


def text_to_sign(text: str):
    """Show ASL images for each character in the text."""
    text = text.lower()
    print("\n[INFO] Animating text:", text)

    cv2.namedWindow("ASL Animation", cv2.WINDOW_NORMAL)

    for char in text:
        if char.isalpha():
            img_name = f"{char.upper()}.jpg"
        elif char == " ":
            img_name = "space.jpg"
        else:
            continue  # ignore numbers/punctuation

        img_path = os.path.join(SIGNS_FOLDER, img_name)
        exists = os.path.exists(img_path)
        print(f"  Char '{char}' -> {img_path}  exists={exists}")

        if not exists:
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"  [WARN] Failed to load image: {img_path}")
            continue

        cv2.imshow("ASL Animation", img)
        # show each frame for 800 ms
        key = cv2.waitKey(800) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


def voice_to_text() -> str:
    """Recognize speech using microphone (Google)."""
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            print("\n🎙  Speak now...")
            r.adjust_for_ambient_noise(source, duration=1)
            audio = r.listen(source)

        text = r.recognize_google(audio, language="en-IN")
        print("✔ Recognized voice text:", text)
        return text
    except Exception as e:
        print("❌ Voice recognition error:", e)
        return ""


def main():
    print("SIGNS_FOLDER:", SIGNS_FOLDER)
    print("Files in SIGNS_FOLDER:", os.listdir(SIGNS_FOLDER))

    while True:
        print("\n=== Voice → Sign Menu ===")
        print("1. Use microphone")
        print("2. Type text manually")
        print("q. Quit")
        choice = input("Choose: ").strip().lower()

        if choice == "1":
            text = voice_to_text()
            if text:
                speak("Showing signs for " + text)
                text_to_sign(text)

        elif choice == "2":
            text = input("Enter a word or sentence: ").strip()
            if text:
                text_to_sign(text)

        elif choice == "q":
            break

        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    main()
