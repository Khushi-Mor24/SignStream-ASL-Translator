import cv2
import mediapipe as mp
import numpy as np
import joblib
import pandas as pd

# Load MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Load trained model
model = joblib.load("models/asl_xgboost.pkl")

# Load labels
labels = pd.read_csv("data/labels.csv")
id2name = dict(zip(labels["class_id"], labels["sign_name"]))

# Word builder variables
current_word = ""
last_letter = ""
cooldown_frames = 0  # prevents repeating the same letter continuously

def extract_landmarks(frame, hands):
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

    return np.concatenate([xs, ys, zs]), res

def main():
    global current_word, last_letter, cooldown_frames

    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)

            if not ret:
                break

            feat, res = extract_landmarks(frame, hands)

            if res and res.multi_hand_landmarks:
                for hand_landmarks in res.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

            prediction_text = ""

            if feat is not None:
                pred = model.predict([feat])[0]
                prediction_text = id2name[int(pred)]

                if cooldown_frames == 0:
                    if prediction_text == "space":
                        current_word += " "
                        cooldown_frames = 20
                    elif prediction_text == "del":
                        current_word = current_word[:-1]
                        cooldown_frames = 20
                    elif prediction_text not in ["nothing"]:
                        current_word += prediction_text
                        cooldown_frames = 20

            if cooldown_frames > 0:
                cooldown_frames -= 1

            # Display prediction
            cv2.putText(frame, f"Prediction: {prediction_text}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display word
            cv2.putText(frame, f"Word: {current_word}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            cv2.imshow("ASL Realtime Recognition (XGBoost)", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
