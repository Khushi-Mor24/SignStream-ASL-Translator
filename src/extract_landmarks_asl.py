import mediapipe as mp
import cv2
import os
import numpy as np

RAW_DIR = "data/raw"
OUT_DIR = "data/processed"
FEATURES_PATH = os.path.join(OUT_DIR, "X.npy")
LABELS_PATH = os.path.join(OUT_DIR, "y.npy")

mp_hands = mp.solutions.hands

def extract_landmarks_from_image(img):
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    ) as hands:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            return None

        hand = results.multi_hand_landmarks[0]

        xs, ys, zs = [], [], []
        for lm in hand.landmark:
            xs.append(lm.x)
            ys.append(lm.y)
            zs.append(lm.z)

        xs = (np.array(xs) - np.mean(xs)) / (np.std(xs) + 1e-6)
        ys = (np.array(ys) - np.mean(ys)) / (np.std(ys) + 1e-6)
        zs = (np.array(zs) - np.mean(zs)) / (np.std(zs) + 1e-6)

        return np.concatenate([xs, ys, zs])  # 21 * 3 = 63 features

def main():
    X = []
    y = []

    os.makedirs(OUT_DIR, exist_ok=True)

    for class_id in sorted(os.listdir(RAW_DIR), key=lambda x: int(x)):
        class_path = os.path.join(RAW_DIR, class_id)
        if not os.path.isdir(class_path):
            continue

        label = int(class_id)
        print(f"Extracting class {label} from {class_path}")

        for file_name in os.listdir(class_path):
            img_path = os.path.join(class_path, file_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            feat = extract_landmarks_from_image(img)
            if feat is not None:
                X.append(feat)
                y.append(label)

    X = np.array(X)
    y = np.array(y)

    print("Final feature shape:", X.shape)
    print("Final labels shape:", y.shape)

    np.save(FEATURES_PATH, X)
    np.save(LABELS_PATH, y)

    print("Saved features to", FEATURES_PATH)
    print("Saved labels to", LABELS_PATH)

if __name__ == "__main__":
    main()
