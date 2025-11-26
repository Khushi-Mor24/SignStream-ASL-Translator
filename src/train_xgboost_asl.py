import os
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Paths to the extracted features
X_PATH = "data/processed/X.npy"
Y_PATH = "data/processed/y.npy"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "asl_xgboost.pkl")

def main():
    # Load features & labels
    X = np.load(X_PATH)
    y = np.load(Y_PATH)

    print("Dataset shapes:", X.shape, y.shape)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Define XGBoost model
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=len(np.unique(y)),
        eval_metric="mlogloss",
        max_depth=8,
        n_estimators=250,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9
    )

    print("Training model...")
    model.fit(X_train, y_train)

    # Evaluate
    print("Evaluating...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f}\n")
    print("Classification report:\n")
    print(classification_report(y_test, y_pred))

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
