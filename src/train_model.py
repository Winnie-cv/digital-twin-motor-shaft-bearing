import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "ml_features.csv"

FEATURES = [
    "rms",
    "peak",
    "mean",
    "std",
    "crest_factor",
    "amp_30hz",
    "amp_60hz",
    "amp_90hz",
]

def main():

    df = pd.read_csv(DATA_FILE)

    X = df[FEATURES]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print()
    print("===== MACHINE LEARNING RESULTS =====")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")
    print(f"Accuracy: {accuracy:.4f}")
    print()

    print("Classification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "Healthy",
                "Slight Misalignment",
                "Severe Misalignment"
            ]
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print()
    print("Feature Importance:")

    importance = pd.Series(
        model.feature_importances_,
        index=FEATURES
    ).sort_values(ascending=False)

    print(importance)

if __name__ == "__main__":
    main()
