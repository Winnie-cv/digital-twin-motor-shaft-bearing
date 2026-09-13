import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_FILE = PROJECT_ROOT / "data" / "processed" / "ml_features.csv"
TEST_FILE = PROJECT_ROOT / "data" / "processed" / "robustness_features.csv"

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

CLASS_NAMES = [
    "healthy",
    "slight_misalignment",
    "severe_misalignment",
]


print("Loading training data...")
train_df = pd.read_csv(TRAIN_FILE)

print("Loading robustness test data...")
test_df = pd.read_csv(TEST_FILE)

X_train = train_df[FEATURES]
y_train = train_df["label"]

X_test = test_df[FEATURES]
y_test = test_df["label"]

print()
print("Training Random Forest on original dataset...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("Training complete.")
print()
print("Testing on NEW robustness dataset...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print()
print("===== ROBUSTNESS TEST RESULTS =====")
print(f"Test samples: {len(y_test)}")
print(f"Accuracy: {accuracy:.4f}")

print()
print("Classification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=CLASS_NAMES
    )
)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print()
print("True class counts:")
print(test_df["condition"].value_counts())

predicted_conditions = pd.Series(y_pred).map({
    0: "healthy",
    1: "slight_misalignment",
    2: "severe_misalignment"
})

print()
print("Predicted class counts:")
print(predicted_conditions.value_counts())

print()
print("===== COMPARISON =====")
print("Baseline held-out test accuracy: 100%")
print(f"Robustness test accuracy: {accuracy * 100:.2f}%")

print()
print("Robustness testing complete.")