import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


TRAIN_FILE = "data/processed/speed_aware_train.csv"
TEST_FILE = "data/processed/speed_aware_robustness.csv"


# Load datasets
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)


# Features used by the ML model
FEATURES = [
    "rms",
    "peak",
    "mean",
    "std",
    "crest_factor",
    "rotational_frequency",
    "amp_1x",
    "amp_2x",
    "amp_3x",
    "ratio_2x_1x",
    "ratio_3x_1x",
]


X_train = train_df[FEATURES]
y_train = train_df["label"]

X_test = test_df[FEATURES]
y_test = test_df["label"]


print("==============================================")
print("VERSION 2: SPEED-AWARE RANDOM FOREST")
print("==============================================")

print()
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print()
print("Training Random Forest...")


model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("Training complete.")

print()
print("Testing on NEW robustness dataset...")


predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)


print()
print("===== VERSION 2 ROBUSTNESS RESULTS =====")
print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy percentage: {accuracy * 100:.2f}%")

print()
print("Classification Report:")

print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "healthy",
            "slight_misalignment",
            "severe_misalignment",
        ],
        zero_division=0
    )
)


print("Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)


print()
print("===== FEATURE IMPORTANCE =====")

importance = pd.Series(
    model.feature_importances_,
    index=FEATURES
).sort_values(ascending=False)

print(importance)


print()
print("===== FINAL COMPARISON =====")

print("Version 1 robustness accuracy: 52.17%")
print(f"Version 2 robustness accuracy: {accuracy * 100:.2f}%")


improvement = (accuracy * 100) - 52.17

print(f"Improvement: {improvement:+.2f} percentage points")


print()
print("Version 2 testing complete.")