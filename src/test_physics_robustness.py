import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


TRAIN_FILE = "data/processed/ml_features.csv"
TEST_FILE = "data/processed/physics_robustness_features.csv"


FEATURES = [
    "rms",
    "peak",
    "mean",
    "std",
    "crest_factor",
    "amp_30hz",
    "amp_60hz",
    "amp_90hz"
]


# Load data
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)


X_train = train_df[FEATURES]
y_train = train_df["label"]

X_test = test_df[FEATURES]
y_test = test_df["label"]


print("==============================================")
print("FINAL PHYSICS ROBUSTNESS TEST")
print("==============================================")

print()
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

print()
print("Training Random Forest...")


model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


model.fit(X_train, y_train)

print("Training complete.")

print()
print("Testing on NEW physics-based robustness data...")


predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)


print()
print("==============================================")
print("FINAL ROBUSTNESS RESULTS")
print("==============================================")

print()
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
            "severe_misalignment"
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
print("==============================================")
print("RESULT COMPARISON")
print("==============================================")

print("Controlled held-out test: 100.00%")
print("Old robustness test:      52.17%")
print(f"Physics robustness test:  {accuracy * 100:.2f}%")

print()
print("Final robustness test complete.")