import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


print("=" * 60)
print("TITLE-ONLY EXTERNAL EVALUATION")
print("=" * 60)


# Load external dataset
external_data = pd.read_csv(
    "data/external_test.csv"
)


# Load title model
model = joblib.load(
    "models/title_model.pkl"
)


# Load title TF-IDF
tfidf = joblib.load(
    "models/title_tfidf.pkl"
)


# Remove missing titles
external_data["title"] = (
    external_data["title"]
    .fillna("")
)


# Transform titles
X_external = tfidf.transform(
    external_data["title"]
)


# Predict
predictions = model.predict(
    X_external
)


# Actual labels
actual_labels = external_data["label"]


# Accuracy
accuracy = accuracy_score(
    actual_labels,
    predictions
)


print(
    "\nExternal Accuracy:",
    round(accuracy, 4)
)


print("\nClassification Report:")

print(
    classification_report(
        actual_labels,
        predictions,
        target_names=[
            "Fake News",
            "Real News"
        ]
    )
)


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        actual_labels,
        predictions
    )
)