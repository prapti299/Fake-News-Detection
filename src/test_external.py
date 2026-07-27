import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# Load external data

external_data = pd.read_csv(
    "data/external_test.csv"
)


# Load trained model

model = joblib.load(
    "models/fake_news_model.pkl"
)


# Load TF-IDF vectorizer

tfidf = joblib.load(
    "models/tfidf_vectorizer.pkl"
)


# Extract titles

X_external = external_data["title"]


y_actual = external_data["label"]


# Transform external text

X_external_tfidf = tfidf.transform(
    X_external
)


# Make predictions

y_predicted = model.predict(
    X_external_tfidf
)


# Calculate accuracy

accuracy = accuracy_score(
    y_actual,
    y_predicted
)


print("=" * 60)

print("EXTERNAL DATASET EVALUATION")

print("=" * 60)


print(
    "\nExternal Accuracy:",
    round(accuracy, 4)
)


print("\nClassification Report:")


print(
    classification_report(
        y_actual,
        y_predicted,
        target_names=[
            "Fake News",
            "Real News"
        ]
    )
)


print("\nConfusion Matrix:")


print(
    confusion_matrix(
        y_actual,
        y_predicted
    )
)