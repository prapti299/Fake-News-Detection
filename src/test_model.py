import pandas as pd
import joblib

from sklearn.metrics import accuracy_score


# Load test data
test_data = pd.read_csv(
    "data/test_news.csv"
)


# Load trained model
model = joblib.load(
    "models/fake_news_model.pkl"
)


# Load TF-IDF vectorizer
tfidf = joblib.load(
    "models/tfidf_vectorizer.pkl"
)


# Convert actual labels to numbers
actual_labels = test_data["actual_label"].map({
    "Fake News": 0,
    "Real News": 1
})


# Transform article text
X_test = tfidf.transform(
    test_data["content"]
)


# Make predictions
predictions = model.predict(
    X_test
)


# Convert predictions to readable names
predicted_labels = [
    "Fake News" if prediction == 0
    else "Real News"
    for prediction in predictions
]


# Display results
results = pd.DataFrame({

    "Article": test_data["content"].str[:100],

    "Actual": test_data["actual_label"],

    "Predicted": predicted_labels

})


print("\n" + "=" * 80)
print("MODEL TEST RESULTS")
print("=" * 80)

print(
    results.to_string(index=False)
)


# Calculate accuracy
accuracy = accuracy_score(
    actual_labels,
    predictions
)


print("\n" + "=" * 80)

print(
    f"Test Accuracy: {accuracy:.4f}"
)

print("=" * 80)