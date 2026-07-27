import pandas as pd

from sklearn.model_selection import train_test_split

from src.data_loader import load_data
from src.preprocessing import preprocess_data


# Load data
df = load_data()


# Preprocess data
df = preprocess_data(df)


# Create the same train-test split used during training
X_train, X_test, y_train, y_test = train_test_split(
    df["content"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)


# Create test dataset
test_data = pd.DataFrame({
    "content": X_test,
    "actual_label": y_test
})


# Convert numerical labels to readable names
test_data["actual_label"] = test_data["actual_label"].map({
    0: "Fake News",
    1: "Real News"
})


# Select 20 articles
test_data = test_data.sample(
    n=20,
    random_state=42
)


# Save test dataset
test_data.to_csv(
    "data/test_news.csv",
    index=False
)


print("Test data created successfully!")
print("\nTest data shape:", test_data.shape)
print("\nSample test data:")
print(test_data.head())