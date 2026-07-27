import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    balanced_accuracy_score
)


print("=" * 70)
print("DOMAIN-ADAPTED TITLE MODEL TRAINING")
print("=" * 70)


# ============================================================
# 1. LOAD ORIGINAL DATA
# ============================================================

fake = pd.read_csv("data/Fake.csv")
real = pd.read_csv("data/True.csv")


fake["label"] = 0
real["label"] = 1


original_data = pd.concat(
    [fake, real],
    ignore_index=True
)


original_data = original_data[
    ["title", "label"]
]


original_data["title"] = (
    original_data["title"]
    .fillna("")
    .astype(str)
)


original_data = original_data[
    original_data["title"].str.strip() != ""
]


original_data = original_data.drop_duplicates(
    subset=["title"]
)


# ============================================================
# 2. LOAD EXTERNAL DATA
# ============================================================

external_data = pd.read_csv(
    "data/external_test.csv"
)


external_data["title"] = (
    external_data["title"]
    .fillna("")
    .astype(str)
)


external_data = external_data[
    external_data["title"].str.strip() != ""
]


external_data = external_data.drop_duplicates(
    subset=["title"]
)


print("\nOriginal dataset:", original_data.shape)
print("External dataset:", external_data.shape)


# ============================================================
# 3. SPLIT EXTERNAL DATA FIRST
# ============================================================

external_train, external_test = train_test_split(
    external_data,
    test_size=0.20,
    random_state=42,
    stratify=external_data["label"]
)


print("\nExternal training data:", external_train.shape)
print("External test data:", external_test.shape)


# ============================================================
# 4. BALANCE ORIGINAL DATA
# ============================================================

# Use approximately the same number of original samples
# as external training samples.

original_sample_size = min(
    len(original_data),
    len(external_train)
)


original_sample = original_data.sample(
    n=original_sample_size,
    random_state=42
)


# ============================================================
# 5. COMBINE TRAINING DATA
# ============================================================

combined_train = pd.concat(
    [
        original_sample,
        external_train
    ],
    ignore_index=True
)


combined_train = combined_train.sample(
    frac=1,
    random_state=42
)


print("\nCombined training data:")
print(combined_train.shape)

print("\nCombined label distribution:")
print(combined_train["label"].value_counts())


# ============================================================
# 6. TF-IDF
# ============================================================

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=100000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)


X_train = tfidf.fit_transform(
    combined_train["title"]
)


X_external_test = tfidf.transform(
    external_test["title"]
)


print("\nTF-IDF shape:")
print(X_train.shape)


# ============================================================
# 7. TRAIN MODEL
# ============================================================

model = LinearSVC(
    C=1.0,
    class_weight="balanced",
    max_iter=10000
)


model.fit(
    X_train,
    combined_train["label"]
)


# ============================================================
# 8. EVALUATE ON UNSEEN EXTERNAL DATA
# ============================================================

predictions = model.predict(
    X_external_test
)


accuracy = accuracy_score(
    external_test["label"],
    predictions
)


balanced_accuracy = balanced_accuracy_score(
    external_test["label"],
    predictions
)


print("\n" + "=" * 70)
print("DOMAIN-ADAPTED MODEL RESULTS")
print("=" * 70)


print(
    f"\nAccuracy: {accuracy:.4f}"
)


print(
    f"Balanced Accuracy: {balanced_accuracy:.4f}"
)


print("\nClassification Report:")


print(
    classification_report(
        external_test["label"],
        predictions,
        target_names=[
            "Fake News",
            "Real News"
        ]
    )
)


# ============================================================
# 9. SAVE MODEL
# ============================================================

joblib.dump(
    model,
    "models/domain_adapted_title_model.pkl"
)


joblib.dump(
    tfidf,
    "models/domain_adapted_title_tfidf.pkl"
)


print(
    "\nDomain-adapted model saved successfully!"
)