import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report


print("=" * 60)
print("TITLE-ONLY MODEL TRAINING")
print("=" * 60)


# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

fake = pd.read_csv("data/Fake.csv")
real = pd.read_csv("data/True.csv")


# Add labels
fake["label"] = 0
real["label"] = 1


# Combine datasets
df = pd.concat(
    [fake, real],
    ignore_index=True
)


# --------------------------------------------------
# 2. CLEAN TITLE DATA
# --------------------------------------------------

df = df[["title", "label"]]

df["title"] = df["title"].fillna("")

df = df[df["title"].str.strip() != ""]

df = df.drop_duplicates(
    subset=["title"]
)


print("\nDataset Shape:", df.shape)

print("\nLabel Distribution:")
print(df["label"].value_counts())


# --------------------------------------------------
# 3. SPLIT DATA
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    df["title"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)


# --------------------------------------------------
# 4. TF-IDF
# --------------------------------------------------

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)


X_train_tfidf = tfidf.fit_transform(X_train)

X_test_tfidf = tfidf.transform(X_test)


print("\nTF-IDF Training Shape:")
print(X_train_tfidf.shape)


# --------------------------------------------------
# 5. TRAIN MODEL
# --------------------------------------------------

model = LinearSVC(
    C=1.0,
    class_weight="balanced",
    max_iter=5000
)


model.fit(
    X_train_tfidf,
    y_train
)


# --------------------------------------------------
# 6. EVALUATE
# --------------------------------------------------

predictions = model.predict(
    X_test_tfidf
)


accuracy = accuracy_score(
    y_test,
    predictions
)


print("\n" + "=" * 60)
print("TITLE MODEL RESULTS")
print("=" * 60)


print(
    "\nAccuracy:",
    round(accuracy, 4)
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions
    )
)


# --------------------------------------------------
# 7. SAVE TITLE MODEL
# --------------------------------------------------

joblib.dump(
    model,
    "models/title_model.pkl"
)


joblib.dump(
    tfidf,
    "models/title_tfidf.pkl"
)


print("\nTitle model saved successfully!")