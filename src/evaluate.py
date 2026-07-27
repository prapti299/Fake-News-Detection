import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from src.data_loader import load_data
from src.preprocessing import preprocess_data


print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)


# --------------------------------------------------
# STEP 1: LOAD DATA
# --------------------------------------------------

print("\nLoading data...")

df = load_data()


# --------------------------------------------------
# STEP 2: PREPROCESS DATA
# --------------------------------------------------

print("Preprocessing data...")

df = preprocess_data(df)


# --------------------------------------------------
# STEP 3: SPLIT DATA
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(

    df["content"],

    df["label"],

    test_size=0.2,

    random_state=42,

    stratify=df["label"]

)


# --------------------------------------------------
# STEP 4: LOAD SAVED MODEL
# --------------------------------------------------

print("\nLoading trained model...")

model = joblib.load(
    "models/fake_news_model.pkl"
)


tfidf = joblib.load(
    "models/tfidf_vectorizer.pkl"
)


# --------------------------------------------------
# STEP 5: TRANSFORM TEST DATA
# --------------------------------------------------

print("Transforming test data...")

X_test_tfidf = tfidf.transform(
    X_test
)


# --------------------------------------------------
# STEP 6: PREDICT
# --------------------------------------------------

print("Making predictions...")

y_pred = model.predict(
    X_test_tfidf
)


# --------------------------------------------------
# STEP 7: EVALUATION METRICS
# --------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)


precision = precision_score(
    y_test,
    y_pred
)


recall = recall_score(
    y_test,
    y_pred
)


f1 = f1_score(
    y_test,
    y_pred
)


print("\n" + "=" * 60)

print("EVALUATION RESULTS")

print("=" * 60)


print(
    f"\nAccuracy: {accuracy:.4f}"
)


print(
    f"Precision: {precision:.4f}"
)


print(
    f"Recall: {recall:.4f}"
)


print(
    f"F1 Score: {f1:.4f}"
)


# --------------------------------------------------
# STEP 8: CLASSIFICATION REPORT
# --------------------------------------------------

print("\nClassification Report:")

print(

    classification_report(

        y_test,

        y_pred,

        target_names=[

            "Fake News",

            "Real News"

        ]

    )

)


# --------------------------------------------------
# STEP 9: CONFUSION MATRIX
# --------------------------------------------------

cm = confusion_matrix(

    y_test,

    y_pred

)


plt.figure(

    figsize=(8, 6)

)


sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues",

    xticklabels=[

        "Fake News",

        "Real News"

    ],

    yticklabels=[

        "Fake News",

        "Real News"

    ]

)


plt.title(

    "Fake News Detection - Confusion Matrix"

)


plt.xlabel(

    "Predicted Label"

)


plt.ylabel(

    "Actual Label"

)


plt.tight_layout()


plt.show()