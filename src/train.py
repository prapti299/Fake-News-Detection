import joblib

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB

from sklearn.svm import LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report
)

from src.data_loader import load_data

from src.preprocessing import preprocess_data


print("=" * 60)
print("FAKE NEWS DETECTION - MODEL TRAINING")
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


print("Final Dataset Shape:", df.shape)


# --------------------------------------------------
# STEP 3: SEPARATE FEATURES AND TARGET
# --------------------------------------------------

X = df["content"]

y = df["label"]


# --------------------------------------------------
# STEP 4: TRAIN-TEST SPLIT
# --------------------------------------------------

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42,

    stratify=y

)


print("Training Samples:", len(X_train))

print("Testing Samples:", len(X_test))


# --------------------------------------------------
# STEP 5: TF-IDF VECTORIZATION
# --------------------------------------------------

print("\nApplying TF-IDF Vectorization...")


tfidf = TfidfVectorizer(

    stop_words="english",

    max_features=50000,

    ngram_range=(1, 2)

)


X_train_tfidf = tfidf.fit_transform(X_train)

X_test_tfidf = tfidf.transform(X_test)


print(

    "TF-IDF Training Shape:",

    X_train_tfidf.shape

)


# --------------------------------------------------
# STEP 6: DEFINE MODELS
# --------------------------------------------------

models = {

    "Logistic Regression":

        LogisticRegression(

            max_iter=1000

        ),


    "Naive Bayes":

        MultinomialNB(),


    "Linear SVM":

        LinearSVC(),


    "Random Forest":

        RandomForestClassifier(

            n_estimators=100,

            random_state=42,

            n_jobs=-1

        )

}


# --------------------------------------------------
# STEP 7: TRAIN MODELS
# --------------------------------------------------

results = {}


best_model = None

best_accuracy = 0

best_model_name = ""


for name, model in models.items():


    print("\n" + "=" * 60)

    print("Training:", name)

    print("=" * 60)


    # Train model

    model.fit(

        X_train_tfidf,

        y_train

    )


    # Make predictions

    y_pred = model.predict(

        X_test_tfidf

    )


    # Calculate accuracy

    accuracy = accuracy_score(

        y_test,

        y_pred

    )


    results[name] = accuracy


    print(

        "Accuracy:",

        round(accuracy, 4)

    )


    print("\nClassification Report:")


    print(

        classification_report(

            y_test,

            y_pred

        )

    )


    # Check best model

    if accuracy > best_accuracy:


        best_accuracy = accuracy

        best_model = model

        best_model_name = name


# --------------------------------------------------
# STEP 8: SAVE BEST MODEL
# --------------------------------------------------

print("\n" + "=" * 60)

print("BEST MODEL")

print("=" * 60)


print("Model:", best_model_name)

print("Accuracy:", best_accuracy)


print("\nSaving model...")


joblib.dump(

    best_model,

    "models/fake_news_model.pkl"

)


joblib.dump(

    tfidf,

    "models/tfidf_vectorizer.pkl"

)


print("\nModel saved successfully!")

print("Training completed!")