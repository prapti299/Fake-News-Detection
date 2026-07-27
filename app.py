import streamlit as st
import joblib
import re


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)


# ==================================================
# LOAD MODELS
# ==================================================

# Full article model
full_text_model = joblib.load(
    "models/fake_news_model.pkl"
)

full_text_tfidf = joblib.load(
    "models/tfidf_vectorizer.pkl"
)


# Title-only model
title_model = joblib.load(
    "models/title_model.pkl"
)

title_tfidf = joblib.load(
    "models/title_tfidf.pkl"
)


# ==================================================
# TEXT CLEANING
# ==================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ==================================================
# USER INTERFACE
# ==================================================

st.title("📰 Fake News Detection System")

st.write(
    """
Enter either a news headline or a complete news article.
The system automatically selects the appropriate machine
learning model.
"""
)


news_text = st.text_area(
    "Enter News",
    height=300,
    placeholder=(
        "Example:\n"
        "Headline only OR paste the complete news article..."
    )
)


# ==================================================
# PREDICTION
# ==================================================

if st.button("🔍 Detect News"):

    if not news_text.strip():

        st.warning(
            "Please enter a headline or news article."
        )

    else:

        # Count words
        word_count = len(
            news_text.split()
        )


        # Clean input
        cleaned_text = clean_text(
            news_text
        )


        # --------------------------------------------------
        # TITLE-ONLY MODEL
        # --------------------------------------------------

        if word_count < 30:

            st.info(
                "🔹 Short input detected: Using Title Model"
            )


            text_vectorized = title_tfidf.transform(
                [cleaned_text]
            )


            prediction = title_model.predict(
                text_vectorized
            )


        # --------------------------------------------------
        # FULL-TEXT MODEL
        # --------------------------------------------------

        else:

            st.info(
                "🔹 Full article detected: Using Full-Text Model"
            )


            text_vectorized = full_text_tfidf.transform(
                [cleaned_text]
            )


            prediction = full_text_model.predict(
                text_vectorized
            )


        # --------------------------------------------------
        # RESULT
        # --------------------------------------------------

        if prediction[0] == 0:

            st.error(
                "⚠️ PREDICTION: FAKE NEWS"
            )

        else:

            st.success(
                "✅ PREDICTION: REAL NEWS"
            )


        st.write(
            f"Input word count: {word_count}"
        )