import re


def clean_text(text):

    # Convert text to lowercase
    text = text.lower()


    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )


    # Remove HTML tags
    text = re.sub(
        r"<.*?>",
        "",
        text
    )


    # Keep only alphabets and spaces
    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )


    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )


    return text.strip()


def preprocess_data(df):

    # Fill missing titles
    df["title"] = df["title"].fillna("")


    # Fill missing article text
    df["text"] = df["text"].fillna("")


    # Combine title and article text
    df["content"] = (
        df["title"] + " " + df["text"]
    )


    # Clean text
    df["content"] = df["content"].apply(
        clean_text
    )


    # Remove duplicate articles
    df = df.drop_duplicates(
        subset=["content"]
    )


    # Keep only required columns
    df = df[
        [
            "content",
            "label"
        ]
    ]


    return df