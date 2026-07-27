import pandas as pd


def load_data():

    # Load Fake News
    fake_news = pd.read_csv(
        "data/Fake.csv"
    )


    # Load Real News
    real_news = pd.read_csv(
        "data/True.csv"
    )


    # Add labels
    # 0 = Fake News
    # 1 = Real News

    fake_news["label"] = 0

    real_news["label"] = 1


    # Combine datasets
    df = pd.concat(
        [
            fake_news,
            real_news
        ],
        ignore_index=True
    )


    return df