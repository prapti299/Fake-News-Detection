import pandas as pd


# Load FakeNewsNet files

politifact_fake = pd.read_csv(
    "data/politifact_fake.csv"
)

politifact_real = pd.read_csv(
    "data/politifact_real.csv"
)

gossipcop_fake = pd.read_csv(
    "data/gossipcop_fake.csv"
)

gossipcop_real = pd.read_csv(
    "data/gossipcop_real.csv"
)


# Add labels

politifact_fake["label"] = 0
gossipcop_fake["label"] = 0

politifact_real["label"] = 1
gossipcop_real["label"] = 1


# Combine all external data

external_data = pd.concat(
    [
        politifact_fake,
        politifact_real,
        gossipcop_fake,
        gossipcop_real
    ],
    ignore_index=True
)


# Keep title and label

external_data = external_data[
    [
        "title",
        "label"
    ]
]


# Remove missing titles

external_data = external_data.dropna(
    subset=["title"]
)


# Remove duplicate titles

external_data = external_data.drop_duplicates(
    subset=["title"]
)


# Save external test data

external_data.to_csv(
    "data/external_test.csv",
    index=False
)


print("External test dataset created successfully!")

print(
    "\nDataset shape:",
    external_data.shape
)

print(
    "\nLabel distribution:"
)

print(
    external_data["label"].value_counts()
)