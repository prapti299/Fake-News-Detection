from src.data_loader import load_data
from src.preprocessing import preprocess_data


print("=" * 60)
print("FAKE NEWS DETECTION PROJECT")
print("=" * 60)


# Load data
print("\nLoading dataset...")

df = load_data()


print("Dataset loaded successfully!")


print("\nOriginal Dataset Shape:")
print(df.shape)


# Preprocess data
print("\nPreprocessing data...")

df = preprocess_data(df)


print("Preprocessing completed!")


print("\nFinal Dataset Shape:")
print(df.shape)


print("\nFinal Columns:")
print(df.columns)


print("\nLabel Distribution:")
print(df["label"].value_counts())


print("\nSample Cleaned Data:")
print(df.head())