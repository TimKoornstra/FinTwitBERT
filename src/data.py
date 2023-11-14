# > Imports
# Standard library
import os
import html
import re

# Third party
from datasets import Dataset
import pandas as pd
from sklearn.model_selection import KFold


def preprocess_tweet(tweet: str) -> str:
    # Unescape HTML characters
    tweet = html.unescape(tweet)

    # Replace URLs wiht URL token
    tweet = re.sub(r"http\S+", "[URL]", tweet)

    # Replace @mentions with @USER token
    tweet = re.sub(r"@\S+", "@USER", tweet)

    # Replace cash tags with [TICKER] token
    # tweet = re.sub(r"\$[A-Z]{1,5}\b", "[TICKER]", tweet)

    return tweet


def load_dataset(path: str, is_test: bool) -> pd.DataFrame:
    dataset = pd.read_csv(path, encoding="utf-8", on_bad_lines="warn")

    # Rename columns
    columns = {"full_text": "text"}
    if is_test:
        columns = {"Text": "text", "Sentiment": "label"}
    dataset = dataset.rename(columns=columns)

    return dataset[list(columns.values())]


def save_preprocessed_dataset(path: str):
    is_test = False
    if path.endswith("test.csv"):
        is_test = True

    dataset = load_dataset(path, is_test=is_test)
    dataset["text"] = dataset["text"].apply(preprocess_tweet)

    # Drop duplicates
    dataset = dataset.drop_duplicates(subset=["text"])

    # Save preprocessed dataset
    os.makedirs("data/preprocessed", exist_ok=True)
    dataset.to_csv(f"data/preprocessed/{path.split('/')[-1]}", index=False)


def load_tweets():
    # Read both datasets
    tweets1 = pd.read_csv("data/preprocessed/tweets1.csv")
    tweets2 = pd.read_csv("data/preprocessed/tweets2.csv")

    # Merge datasets
    dataset = pd.concat([tweets1, tweets2], ignore_index=True)

    # Drop duplicates
    return dataset.drop_duplicates(subset=["text"])


def load_pretraining_data(val_size: float = 0.1):
    dataset = load_tweets()

    # Randomly sample 10% of the data for validation, set the random state for reproducibility
    validation_set = dataset.sample(frac=val_size, random_state=42)

    # Drop the validation set from the original dataset to create the training set
    training_set = dataset.drop(validation_set.index)

    # Convert the pandas DataFrames into Hugging Face Datasets
    training_dataset = Dataset.from_pandas(training_set)
    validation_dataset = Dataset.from_pandas(validation_set)

    return training_dataset, validation_dataset


def kfold_pretraining_data(k: int = 5):
    training_datasets = []
    validation_datasets = []

    df = load_tweets()

    # Assuming 'dataset' is a list or array of your data
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # This will yield indices for 5 splits
    for train_index, val_index in kf.split(df):
        training_datasets.append(Dataset.from_pandas(df.iloc[train_index]))
        validation_datasets.append(Dataset.from_pandas(df.iloc[val_index]))

    return training_datasets, validation_datasets


def load_test_data() -> Dataset:
    test = pd.read_csv("data/preprocessed/test.csv")
    return Dataset.from_pandas(test)
