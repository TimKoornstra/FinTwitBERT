# > Imports
# Standard library
import os
import html
import re

# Third party
from datasets import Dataset
import pandas as pd


def preprocess_tweet(tweet: str) -> str:
    # Unescape HTML characters
    tweet = html.unescape(tweet)

    # Replace URLs wiht URL token
    tweet = re.sub(r"http\S+", "[URL]", tweet)

    # Replace @mentions with @USER token
    tweet = re.sub(r"@\S+", "@USER", tweet)

    return tweet


def load_dataset(path: str) -> pd.DataFrame:
    dataset = pd.read_csv(path, encoding="utf-8", on_bad_lines="warn")
    dataset = dataset[["full_text"]]
    dataset = dataset.rename(columns={"full_text": "text"})
    return dataset


def load_validation(path: str) -> pd.DataFrame:
    dataset = pd.read_csv(path, encoding="utf-8", on_bad_lines="warn")
    dataset = dataset.rename(columns={"Text": "text", "Sentiment": "label"})
    return dataset


def preprocess_dataset(dataset: str) -> Dataset:
    dataset["text"] = dataset["text"].apply(preprocess_tweet)

    # Return a HuggingFace Dataset
    return Dataset.from_pandas(dataset)


def save_preprocessed_dataset(path: str):
    dataset = load_dataset(path)
    dataset["text"] = dataset["text"].apply(preprocess_tweet)

    # Save preprocessed dataset
    os.makedirs("data/preprocessed", exist_ok=True)
    dataset.to_csv(f"data/preprocessed/{path.split('/')[-1]}", index=False)


def save_preprocessed_validation(path: str = "data/validation.csv"):
    dataset = load_validation(path)
    dataset["text"] = dataset["text"].apply(preprocess_tweet)
    dataset.to_csv(f"data/preprocessed/{path.split('/')[-1]}", index=False)


def load_pretraining_data() -> Dataset:
    # Read both datasets
    tweets1 = pd.read_csv("data/preprocessed/tweets1.csv")
    tweets2 = pd.read_csv("data/preprocessed/tweets2.csv")

    # Merge datasets
    dataset = pd.concat([tweets1, tweets2], ignore_index=True)

    validation = dataset.sample(frac=0.1, random_state=42)

    # Return a HuggingFace Dataset
    return Dataset.from_pandas(dataset), Dataset.from_pandas(validation)


def load_validation_data() -> Dataset:
    validation = pd.read_csv("data/preprocessed/validation.csv")
    return Dataset.from_pandas(validation)
