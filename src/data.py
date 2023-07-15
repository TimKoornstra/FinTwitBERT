# > Imports
# Standard library
import html
import re

# Third party
from datasets import Dataset
import pandas as pd


def preprocess_tweet(tweet):
    # Unescape HTML characters
    tweet = html.unescape(tweet)

    # Replace URLs
    tweet = re.sub(r"http\S+", "", tweet)

    # Replace @mentions with @USER token
    tweet = re.sub(r"@\S+", "@USER", tweet)

    return tweet


def load_dataset(path):
    dataset = pd.read_csv(path, sep=";")
    dataset = dataset[["full_text"]]
    dataset = dataset.rename(columns={"full_text": "text"})
    return dataset


def preprocess_dataset(dataset):
    dataset["text"] = dataset["text"].apply(preprocess_tweet)

    # Return a HuggingFace Dataset
    return Dataset.from_pandas(dataset)
