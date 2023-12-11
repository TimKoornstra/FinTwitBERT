# > Imports
# Standard library
import os
import html
import re

# Third party
from datasets import Dataset, load_dataset
import pandas as pd
from sklearn.model_selection import KFold


def preprocess_tweet(tweet: str) -> str:
    """
    Preprocess a tweet by replacing URLs, @mentions, and cash tags with tokens.

    Parameters
    ----------
    tweet : str
        The tweet to preprocess.

    Returns
    -------
    str
        The preprocessed tweet.
    """
    # Unescape HTML characters
    tweet = html.unescape(tweet)

    # Replace URLs wiht URL token
    tweet = re.sub(r"http\S+", "[URL]", tweet)

    # Replace @mentions with @USER token
    tweet = re.sub(r"@\S+", "@USER", tweet)

    # Replace cash tags with [TICKER] token
    # tweet = re.sub(r"\$[A-Z]{1,5}\b", "[TICKER]", tweet)

    return tweet


def load_local_dataset(path: str, is_test: bool) -> pd.DataFrame:
    """
    Load a dataset from a CSV file given as path.

    Parameters
    ----------
    path : str
        The path to the CSV file.
    is_test : bool
        If the dataset is a test dataset.

    Returns
    -------
    pd.DataFrame
        The dataset as a pandas DataFrame.
    """
    dataset = pd.read_csv(path, encoding="utf-8", on_bad_lines="warn")

    # Rename columns
    columns = {"full_text": "text"}
    if is_test:
        columns = {"Text": "text", "Sentiment": "label"}
    dataset = dataset.rename(columns=columns)

    return dataset[list(columns.values())]


def load_fintwit_datasets() -> list:
    """
    Loads all fintwit datasets from the data/pretrain folder.

    Returns
    -------
    list
        The datasets as a list of tuples (path, dataset).
    """
    # Rename columns
    columns = {"tweet_text": "text"}
    datasets = []

    # Read all files in the data/pretrain folder
    for path in os.listdir("data/pretrain"):
        # Read files starting with fintwit
        if path.startswith("fintwit"):
            dataset = pd.read_csv(f"data/{path}", encoding="utf-8", on_bad_lines="warn")
            dataset = dataset.rename(columns=columns)
            dataset = dataset[list(columns.values())]

            # Drop rows where text is NaN
            dataset = dataset.dropna(subset=["text"])

            datasets.append((path, dataset))

    return datasets


def preprocess_fintwit_dataset():
    """
    Preprocesses all fintwit datasets and saves them to data/pretrain/preprocessed.
    """
    datasets = load_fintwit_datasets()
    for path, dataset in datasets:
        dataset["text"] = dataset["text"].apply(preprocess_tweet)
        dataset = dataset.drop_duplicates(subset=["text"])
        dataset.to_csv(f"data/pretrain/preprocessed/{path}", index=False)


def save_preprocessed_dataset(path: str):
    """
    Loads a dataset from a CSV file given as path, preprocesses it and saves it to data/pretrain/preprocessed.

    Parameters
    ----------
    path : str
        The path to the CSV file.
    """
    is_test = False
    if path.endswith("test.csv"):
        is_test = True

    dataset = load_local_dataset(path, is_test=is_test)
    dataset["text"] = dataset["text"].apply(preprocess_tweet)

    # Drop duplicates
    dataset = dataset.drop_duplicates(subset=["text"])

    # Save preprocessed dataset
    os.makedirs("data/pretrain/preprocessed", exist_ok=True)
    dataset.to_csv(f"data/pretrain/preprocessed/{path.split('/')[-1]}", index=False)


def load_pretrain() -> pd.DataFrame:
    """
    Loads all the pretraining datasets from the data/pretrain/preprocessed folder.
    Excluding the test dataset.

    Returns
    -------
    pd.DataFrame
        The complete pretraining dataset as a pandas DataFrame.
    """
    datasets = []
    # Read all files in the data/pretrain/preprocessed folder
    for path in os.listdir("data/pretrain/preprocessed"):
        if path != "test.csv":
            dataset = pd.read_csv(f"data/pretrain/preprocessed/{path}")
            datasets.append(dataset)

    # Merge datasets
    dataset = pd.concat(datasets, ignore_index=True)

    # Drop duplicates
    return dataset.drop_duplicates(subset=["text"])


def load_pretraining_data(val_size: float = 0.1) -> tuple:
    """
    Loads the pretraining data and splits it into a training and validation set.

    Parameters
    ----------
    val_size : float, optional
        The size of the validation set, by default 0.1

    Returns
    -------
    tuple
        The training and validation datasets.
    """
    dataset = load_pretrain()

    # Randomly sample 10% of the data for validation, set the random state for reproducibility
    validation_set = dataset.sample(frac=val_size, random_state=42)

    # Drop the validation set from the original dataset to create the training set
    training_set = dataset.drop(validation_set.index)

    # Convert the pandas DataFrames into Hugging Face Datasets
    training_dataset = Dataset.from_pandas(training_set)
    validation_dataset = Dataset.from_pandas(validation_set)

    return training_dataset, validation_dataset


def load_finetuning_data(val_size: float = 0.1) -> tuple:
    """
    Loads and preprocesses the finetuning data and splits it into a training and validation set.

    Parameters
    ----------
    val_size : float, optional
        The size of the validation set, by default 0.1

    Returns
    -------
    tuple
        The training and validation datasets.
    """

    # https://huggingface.co/datasets/TimKoornstra/financial-tweets-sentiment
    dataset = load_dataset(
        "TimKoornstra/financial-tweets-sentiment",
        split="train",
        cache_dir="data/finetune/",
    )

    # Convert to pandas
    dataset = dataset.to_pandas()

    # Preprocess tweets
    dataset["tweet"] = dataset["tweet"].apply(preprocess_tweet)

    # Rename columns
    dataset = dataset.rename(columns={"tweet": "text", "sentiment": "label"})

    # Set labels to int
    dataset["label"] = dataset["label"].astype(int)

    # Randomly sample 10% of the data for validation, set the random state for reproducibility
    validation_set = dataset.sample(frac=val_size, random_state=42)

    # Drop the validation set from the original dataset to create the training set
    training_set = dataset.drop(validation_set.index)

    # Convert the pandas DataFrames into Hugging Face Datasets
    training_dataset = Dataset.from_pandas(training_set)
    validation_dataset = Dataset.from_pandas(validation_set)

    return training_dataset, validation_dataset


def kfold_pretraining_data(k: int = 5) -> tuple:
    """
    Split the pretraining data into k folds.

    Parameters
    ----------
    k : int, optional
        The number of folds, by default 5

    Returns
    -------
    tuple
        The training and validation datasets.
    """
    training_datasets = []
    validation_datasets = []

    df = load_pretrain()

    # Assuming 'dataset' is a list or array of your data
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # This will yield indices for 5 splits
    for train_index, val_index in kf.split(df):
        training_datasets.append(Dataset.from_pandas(df.iloc[train_index]))
        validation_datasets.append(Dataset.from_pandas(df.iloc[val_index]))

    return training_datasets, validation_datasets


def load_test_data() -> Dataset:
    """
    Loads the pretraining test dataset.

    Returns
    -------
    Dataset
        The test dataset.
    """
    test = pd.read_csv("data/pretrain/preprocessed/test.csv")
    return Dataset.from_pandas(test)
