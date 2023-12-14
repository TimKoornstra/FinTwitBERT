# > Imports
# Standard library
import os
import html
import re

# Third party
from datasets import Dataset, load_dataset, concatenate_datasets
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


def load_fintwit_dataset() -> list:
    """
    Loads all fintwit datasets from the data/pretrain folder.

    Returns
    -------
    list
        The datasets as a list of tuples (path, dataset).
    """
    dataset = load_dataset(
        "StephanAkkerman/financial-tweets",
        split="train",
        cache_dir="data/pretrain/",
    )

    # Rename columns
    dataset = dataset.rename_column("tweet", "text")

    # Preprocess tweets
    dataframe = preprocess_dataset(dataset)

    return dataframe


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
    # 0: neutral, 1: bullish, 2: bearish
    dataset = load_dataset(
        "TimKoornstra/financial-tweets-sentiment",
        split="train",
        cache_dir="data/finetune/",
    )

    # Rename columns
    dataset = dataset.rename_column("tweet", "text")
    dataset = dataset.rename_column("sentiment", "label")

    dataframe = preprocess_dataset(dataset)
    training_dataset, validation_dataset = split_dataframe(dataframe, val_size=val_size)
    return training_dataset, validation_dataset


def preprocess_dataset(dataset: Dataset) -> pd.DataFrame:
    # Convert to pandas
    dataframe = dataset.to_pandas()

    # Set labels to int
    if "label" in dataframe.columns:
        dataframe["label"] = dataframe["label"].astype(int)

        # Drop all columns that are not text or label
        dataframe = dataframe[["text", "label"]]
    else:
        dataframe = dataframe[["text"]]

    # Preprocess tweets
    dataframe["text"] = dataframe["text"].apply(preprocess_tweet)

    # Drop duplicates
    dataframe = dataframe.drop_duplicates(subset=["text"])

    # Drop empty text tweets
    dataset = dataset.dropna(subset=["text"])

    return dataframe


def split_dataframe(dataframe: pd.DataFrame, val_size: float = 0.1):
    # Randomly sample 10% of the data for validation, set the random state for reproducibility
    validation_set = dataframe.sample(frac=val_size, random_state=42)

    # Drop the validation set from the original dataset to create the training set
    training_set = dataframe.drop(validation_set.index)

    # Convert the pandas DataFrames into Hugging Face Datasets
    training_dataset = Dataset.from_pandas(training_set)
    validation_dataset = Dataset.from_pandas(validation_set)

    return training_dataset, validation_dataset


def adjust_labels(dataset):
    # Original labels: 0: negative, 1: neutral, 2: positive
    # New labels: 0: neutral, 1: bullish, 2: bearish
    label_mapping = {
        0: 2,  # negative to bearish
        1: 0,  # neutral to neutral
        2: 1,  # positive to bullish
    }
    dataset["label"] = label_mapping[dataset["label"]]
    return dataset


def load_tweet_eval():
    # 0: negative, 1: neutral, 2: positive
    # https://huggingface.co/datasets/tweet_eval/viewer/sentiment
    dataset = load_dataset(
        "tweet_eval",
        cache_dir="data/pre-finetune/",
        name="sentiment",
    )

    # Concatenate the splits into a single dataset
    dataset = concatenate_datasets(
        [dataset["train"], dataset["test"], dataset["validation"]]
    )

    # Change labels to match other datasets
    dataset = dataset.map(adjust_labels)

    dataframe = preprocess_dataset(dataset)
    training_dataset, validation_dataset = split_dataframe(dataframe)
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
