# > Imports
# Standard library
import html
import random
import re
from collections import Counter

# Third party
from datasets import Dataset, load_dataset, concatenate_datasets
import pandas as pd
import numpy as np


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

    # Replace URLs with URL token
    tweet = re.sub(r"http\S+", "[URL]", tweet)

    # Replace @mentions with @USER token
    tweet = re.sub(r"@\S+", "@USER", tweet)

    # Replace cash tags with [TICKER] token
    # tweet = re.sub(r"\$[A-Z]{1,5}\b", "[TICKER]", tweet)

    return tweet


def load_pretraining_data(val_size: float = 0.1) -> tuple:
    """
    Loads all the pretraining datasets from the data/pretrain/preprocessed folder.
    Excluding the test dataset.

    Returns
    -------
    pd.DataFrame
        The complete pretraining dataset as a pandas DataFrame.
    """
    dataset = load_dataset(
        "StephanAkkerman/crypto-stock-tweets",
        split="train",
        cache_dir="data/pretrain/",
    )

    dataframe = preprocess_dataset(dataset)
    training_dataset, validation_dataset = split_dataframe(dataframe, val_size=val_size)
    return training_dataset, validation_dataset


def load_synthetic_data() -> Dataset:
    """
    Loads all the synthetic datasets from the data/synthetic/preprocessed folder.
    Excluding the test dataset.

    """
    dataset = load_dataset(
        "TimKoornstra/synthetic-financial-tweets-sentiment",
        split="train",
        cache_dir="data/synthetic/",
        ignore_verifications=True,
        # download_mode="force_redownload",
    )

    # Rename columns
    dataset = dataset.rename_column("tweet", "text")
    dataset = dataset.rename_column("sentiment", "label")

    dataframe = preprocess_dataset(dataset)
    return Dataset.from_pandas(dataframe)


def load_finetuning_data(val_size: float = 0.2) -> tuple:
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
        # download_mode="force_redownload",
    )

    # Rename columns
    dataset = dataset.rename_column("tweet", "text")
    dataset = dataset.rename_column("sentiment", "label")

    dataframe = preprocess_dataset(dataset)
    training_dataset, validation_dataset = split_dataframe(dataframe, val_size=val_size)
    return training_dataset, validation_dataset


def simple_oversample(dataset: Dataset) -> pd.DataFrame:
    from imblearn.over_sampling import RandomOverSampler

    dataframe = dataset.to_pandas()

    # Extract texts and labels
    texts = dataframe["text"].tolist()
    labels = dataframe["label"].tolist()

    # Define RandomOverSampler
    ros = RandomOverSampler(random_state=0)

    # Resample indices and labels
    resampled_indices, resampled_labels = ros.fit_resample(
        np.array(list(range(len(texts)))).reshape(-1, 1), labels
    )

    # Extract resampled texts based on the resampled indices
    resampled_texts = [texts[i[0]] for i in resampled_indices]

    # Convert back to dataframe
    dataframe = pd.DataFrame({"text": resampled_texts, "label": resampled_labels})

    # To dataset
    return Dataset.from_pandas(dataframe)


def synonym_oversample(dataset: Dataset) -> pd.DataFrame:
    import nlpaug.augmenter.word as naw

    dataframe = dataset.to_pandas()

    # Extract texts and labels
    texts = [example["text"] for example in dataset]
    labels = [example["label"] for example in dataset]

    # Determine class frequencies and find the maximum frequency
    class_counts = Counter(labels)
    max_count = max(class_counts.values())

    # Initialize nlpaug synonym augmenter
    augmenter = naw.SynonymAug(aug_src="wordnet")

    # Augment data for each class to match the count of the majority class
    augmented_texts = []
    augmented_labels = []
    for label in class_counts.keys():
        class_texts = [text for text, lbl in zip(texts, labels) if lbl == label]
        augment_count = max_count - class_counts[label]

        # Randomly choose and augment texts from the class until reaching the desired count
        while augment_count > 0:
            for text in random.choices(
                class_texts, k=min(augment_count, len(class_texts))
            ):
                augmented_text = augmenter.augment(text)
                augmented_texts.append(augmented_text[0])
                augmented_labels.append(label)
                augment_count -= 1

    # Combine original and augmented data
    resampled_texts = texts + augmented_texts
    resampled_labels = labels + augmented_labels

    # Convert back to dataframe
    dataframe = pd.DataFrame({"text": resampled_texts, "label": resampled_labels})

    # To dataset
    return Dataset.from_pandas(dataframe)


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
    dataframe = dataframe.dropna(subset=["text"])

    # Drop 1 word tweets
    dataframe = dataframe[dataframe["text"].apply(lambda x: len(x.split()) > 1)]

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
    from sklearn.model_selection import KFold

    training_datasets = []
    validation_datasets = []

    df = load_pretraining_data(val_size=0)[0].to_pandas()

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
    return load_finetuning_data()[0]
