import os
import csv
import tqdm
import requests
import pandas as pd

from datasets import load_dataset

# Local imports
from api_utils import get_api_response
from data_cleaning import parse_tweets


def write_tweets(
    generated_data: str,
    tweets: list,
    writer_raw,
    writer_failed,
):
    if tweets:
        for tweet in tweets:
            writer_raw.writerow([tweet])
    else:
        writer_failed.writerow([generated_data])


def process_tweets_for_sentiment(
    sentiment: str,
    dataframe: pd.DataFrame,
    writer_raw,
    writer_failed,
):
    df_sample = dataframe[
        dataframe["sentiment"] == sentiment_to_label[sentiment]
    ].sample(10)
    try:
        res = get_api_response(df_sample["tweet"].tolist(), sentiment=sentiment)
        generated_data = res.json()["output"]["choices"][0]["text"]
        tweets = parse_tweets(generated_data)
        write_tweets(generated_data, tweets, writer_raw, writer_failed)
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
    except Exception as e:
        print(f"Error processing tweets for {sentiment}: {e}")


if __name__ == "__main__":
    N_REQUESTS = 10
    OUTPUT_DIR = "output/synthetic"

    sentiment_to_label = {"neutral": 0, "positive": 1, "negative": 2}

    dataframe = load_dataset(
        "TimKoornstra/financial-tweets-sentiment",
        split="train",
        cache_dir="data/finetune/",
    ).to_pandas()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for sentiment, label in sentiment_to_label.items():
        print(sentiment)
        # Corrected file paths
        raw_file_path = os.path.join(OUTPUT_DIR, f"raw-{sentiment}-tweets.csv")
        failed_file_path = os.path.join(OUTPUT_DIR, f"failed-{sentiment}-tweets.csv")

        with open(raw_file_path, "a", newline="", encoding="utf-8") as f_raw, open(
            failed_file_path, "a", newline="", encoding="utf-8"
        ) as f_failed:

            for _ in tqdm.tqdm(range(N_REQUESTS)):
                process_tweets_for_sentiment(
                    sentiment, dataframe, csv.writer(f_raw), csv.writer(f_failed)
                )
