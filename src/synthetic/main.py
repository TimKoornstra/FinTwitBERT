import csv
import tqdm
import requests
import pandas as pd
from pathlib import Path

from datasets import load_dataset

# Local imports
from api_utils import get_api_response
from data_cleaning import parse_tweets


def write_tweets(
    generated_data: str, tweets: list, sentiment: str, writer_raw, writer_failed
):
    if tweets:
        for tweet in tweets:
            writer_raw.writerow(
                [tweet, sentiment_to_label[sentiment]]
            )  # Include sentiment in each row
    else:
        writer_failed.writerow(
            [generated_data, sentiment_to_label[sentiment]]
        )  # Include sentiment for failed cases


def process_tweets_for_sentiment(
    sentiment: str, dataframe: pd.DataFrame, writer_raw, writer_failed
):
    df_sample = dataframe[
        dataframe["sentiment"] == sentiment_to_label[sentiment]
    ].sample(10)
    try:
        res = get_api_response(df_sample["tweet"].tolist(), sentiment=sentiment)
        generated_data = res.json()["output"]["choices"][0]["text"]
        tweets = parse_tweets(generated_data)
        write_tweets(generated_data, tweets, sentiment, writer_raw, writer_failed)
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
    except Exception as e:
        print(f"Error processing tweets for {sentiment}: {e}")


if __name__ == "__main__":
    N_REQUESTS = 10
    OUTPUT_DIR = Path("output/synthetic")

    sentiment_to_label = {"neutral": 0, "positive": 1, "negative": 2}

    dataframe = load_dataset(
        "TimKoornstra/financial-tweets-sentiment",
        split="train",
        cache_dir="data/finetune/",
    ).to_pandas()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_file_path = OUTPUT_DIR / "raw-tweets.csv"
    failed_file_path = OUTPUT_DIR / "failed-tweets.csv"

    while True:
        with open(raw_file_path, "a", newline="", encoding="utf-8") as f_raw, open(
            failed_file_path, "a", newline="", encoding="utf-8"
        ) as f_failed:
            writer_raw = csv.writer(f_raw)
            writer_failed = csv.writer(f_failed)

            # Optionally write headers if the files are new/empty
            if raw_file_path.stat().st_size == 0:
                writer_raw.writerow(["Tweet", "Sentiment"])
            if failed_file_path.stat().st_size == 0:
                writer_failed.writerow(["Generated Data", "Sentiment"])

            for sentiment in sentiment_to_label.keys():
                print(sentiment)
                # Read amount of sentiment tweets in the csv file
                tweets_df = pd.read_csv(raw_file_path)
                n_sentiment_tweets = tweets_df[
                    tweets_df["sentiment"] == sentiment_to_label[sentiment]
                ].shape[0]
                print(n_sentiment_tweets)
                if n_sentiment_tweets > 500_000:
                    print(f"Skipping {sentiment} as it has enough tweets")
                    continue
                for _ in tqdm.tqdm(range(N_REQUESTS)):
                    process_tweets_for_sentiment(
                        sentiment, dataframe, writer_raw, writer_failed
                    )
