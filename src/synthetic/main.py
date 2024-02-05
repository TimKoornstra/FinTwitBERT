import requests
import csv
import os
import tqdm

# Third party
from dotenv import load_dotenv
from datasets import load_dataset

# Local imports
from data_cleaning import parse_tweets
from api_utils import get_api_response

if __name__ == "__main__":
    output_dir = "output/synthetic"
    N_REQUESTS = 10

    sentiment_to_label = {
        "neutral": 0,
        "positive": 1,
        "negative": 2,
    }

    load_dotenv()
    key = os.getenv("TOGETHER_API")

    dataset = load_dataset(
        "TimKoornstra/financial-tweets-sentiment",
        split="train",
        cache_dir="data/finetune/",
        # download_mode="force_redownload",
    )
    dataframe = dataset.to_pandas()

    # Create "synthetic" directory in output
    os.makedirs(output_dir, exist_ok=True)

    # Open the CSV file. The 'a' mode appends to the file without truncating
    # it.
    while True:
        for s, label in sentiment_to_label.items():
            print(s)
            with open(
                f"{output_dir}/raw-{s}-tweets.csv",
                "a",
                newline="",
                encoding="utf-8",
            ) as f_raw, open(
                f"{output_dir}/failed-{s}-tweets.csv",
                "a",
                newline="",
                encoding="utf-8",
            ) as f_failed:
                writer_raw = csv.writer(f_raw)
                writer_failed = csv.writer(f_failed)

                for i in tqdm.tqdm(range(N_REQUESTS)):
                    try:
                        df_sample = dataframe[
                            dataframe["sentiment"] == sentiment_to_label[s]
                        ].sample(10)
                        res = get_api_response(
                            df_sample["tweet"].tolist(), key=key, sentiment=s
                        )
                        generated_data = res.json()["output"]["choices"][0]["text"]

                        # Parse the tweets
                        tweets = parse_tweets(generated_data)

                        if tweets != []:
                            for tweet in tweets:
                                writer_raw.writerow(tweet)
                        else:
                            writer_failed.writerow([generated_data])

                    except requests.exceptions.JSONDecodeError:
                        print("Empty JSON. Skipping...")
                        continue

                    except Exception as e:
                        print(f"Error: {e}")
                        continue
