import requests
import tqdm
import csv
import os
import re

# Third party
from dotenv import load_dotenv
from datasets import load_dataset


def get_api_response(sampled_tweets: list, key: str, sentiment: str):
    prompt_start = (
        f"Create synthetic {sentiment.upper()} tweets about the financial "
        "market. Examples:"
    )
    sampled_tweets = "\n".join(sampled_tweets)

    return requests.post(
        "https://api.together.xyz/inference",
        json={
            "model": "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
            "max_tokens": 512,
            "prompt": "",
            "request_type": "language-model-inference",
            "temperature": 1.05,
            "top_p": 1,
            "top_k": 100,
            "repetition_penalty": 1.3,
            "stop": ["<|im_end|>", "<|im_start|>"],
            "messages": [
                {
                    "content": f"{prompt_start}\n{sampled_tweets}",
                    "role": "user",
                },
            ],
            "prompt_format_string":
            "<|im_start|>user\n {prompt}\n<|im_end|>\n<|im_start|>assistant\n",
            "repetitive_penalty": 1.3,
        },
        headers={
            "Authorization": f"Bearer {key}",
        },
    )


def clean_tweet(tweet):
    if not isinstance(tweet, str):
        return None

    # Remove "⃣" character
    tweet = tweet.replace('⃣', '')

    # Remove quotes within the tweet
    tweet = tweet.replace('"', '')

    # Remove enumeration like "1.", "1/", "1)", "2)", etc., "1:", "2:", and
    # standalone numbers
    tweet = re.sub(r'^\d+[\./):]?\s*', '', tweet)

    # Remove itemization like "-" at the beginning of the tweet
    tweet = re.sub(r'^-\s*', '', tweet)

    # Remove tweets containing only hashtags, cashtags or @users
    if re.fullmatch(r'(\s*[@$#]\w+\s*)+', tweet):
        return None

    # Remove tweets with the words "sentiment", "tweets", or
    # "synthetic"
    if any(word in tweet.lower() for word in
           ["sentiment", "tweets", "synthetic"]):
        return None

    # Strip all text
    tweet = tweet.strip()

    # Remove tweets with less than 3 words
    if len(tweet.split()) < 3:
        return None

    return tweet


if __name__ == "__main__":
    SENTIMENT = "negative"
    N_REQUESTS = 100

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

    # Open the CSV file. The 'a' mode appends to the file without truncating
    # it.
    with open(f"raw-{SENTIMENT}-tweets.csv", "a",
              newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        for i in tqdm.tqdm(range(N_REQUESTS)):
            try:
                # Random sample of 10 tweets, with the specified sentiment
                df_sample = dataframe[dataframe["sentiment"]
                                      == sentiment_to_label[SENTIMENT]
                                      ].sample(10)
                res = get_api_response(df_sample["tweet"].tolist(),
                                       key=key, sentiment=SENTIMENT)
                generated = res.json()[
                    "output"]["choices"][0]["text"].split("\n")

                for tweet in generated:
                    writer.writerow([tweet])

            except requests.exceptions.JSONDecodeError:
                print("Empty JSON. Skipping...")
                continue

    # Clean the tweets and write them to a new CSV file
    with open(f"clean-{SENTIMENT}-tweets.csv", "a",
              newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        with open(f"raw-{SENTIMENT}-tweets.csv", "r",
                  newline="", encoding="utf-8") as f2:
            reader = csv.reader(f2)

            for row in reader:
                tweet = clean_tweet(row[0])

                if tweet is not None:
                    writer.writerow([tweet])
