import requests
import tqdm
import csv
import os
import re
import json

# Third party
from dotenv import load_dotenv
from datasets import load_dataset


def get_api_response(sampled_tweets: list, key: str, sentiment: str):
    prompt_start = (
        f"Create synthetic {sentiment.upper()} tweets about the financial "
        "market or crypto currencies. Examples:"
    )

    # Convert sampled_tweets to a JSON format with indices
    indexed_tweets = {str(index): tweet for index,
                      tweet in enumerate(sampled_tweets)}
    sampled_tweets_json = json.dumps(
        indexed_tweets, indent=2, ensure_ascii=False)

    return requests.post(
        "https://api.together.xyz/inference",
        json={
            "model": "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
            "max_tokens": 512,
            "prompt": "",
            "request_type": "language-model-inference",
            "temperature": 0.7,
            "top_p": 1,
            "top_k": 100,
            "repetition_penalty": 1.0,
            "stop": ["<|im_end|>", "<|im_start|>"],
            "messages": [
                {
                    "content": f"{prompt_start}\n{sampled_tweets_json}\nDo "
                    "not explain your answer. Only answer according to the "
                    "shown valid JSON format.",
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


def parse_tweets(data_string):
    try:
        # Extract JSON object from the string
        json_match = re.search(r'\{.*\}', data_string, re.DOTALL)
        if json_match:
            json_string = json_match.group()

            # Try parsing as standard JSON first
            parsed_json = json.loads(json_string)
            if isinstance(parsed_json, dict):
                return list(parsed_json.values())
        else:
            return None
    except json.JSONDecodeError:
        # Handle non-standard JSON formats
        try:
            # Handle non-standard JSON formats
            # Process each line to handle non-standard formats
            formatted_lines = []
            for line in json_string.split('\n'):
                # Add quotes around values that are not enclosed in quotes
                line = re.sub(r'(\d+):\s*([^",\n]+)', r'\1: "\2"', line)

                # Escape quotes within string values
                def replace_func(match):
                    key, value = match.groups()
                    value = value.replace('\\', '\\\\').replace('"', '\\"')
                    return '{}: "{}"'.format(key, value)
                line = re.sub(r'(\d+):\s*"(.+?)"', replace_func, line)

                # Ensure the line ends with a comma
                if not line.endswith(','):
                    line += ','

                formatted_lines.append(line)

            # Join the lines and remove the last comma
            formatted_string = '\n'.join(formatted_lines).rstrip(',\n')

            # Attempt to parse again
            parsed_json = json.loads(f'{{{formatted_string}}}')
            if isinstance(parsed_json, dict):
                return list(parsed_json.values())
        except json.JSONDecodeError:
            # If all parsing fails, return None
            return None
        except Exception as e:
            print(f"Error: {e}")
            return None

    return None


if __name__ == "__main__":
    SENTIMENT = "negative"
    N_REQUESTS = 3000

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
              newline="", encoding="utf-8") as f_raw, \
            open(f"failed-{SENTIMENT}-tweets.csv", "a",
                 newline="", encoding="utf-8") as f_failed:
        writer_raw = csv.writer(f_raw)
        writer_failed = csv.writer(f_failed)

        for i in tqdm.tqdm(range(N_REQUESTS)):
            try:
                df_sample = dataframe[dataframe["sentiment"]
                                      == sentiment_to_label[SENTIMENT]].sample(10)
                res = get_api_response(
                    df_sample["tweet"].tolist(), key=key, sentiment=SENTIMENT)
                generated_data = res.json()["output"]["choices"][0]["text"]

                # Parse the tweets
                tweets = parse_tweets(generated_data)

                if tweets is not None:
                    for tweet in tweets:
                        writer_raw.writerow([tweet])
                else:
                    writer_failed.writerow([generated_data])

            except requests.exceptions.JSONDecodeError:
                print("Empty JSON. Skipping...")
                continue

            except Exception as e:
                print(f"Error: {e}")
                continue

    # TODO: Clean the tweets
    """
    # Clean the tweets and write them to a new CSV file
    with open(f"clean-{SENTIMENT}-tweets.csv", "w",
              newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        with open(f"raw-{SENTIMENT}-tweets.csv", "r",
                  newline="", encoding="utf-8") as f2:
            reader = csv.reader(f2)

            for row in reader:
                tweet = clean_tweet(row[0])

                if tweet is not None:
                    writer.writerow([tweet])
    """
