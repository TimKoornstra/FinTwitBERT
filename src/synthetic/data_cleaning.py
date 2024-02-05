import re

import json_repair
import numpy as np

# Compile regular expressions outside the function for efficiency
emoji_only_pattern = re.compile(r"^[\U00010000-\U0010ffff]+$")
rocket_pattern = re.compile(r"(🚀){4,}")
unwanted_words_pattern = re.compile(r"sentiment|tweets|synthetic", re.IGNORECASE)
unwanted_chars_pattern = re.compile(r"⃣|\"")
enum_and_itemization_pattern = re.compile(r"^\d+[\./):]?\s*|^-\s*")
hashtag_cashtag_user_pattern = re.compile(r"(\s*[@$#]\w+\s*)+")
start_itemization_pattern = re.compile(r"^(?:\d+[\./):]?\s*|-\s*)")


def replace_rockets(tweet: str) -> str:
    if rocket_pattern.search(tweet):
        # Generate a random number between 1 and 6 (inclusive) for each match
        return rocket_pattern.sub(lambda match: str(np.random.randint(1, 7)), tweet)
    return tweet


def clean_tweet(tweet) -> str or None:
    if not isinstance(tweet, str):
        return None

    tweet = start_itemization_pattern.sub("", tweet)
    tweet = replace_rockets(tweet)
    tweet = unwanted_chars_pattern.sub("", tweet)
    tweet = enum_and_itemization_pattern.sub("", tweet)

    if (
        unwanted_words_pattern.search(tweet)
        or emoji_only_pattern.fullmatch(tweet)
        or hashtag_cashtag_user_pattern.fullmatch(tweet)
    ):
        return None

    # Remove leading and trailing whitespace
    tweet = tweet.strip()

    if len(tweet.split()) < 3:
        return None

    return tweet


def parse_tweets(data_string: str) -> list:
    # Attempt to directly parse the JSON string
    parsed_json = json_repair.loads(data_string)

    if isinstance(parsed_json, dict):
        # Clean the tweets once and filter out None values in one step
        cleaned_tweets = [clean_tweet(tweet) for tweet in parsed_json.values()]
        return [tweet for tweet in cleaned_tweets if tweet is not None]

    return []
