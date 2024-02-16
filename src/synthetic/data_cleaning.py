import re

import json_repair
import numpy as np

# Compile regular expressions outside the function for efficiency
emoji_only_pattern = re.compile(r"^[\U00010000-\U0010ffff]+$")
unwanted_words_pattern = re.compile(r"sentiment|tweets|synthetic", re.IGNORECASE)
unwanted_words_pattern2 = re.compile(
    r"\b(positive|negative|neutral)\s+tweet\b.*$", re.IGNORECASE
)
unwanted_chars_pattern = re.compile(r"⃣|\"")
enum_and_itemization_pattern = re.compile(r"^\d+[\./):]?\s*|^-\s*")
hashtag_cashtag_user_pattern = re.compile(r"(\s*[@$#]\w+\s*)+")
start_itemization_pattern = re.compile(r"^(?:\d+[\./):]?\s*|-\s*)")
json_pattern = re.compile(r"\{[\s\S]*?\}")
emoji_pattern = re.compile(
    "["
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "\U000024C2-\U0001F251"
    "]+",
)
tag_pattern = re.compile(r"(\$\w+|\#\w+)")


def unique_emojis(match):
    unique = []
    for emoji in match.group(0):
        if len(unique) == 0 or emoji != unique[-1]:
            unique.append(emoji)
    return "".join(unique)


def remove_duplicate_cashtags_and_hashtags(tweet):
    # Find all cashtags and hashtags in the tweet
    tags = tag_pattern.findall(tweet)

    # Remove duplicates by converting the list to a set, then back to a list
    unique_tags = list(set(tags))

    # Sort the unique tags to maintain a consistent order
    unique_tags.sort(key=lambda x: tags.index(x))

    # Remove the original tags in the tweet with the unique tags
    for tag in set(
        tags
    ):  # Iterate through set to avoid processing the same tag multiple times
        if tags.count(tag) > 1:  # Check if this tag appeared more than once
            tweet = tweet.replace(
                tag, "", tags.count(tag) - 1
            )  # Remove duplicates of this tag

    return tweet


def clean_tweet(tweet: str) -> str:
    if not isinstance(tweet, str):
        return None

    tweet = start_itemization_pattern.sub("", tweet)
    tweet = emoji_pattern.sub(unique_emojis, tweet)
    tweet = unwanted_chars_pattern.sub("", tweet)
    tweet = enum_and_itemization_pattern.sub("", tweet)
    tweet = remove_duplicate_cashtags_and_hashtags(tweet)

    if (
        unwanted_words_pattern.search(tweet)
        or unwanted_words_pattern2.search(tweet)
        or emoji_only_pattern.fullmatch(tweet)
        or hashtag_cashtag_user_pattern.fullmatch(tweet)
    ):
        return None

    # Remove leading and trailing whitespace
    tweet = tweet.strip()

    # Remove opening and closing quotation marks
    tweet = tweet.strip('"')

    if len(tweet.split()) < 3:
        return None

    return tweet


def parse_tweets(data_string: str) -> list:
    # Find the part of the string that is JSON
    match = json_pattern.search(data_string)
    if match is None:
        return []
    data_string = match.group(0)

    # Attempt to directly parse the JSON string
    parsed_json = json_repair.loads(data_string)

    if isinstance(parsed_json, dict):
        # Clean the tweets once and filter out None values in one step
        cleaned_tweets = [clean_tweet(tweet) for tweet in parsed_json.values()]
        return [tweet for tweet in cleaned_tweets if tweet is not None]

    return []
