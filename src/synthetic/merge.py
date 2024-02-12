import pandas as pd
import os


def merge_with_main(save_as_csv=False):
    # Download from: https://huggingface.co/datasets/TimKoornstra/synthetic-financial-tweets-sentiment
    main = pd.read_parquet("train-00000-of-00001.parquet", engine="pyarrow")

    tweets = pd.read_csv("raw-tweets.csv")

    # Change the column sentiment values to match the main dataframe
    tweets["sentiment"] = tweets["sentiment"].map(
        {"bearish": 2, "bullish": 1, "neutral": 0}
    )

    # Append the tweets dataframe to the main dataframe
    df = pd.concat([main, tweets], ignore_index=True)

    # Set "tweet" column to string type
    df["tweet"] = df["tweet"].astype(str)

    # Save the dataframe to a new parquet file
    df.to_parquet("main-with-tweets.parquet", engine="pyarrow")

    if save_as_csv:
        df.to_csv("main-with-tweets.csv", index=False)


def clean_main():
    main = pd.read_parquet(
        "output/synthetic/train-00000-of-00001.parquet", engine="pyarrow"
    )
    # Count duplicates
    print("Found duplicate rows:", main.duplicated().sum())

    # Print example of duplicate rows
    print(main[main.duplicated()])

    # Remove duplicates
    main = main.drop_duplicates()

    # Print the number of rows with missing values
    print("Number of rows with missing values:", main.isna().sum().sum())

    # Remove rows with empty tweets
    main = main[main["tweet"].notna()]

    # Remove rows with empty sentiment
    main = main[main["sentiment"].notna()]

    # Rename the old file
    os.rename(
        "output/synthetic/train-00000-of-00001.parquet",
        "output/synthetic/train-00000-of-00001_old.parquet",
    )

    # Write the cleaned dataframe to a new parquet file
    main.to_parquet("output/synthetic/train-00000-of-00001.parquet", engine="pyarrow")


def csv_to_parquet(csv_name: str = "output/synthetic/raw-tweets.csv"):
    tweets = pd.read_csv(csv_name)
    tweets.to_parquet("output/synthetic/train-00000-of-00001.parquet", engine="pyarrow")


def main_stats():
    main = pd.read_parquet(
        "output/synthetic/train-00000-of-00001.parquet", engine="pyarrow"
    )
    print(main["sentiment"].value_counts())
    print(len(main))
