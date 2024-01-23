import requests
import tqdm
import csv
import os

# Third party
from dotenv import load_dotenv
from datasets import load_dataset


def get_api_response(sampled_tweets: list):
    prompt_start = (
        "Create synthetic BEARISH tweets about the financial market. Examples:"
    )
    sampled_tweets = sampled_tweets.join("\n")

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
            "prompt_format_string": "<|im_start|>user\n {prompt}\n<|im_end|>\n<|im_start|>assistant\n",
            "repetitive_penalty": 1.3,
        },
        headers={
            "Authorization": f"Bearer {key}",
        },
    )


if __name__ == "__main__":
    load_dotenv()
    key = os.getenv("TOGETHER_API")

    dataset = load_dataset(
        "TimKoornstra/financial-tweets-sentiment",
        split="train",
        cache_dir="data/finetune/",
        # download_mode="force_redownload",
    )
    dataframe = dataset.to_pandas()

    # Open the CSV file. The 'a' mode appends to the file without truncating it.
    with open("bearish-tweets.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        for i in tqdm.tqdm(range(1000)):
            # Random sample of 10 tweets, where the label = 2 (bearish)
            df_sample = dataframe[dataframe["sentiment"] == 2].sample(10)
            res = get_api_response(df_sample["tweet"].tolist())
            generated = res.json()["output"]["choices"][0]["text"].split("\n")

            for tweet in generated:
                writer.writerow([tweet])
