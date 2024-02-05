import requests
import json


def get_api_response(sampled_tweets: list, key: str, sentiment: str):
    prompt_start = (
        f"Create synthetic {sentiment.upper()} tweets about the financial "
        "market or crypto currencies. Examples:"
    )

    # Convert sampled_tweets to a JSON format with indices
    indexed_tweets = {str(index): tweet for index, tweet in enumerate(sampled_tweets)}
    sampled_tweets_json = json.dumps(indexed_tweets, indent=2, ensure_ascii=False)

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
            "prompt_format_string": "<|im_start|>user\n {prompt}\n<|im_end|>\n<|im_start|>assistant\n",
            "repetitive_penalty": 1.3,
        },
        headers={
            "Authorization": f"Bearer {key}",
        },
    )
