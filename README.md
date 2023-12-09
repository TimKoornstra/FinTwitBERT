# FinTwitBERT

![FinTwitBERT Logo](img/logo.png)

---

![Supported versions](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-GPL--3.0-orange)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

FinTwitBERT is a language model specifically trained to understand and analyze financial conversations on Twitter. It's designed to pick up on the unique ways people talk about finance online, making it a valuable tool for anyone interested in financial trends and sentiments expressed through tweets.

## Introduction

Understanding financial markets can be challenging, especially when analyzing the vast amount of opinions and discussions on social media. FinTwitBERT is here to make sense of financial conversations on Twitter. It's a specialized tool that interprets the unique language and abbreviations used in financial tweets, helping users gain insights into market trends and sentiments.

This model was developed to fill a gap in traditional language processing tools, which often struggle with the shorthand and jargon found in financial tweets. Whether you're a financial professional, a market enthusiast, or someone curious about financial trends on social media, FinTwitBERT offers an easy-to-use solution to navigate and understand these discussions.

## Table of Contents
- [Datasets](#datasets)
- [Model Details](#model-details)
- [Model Results](#model-results)
- [Installation](#installation)
- [Usage](#usage)
- [Finetuning](#finetuning-datasets)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)

## Datasets
### Pre-training Datasets
FinTwitBERT utilizes a diverse set of financial tweets for pre-training, including Taborda et al.'s [Stock Market Tweets Data](https://ieee-dataport.org/open-access/stock-market-tweets-data) with over 940K tweets, and our own dataset, [Financial Tweets - Cryptocurrency](https://huggingface.co/datasets/StephanAkkerman/financial-tweets-crypto), with detailed statistics provided below.

#### Our Dataset Statistics:
- Total Tweets: [Number of Tweets]
- Categories: [Details about Categories]
- Data Preprocessing: [Information about preprocessing steps]

### Finetuning Datasets
For finetuning, we use several datasets, each offering varied sentiments in financial contexts:

| Dataset | Bearish | Neutral | Bullish | Total |
|---------|---------|---------|---------|-------|
| [yash612_stock_data.csv](https://www.kaggle.com/yash612/stockmarket-sentiment-dataset) | 2,106 | 0 | 3,685 | 5,791 |
| [mattgilgo_scored_tweets_total.csv](https://www.kaggle.com/mattgilgo/stock-related-tweet-sentiment) | 869 | 905 | 8,302 | 10,076 |
| [surge-ai_sentiment.csv](https://github.com/surge-ai/stock-sentiment/blob/main/sentiment.csv) | 173 | 0 | 327 | 500 |
| [poojathakoor_twitter-stock-sentiment.csv](https://github.com/poojathakoor/twitter-stock-sentiment/tree/master/twitter_stock_sentiment/training_data) | 600 | 500 | 707 | 1,807 |
| [surge-ai_crypto-sentiment.csv](https://github.com/surge-ai/crypto-sentiment/blob/main/sentiment.csv) | 260 | 0 | 302 | 562 |
| [zeroshot_twitter-financial-news-sentiment.csv](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment) | 1789 | 7744 | 2398 | 11931 |
| [Taborda_tweets_labelled.csv](https://ieee-dataport.org/open-access/stock-market-tweets-data) | 348 | 424 | 528 | 1300 |
| [ChanceFocus_fiqa.csv](https://huggingface.co/datasets/ChanceFocus/fiqa-sentiment-classification) | x | x | x | x |
| [moritzwilksch_labeled_tweets.csv](https://github.com/moritzwilksch/MasterThesis/blob/main/data/labeled/labeled_tweets.parquet) | x | x | x | x |
| Total (no duplicates) | 6,647 | 17,352 | 18,280 | 42,279 |

## Model Details
FinTwitBERT is based on [FinBERT](https://huggingface.co/ProsusAI/finbert) with added masks for user mentions (`@USER`) and URLs (`[URL]`). The model is pre-trained for 10 epochs with a focus on minimizing loss and applying early stopping to prevent overfitting.

Find the pre-trained model and tokenizer here: [FinTwitBERT on HuggingFace](https://huggingface.co/StephanAkkerman/FinTwitBERT).

## Model Results
TODO: Compare loss, accuracy, and F1 between FinTwitBERT and other models as a table.

## Installation
```bash
# Clone this repository
git clone https://github.com/TimKoornstra/FinTwitBERT
# Install required packages
pip install -r requirements.txt
```

## Usage
We offer two models [FinTwitBERT](https://huggingface.co/StephanAkkerman/FinTwitBERT) and [FinTwitBERT-sentiment](https://huggingface.co/StephanAkkerman/FinTwitBERT-sentiment). The first is a pre-trained model and tokenizer for masked language modeling (MLM) which can be finetuned for other tasks such as sentiment analysis. This is what the second model is about, it is fine-tuned on sentiment analysis and labels tweets into three categories: bearish, neutral, and bullish.

### Pre-trained model
```python
from transformers import BertForMaskedLM, AutoTokenizer
import torch

# Load pre-trained model and tokenizer
model = BertForMaskedLM.from_pretrained("StephanAkkerman/FinTwitBERT")
tokenizer = AutoTokenizer.from_pretrained("StephanAkkerman/FinTwitBERT")

# Example sentence with a masked token
text = "AAPL is a [MASK] sector stock."

# Tokenize the text
input = tokenizer.encode_plus(text, return_tensors="pt")

# Predict the masked token
model.eval()  # Put the model in evaluation mode
with torch.no_grad():
    outputs = model(**input)
    predictions = outputs.logits

# Get the predicted token
predicted_index = torch.argmax(predictions[0, input["input_ids"][0] == tokenizer.mask_token_id], axis=1)
predicted_token = tokenizer.decode(predicted_index)

# Print the sentence with the predicted token
print(text.replace("[MASK]", predicted_token))
```

### Fine-tuned model
```python
from transformers import BertForMaskedLM, AutoTokenizer, pipeline
import torch

# Load pre-trained model and tokenizer
model = BertForSequenceClassification.from_pretrained("StephanAkkerman/FinTwitBERT-sentiment")
tokenizer = AutoTokenizer.from_pretrained("StephanAkkerman/FinTwitBERT-sentiment")

# Create a pipeline for text classification (sentiment analysis)
sentiment_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Example sentence
sentence = "The new product launch was a tremendous success, boosting sales and customer satisfaction."

# Print the result
print(sentiment_pipeline(sentence))
```

## Citation
If you use FinTwitBERT or FinTwitBERT-sentiment in your research, please cite us as follows, noting that both authors contributed equally to this work:

```bibtex
@misc{FinTwitBERT,
  author = {Stephan Akkerman, Tim Koornstra},
  title = {FinTwitBERT: A Specialized Language Model for Financial Tweets},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/TimKoornstra/FinTwitBERT}}
}
```

## Contributing
Contributions are welcome! If you have a feature request, bug report, or proposal for code refactoring, please feel free to open an issue on GitHub. We appreciate your help in improving this project.

## License
This project is licensed under the GPL-3.0 License. See the [LICENSE](LICENSE) file for details.
