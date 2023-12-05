# FinTwitBERT

FinTwitBERT is a language model specifically pre-trained on a large dataset of financial tweets. This specialized BERT model aims to capture the unique jargon and communication style found in the financial Twitter sphere, making it an ideal tool for sentiment analysis, trend prediction, and other financial NLP tasks.

## Table of Contents
- [Dataset](#dataset)
- [Model Details](#model-details)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

## Datasets
### Pre-training
FinTwitBERT is pre-trained on Taborda et al.'s [Stock Market Tweets Data](https://ieee-dataport.org/open-access/stock-market-tweets-data) consisting of 943,672 tweets, including 1,300 labeled tweets. All labeled tweets are used for evaluation of the pre-trained model, using perplexity as a measurement. The other tweets are used for pre-training with 10% being used for model validation.

For pre-training on cryptocurrency tweets we used [Financial Tweets - Cryptocurrency](https://huggingface.co/datasets/StephanAkkerman/financial-tweets-crypto).

### Finetuning
The following datasets are used for finetuning.

| Dataset | Bearish | Neutral | Bullish | Total |
|---------|---------|---------|---------|-------|
| [Datasat1.csv](https://www.kaggle.com/yash612/stockmarket-sentiment-dataset) | 2,106 | 0 | 3,685 | 5,791 |
| [Dataset2.csv](https://www.kaggle.com/chenxidong/stock-tweet-sentiment-dataset) | 2,598 | 17,330 | 8,512 | 29,440 |
| [Dataset3.csv](https://www.kaggle.com/utkarshxy/stock-markettweets-lexicon-data) | 348 | 424 | 528 | 1,300 |
| [Dataset4.csv](https://www.kaggle.com/mattgilgo/stock-related-tweet-sentiment) | 869 | 905 | 8,302 | 10,076 |
| [Dataset5.csv](https://github.com/surge-ai/stock-sentiment/blob/main/sentiment.csv) | 173 | 0 | 327 | 500 |
| [Dataset6.csv](https://github.com/poojathakoor/twitter-stock-sentiment/tree/master/twitter_stock_sentiment/training_data) | 600 | 500 | 707 | 1,807 |
| [Dataset7.csv](https://github.com/surge-ai/crypto-sentiment/blob/main/sentiment.csv) | 260 | 0 | 302 | 562 |
| [Dataset8.csv](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment) | 1789 | 7744 | 2398 | 11931 |
| [Dataset9.csv](https://ieee-dataport.org/open-access/stock-market-tweets-data) | 348 | 424 | 528 | 1300 |
| Total (no duplicates) | 6,647 | 17,352 | 18,280 | 42,279 |

## Model details
We use the [FinBERT](https://huggingface.co/ProsusAI/finbert) model and tokenizer from ProsusAI as our base. We added two masks to the tokenizer: `@USER` for user mentions and `[URL]` for URLs in tweets. The model is then pre-trained for 10 epochs using loss at the metric for the best model. We apply early stopping to prevent overfitting the model.

The latest pre-trained model and tokenizer can be found here on huggingface: https://huggingface.co/StephanAkkerman/FinTwitBERT.

## Installation
```bash
# Clone this repository
git clone https://github.com/TimKoornstra/FinTwitBERT
# Install required packages
pip install -r requirements.txt
```
## Usage
The model can be finetuned for specific tasks such as sentiment classification. For more information about it, you can visit our other repository: https://github.com/TimKoornstra/stock-sentiment-classifier.

## Contributing
Contributions are welcome! If you have a feature request, bug report, or proposal for code refactoring, please feel free to open an issue on GitHub. I appreciate your help in improving this project.

## License
This project is licensed under the GPL-3.0 License. See the [LICENSE](LICENSE) file for details.
