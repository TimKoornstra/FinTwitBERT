# FinTwitBERT

FinTwitBERT is a specialized language model pre-trained on a vast dataset of financial tweets. By leveraging the unique jargon and communication styles prevalent in the financial Twitter sphere, this model excels in sentiment analysis, trend prediction, and other financial NLP tasks.

## Table of Contents
- [Introduction](#introduction)
- [Datasets](#datasets)
- [Model Details](#model-details)
- [Installation](#installation)
- [Usage](#usage)
- [Finetuning](#finetuning-datasets)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)

## Introduction
FinTwitBERT enhances traditional BERT models by focusing on the nuances of financial communication on Twitter. This specialization allows for more accurate interpretations and predictions based on financial tweets, which is crucial for market analysis and sentiment understanding.

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
| [Dataset1.csv](https://www.kaggle.com/yash612/stockmarket-sentiment-dataset) | 2,106 | 0 | 3,685 | 5,791 |
| [Dataset2.csv](https://www.kaggle.com/chenxidong/stock-tweet-sentiment-dataset) | 2,598 | 17,330 | 8,512 | 29,440 |
| [Dataset3.csv](https://www.kaggle.com/utkarshxy/stock-markettweets-lexicon-data) | 348 | 424 | 528 | 1,300 |
| [Dataset4.csv](https://www.kaggle.com/mattgilgo/stock-related-tweet-sentiment) | 869 | 905 | 8,302 | 10,076 |
| [Dataset5.csv](https://github.com/surge-ai/stock-sentiment/blob/main/sentiment.csv) | 173 | 0 | 327 | 500 |
| [Dataset6.csv](https://github.com/poojathakoor/twitter-stock-sentiment/tree/master/twitter_stock_sentiment/training_data) | 600 | 500 | 707 | 1,807 |
| [Dataset7.csv](https://github.com/surge-ai/crypto-sentiment/blob/main/sentiment.csv) | 260 | 0 | 302 | 562 |
| [Dataset8.csv](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment) | 1789 | 7744 | 2398 | 11931 |
| [Dataset9.csv](https://ieee-dataport.org/open-access/stock-market-tweets-data) | 348 | 424 | 528 | 1300 |
| Total (no duplicates) | 6,647 | 17,352 | 18,280 | 42,279 |

## Model Details
FinTwitBERT is based on [FinBERT](https://huggingface.co/ProsusAI/finbert) with added masks for user mentions (`@USER`) and URLs (`[URL]`). The model is pre-trained for 10 epochs with a focus on minimizing loss and applying early stopping to prevent overfitting.

Find the pre-trained model and tokenizer here: [FinTwitBERT on HuggingFace](https://huggingface.co/StephanAkkerman/FinTwitBERT).

## Installation
```bash
# Clone this repository
git clone https://github.com/TimKoornstra/FinTwitBERT
# Install required packages
pip install -r requirements.txt
```

## Usage
The model can be finetuned for specific tasks such as sentiment classification. For more information about it, you can visit [our stock sentiment classifier repository](https://github.com/TimKoornstra/stock-sentiment-classifier).

## Citation
If you use FinTwitBERT or FinTwitBERT-sentiment in your research, please cite us as follows, noting that both authors contributed equally to this work:

```
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
Contributions are welcome! If you have a feature request, bug report, or proposal for code refactoring, please feel free to open an issue on GitHub. I appreciate your help in improving this project.

## License
This project is licensed under the GPL-3.0 License. See the [LICENSE](LICENSE) file for details.
