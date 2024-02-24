# A Specialized Language Model for Financial Tweets

![FinTwitBERT Logo](img/logo.png)

---

<p align="center">
  <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Supported versions">
  <img src="https://img.shields.io/badge/license-GPL--3.0-orange" alt="License">
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"></a>
</p>


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
FinTwitBERT utilizes a diverse set of financial tweets for pre-training, including Taborda et al.'s [Stock Market Tweets Data](https://ieee-dataport.org/open-access/stock-market-tweets-data) with over 940K tweets, and our dataset, [Financial Tweets](https://huggingface.co/datasets/StephanAkkerman/financial-tweets), with detailed statistics provided below.

### Finetuning Datasets
For finetuning, we use several datasets, each offering varied sentiments in financial contexts. A collection of real-world, labeled datasets can be found on [Huggingface](https://huggingface.co/datasets/TimKoornstra/financial-tweets-sentiment). On top of that, we also created a synthetic dataset containing 1.43M tweets and corresponding sentiment labels. You can find that dataset [here](https://huggingface.co/datasets/TimKoornstra/synthetic-financial-tweets-sentiment).

## Model Details
FinTwitBERT is based on [FinBERT](https://huggingface.co/ProsusAI/finbert) with added masks for user mentions (`@USER`) and URLs (`[URL]`). The model is pre-trained for 10 epochs with a focus on minimizing loss and applying early stopping to prevent overfitting.

Access the pre-trained model and tokenizer at [FinTwitBERT on HuggingFace](https://huggingface.co/StephanAkkerman/FinTwitBERT). For the fine-tuned version, visit [FinTwitBERT-sentiment on HuggingFace](https://huggingface.co/StephanAkkerman/FinTwitBERT-sentiment).

## Installation
```bash
# Clone this repository
git clone https://github.com/TimKoornstra/FinTwitBERT
# Install required packages
pip install -r requirements.txt
```

## Usage
We offer two models: [FinTwitBERT](https://huggingface.co/StephanAkkerman/FinTwitBERT) and [FinTwitBERT-sentiment](https://huggingface.co/StephanAkkerman/FinTwitBERT-sentiment). The first is a pre-trained model and tokenizer for masked language modeling (MLM) which can be finetuned for other tasks such as sentiment analysis. This is what the second model is about, it is fine-tuned on sentiment analysis and labels tweets into three categories: bearish, neutral, and bullish.

### Pre-trained model
```python
from transformers import pipeline

pipe = pipeline(
    "fill-mask",
    model="StephanAkkerman/FinTwitBERT",
)
print(pipe("Bitcoin is a [MASK] coin."))
```

### Fine-tuned model
```python
from transformers import pipeline

pipe = pipeline(
    "sentiment-analysis",
    model="StephanAkkerman/FinTwitBERT-sentiment",
)

print(pipe("Nice 9% pre market move for $para, pump my calls Uncle Buffett 🤑"))
```

### Weights and Biases (wandb) usage
If you would like to train this model yourself and report the metrics to weights and biases (wandb.ai). You can do so by adding a wandb.env file with the following content: `WANDB_API_KEY=your_wandb_api_key`.

## Citation
If you use FinTwitBERT or FinTwitBERT-sentiment in your research, please cite us as follows, noting that both authors contributed equally to this work:

```bibtex
@misc{FinTwitBERT,
  author = {Stephan Akkerman, Tim Koornstra},
  title = {FinTwitBERT: A Specialized Language Model for Financial Tweets},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/TimKoornstra/FinTwitBERT}}
}
```

```bibtex
@misc{FinTwitBERT-sentiment,
  author = {Stephan Akkerman, Tim Koornstra},
  title = {FinTwitBERT-sentiment: A Sentiment Classifier for Financial Tweets},
  year = {2024},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/StephanAkkerman/FinTwitBERT-sentiment}}
}
```

## Contributing
Contributions are welcome! If you have a feature request, bug report, or proposal for code refactoring, please feel free to open an issue on GitHub. We appreciate your help in improving this project.

## License
This project is licensed under the GPL-3.0 License. See the [LICENSE](LICENSE) file for details.
