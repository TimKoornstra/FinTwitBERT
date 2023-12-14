import wandb
from transformers import BertForSequenceClassification, AutoTokenizer, pipeline
from transformers.pipelines.pt_utils import KeyDataset
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score


class Evaluate:
    def __init__(self, use_baseline: bool = False, baseline_model: int = 1):
        if not use_baseline:
            self.model = BertForSequenceClassification.from_pretrained(
                "output/FinTwitBERT-sentiment",
                num_labels=3,
                id2label={0: "NEUTRAL", 1: "BULLISH", 2: "BEARISH"},
                label2id={"NEUTRAL": 0, "BULLISH": 1, "BEARISH": 2},
            )
            self.model.config.problem_type = "single_label_classification"
            self.tokenizer = AutoTokenizer.from_pretrained(
                "output/FinTwitBERT-sentiment"
            )
        else:
            if baseline_model == 0:
                self.model = BertForSequenceClassification.from_pretrained(
                    "yiyanghkust/finbert-tone", cache_dir="baseline/"
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "yiyanghkust/finbert-tone", cache_dir="baseline/"
                )
            elif baseline_model == 1:
                self.model = BertForSequenceClassification.from_pretrained(
                    "ProsusAI/finbert", cache_dir="baseline/"
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "ProsusAI/finbert", cache_dir="baseline/"
                )
        self.model.eval()
        # https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.pipeline
        self.pipeline = pipeline(
            "text-classification", model=self.model, tokenizer=self.tokenizer, device=0
        )

    def encode(self, data):
        return self.tokenizer(
            data["text"], truncation=True, padding="max_length", max_length=512
        )

    def load_test_data(self, tokenize: bool = True):
        dataset = load_dataset(
            "financial_phrasebank",
            cache_dir="data/finetune/",
            split="train",
            name="sentences_50agree",
        )

        # Rename sentence to text
        dataset = dataset.rename_column("sentence", "text")

        if tokenize:
            # Apply the tokenize function to the dataset
            tokenized_dataset = dataset.map(self.encode, batched=True)

            # Set the format for pytorch tensors
            tokenized_dataset.set_format(
                type="torch",
                columns=["input_ids", "token_type_ids", "attention_mask", "label"],
            )

            return tokenized_dataset
        return dataset

    def calculate_metrics(self, batch_size: int = 32):
        dataset = self.load_test_data(tokenize=False)

        # Convert numerical labels to textual labels
        true_labels = [
            dataset.features["label"].int2str(label) for label in dataset["label"]
        ]

        pred_labels = []
        for out in self.pipeline(KeyDataset(dataset, "text"), batch_size=batch_size):
            pred_labels.append(out["label"].lower())

        # Convert bullish to positive and bearish to negative
        label_mapping = {"bullish": "positive", "bearish": "negative"}
        pred_labels = [label_mapping.get(label, label) for label in pred_labels]

        accuracy = accuracy_score(true_labels, pred_labels)
        f1 = f1_score(true_labels, pred_labels, average="weighted")

        output = {
            "test/final_accuracy": accuracy,
            "test/final_f1_score": f1,
        }

        if wandb.run is not None:
            wandb.log(output)
        print(output)

        # Create confusion matrix and log it to wandb
        # https://docs.wandb.ai/guides/track/log/media
