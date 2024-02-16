import wandb
from transformers import BertForSequenceClassification, AutoTokenizer, pipeline
from transformers.pipelines.pt_utils import KeyDataset
from datasets import load_dataset
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

from data import load_finetuning_data


class Evaluate:
    def __init__(self, use_baseline: bool = False, baseline_model: int = 1):
        if not use_baseline:
            labels = ["NEUTRAL", "BULLISH", "BEARISH"]
            self.model = BertForSequenceClassification.from_pretrained(
                "output/FinTwitBERT-sentiment",
                num_labels=len(labels),
                id2label={k: v for k, v in enumerate(labels)},
                label2id={v: k for k, v in enumerate(labels)},
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
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0,
            padding="max_length",
            truncation=True,
            max_length=512,
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

    def evaluate_model(self):
        # For plotting the confusion matrix of the validation / eval set
        true_labels, pred_labels = self.get_labels(category="finetune")
        self.plot_confusion_matrix("eval", true_labels, pred_labels)

        # For calculating the accuracy and F1 score
        # And plotting the confusion matrix
        true_labels, pred_labels = self.get_labels(category="test")
        self.calculate_metrics(true_labels, pred_labels)
        self.plot_confusion_matrix("test", true_labels, pred_labels)

    def get_labels(self, category: str, batch_size: int = 32):
        if category == "test":
            dataset = self.load_test_data(tokenize=False)
            # Convert numerical labels to textual labels
            true_labels = [
                dataset.features["label"].int2str(label) for label in dataset["label"]
            ]

        elif category == "finetune":
            # Get the validation dataset
            _, dataset = load_finetuning_data()
            # 0: neutral, 1: bullish, 2: bearish
            int2str = {0: "neutral", 1: "bullish", 2: "bearish"}
            true_labels = [int2str[label] for label in dataset["label"]]
        else:
            raise ValueError("Invalid category name")

        pred_labels = []
        for out in self.pipeline(KeyDataset(dataset, "text"), batch_size=batch_size):
            pred_labels.append(out["label"].lower())

        # Convert bullish to positive and bearish to negative
        if category == "test":
            label_mapping = {"bullish": "positive", "bearish": "negative"}
            pred_labels = [label_mapping.get(label, label) for label in pred_labels]

        return true_labels, pred_labels

    def calculate_metrics(self, true_labels, pred_labels):
        # Compute accuracy and F1 score
        accuracy = accuracy_score(true_labels, pred_labels)
        f1 = f1_score(true_labels, pred_labels, average="weighted")

        # Log metrics to wandb
        output = {
            "test/final_accuracy": accuracy,
            "test/final_f1_score": f1,
        }

        if wandb.run is not None:
            wandb.log(output)

        print(output)

    def plot_confusion_matrix(self, category: str, true_labels, pred_labels):
        # Create confusion matrix
        label_encoder = LabelEncoder()
        true_labels_encoded = label_encoder.fit_transform(true_labels)
        pred_labels_encoded = label_encoder.transform(pred_labels)

        cm = confusion_matrix(true_labels_encoded, pred_labels_encoded)
        labels = label_encoder.classes_

        # Plot confusion matrix
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            ax=ax,
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
        )
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.title("Confusion Matrix")

        # Log confusion matrix to wandb
        wandb.log({f"{category}/confusion_matrix": wandb.Image(fig)})

        # Close the plot
        plt.close(fig)
