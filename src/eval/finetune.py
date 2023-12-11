import os

from tqdm import tqdm
import torch
import wandb
from transformers import BertForMaskedLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score


class Evaluate:
    def __init__(self, use_baseline: bool = False):
        if not use_baseline:
            self.model = BertForMaskedLM.from_pretrained("output/FinTwitBERT-sentiment")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "output/FinTwitBERT-sentiment"
            )
        else:
            self.model = BertForMaskedLM.from_pretrained(
                "yiyanghkust/finbert-tone", cache_dir="baseline/"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                "yiyanghkust/finbert-tone", cache_dir="baseline/"
            )
        self.model.eval()

    def encode(self, data):
        return self.tokenizer(
            data["text"], truncation=True, padding="max_length", max_length=512
        )

    def load_test_data(self):
        dataset = load_dataset(
            "financial_phrasebank",
            cache_dir="data/finetune/",
            split="train",
            name="sentences_50agree",
        )

        # Rename sentence to text
        dataset = dataset.rename_column("sentence", "text")

        # Apply the tokenize function to the dataset
        tokenized_dataset = dataset.map(self.encode, batched=True)

        # Set the format for pytorch tensors
        tokenized_dataset.set_format(
            type="torch",
            columns=["input_ids", "token_type_ids", "attention_mask", "label"],
        )

        return tokenized_dataset

    def calculate_metrics(self, batch_size: int = 32):
        tokenized_dataset = self.load_test_data()
        loader = DataLoader(tokenized_dataset, batch_size=batch_size)
        total_loss = 0
        true_labels = []
        pred_labels = []

        with torch.no_grad():
            for batch in tqdm(loader):
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                logits = outputs.logits
                loss = outputs.loss
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=-1)
                true_labels.extend(batch["labels"].tolist())
                pred_labels.extend(predictions.tolist())

        average_loss = total_loss / len(loader)
        accuracy = accuracy_score(true_labels, pred_labels)
        f1 = f1_score(true_labels, pred_labels, average="weighted")

        output = {
            "test/final_average_loss": average_loss,
            "test/final_accuracy": accuracy,
            "test/final_f1_score": f1,
        }

        if not os.path.exists(".env"):
            print(output)
        else:
            wandb.log(output)
