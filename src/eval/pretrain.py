import os

from tqdm import tqdm
from transformers import BertForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling
import torch
from torch.utils.data import DataLoader
import wandb

from data import load_test_data


class Evaluate:
    def __init__(self, use_baseline: bool = False):
        if not use_baseline:
            self.model = BertForMaskedLM.from_pretrained("output/FinTwitBERT")
            self.tokenizer = AutoTokenizer.from_pretrained("output/FinTwitBERT")
        else:
            self.model = BertForMaskedLM.from_pretrained(
                "yiyanghkust/finbert-pretrain", cache_dir="baseline/"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                "yiyanghkust/finbert-pretrain", cache_dir="baseline/"
            )
        self.model.eval()

    def load_validation_data(self, length: int = None):
        # Load preprocessed data using your custom function
        dataset = load_test_data()

        # Select a subset of the data
        if length:
            dataset = dataset.select(range(length))

        # Apply the tokenize function to the dataset
        tokenized_dataset = dataset.map(self.encode, batched=True)

        # Set the format for pytorch tensors
        tokenized_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "token_type_ids"]
        )

        return tokenized_dataset

    # Same function as in model.py
    def encode(self, data):
        return self.tokenizer(data["text"], truncation=True, padding="max_length")

    def calculate_perplexity(self, val_length=None, batch_size=16):
        encoded_dataset = self.load_validation_data(val_length)

        # Prepare for MLM training
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15
        )

        # DataLoader
        dataloader = DataLoader(
            encoded_dataset, batch_size=batch_size, collate_fn=data_collator
        )

        total_loss = 0
        total_length = 0

        with torch.no_grad():
            for batch in tqdm(dataloader):
                outputs = self.model(
                    **{k: v.to(self.model.device) for k, v in batch.items()}
                )
                loss = outputs.loss
                total_loss += loss.item() * batch["input_ids"].shape[0]  # batch size
                total_length += batch["input_ids"].shape[0]

        # Calculate perplexity
        try:
            # Make sure to wrap the float in a tensor before calling torch.exp
            average_loss = total_loss / total_length
            perplexity = torch.exp(torch.tensor(average_loss)).item()
        except OverflowError:
            perplexity = float("inf")

        output = {
            "perplexity": perplexity,
        }

        if not os.path.exists(".env"):
            print(output)
        else:
            wandb.log(output)

    def calculate_masked_examples(self):
        examples = [
            "Paris is the [MASK] of France.",
            "The goal of life is [MASK].",
            "AAPL is a [MASK] sector stock.",
            "I predict that this stock will go [MASK].",
        ]

        for example in examples:
            self.calculate_masked_probs(example)
            print()

    def calculate_masked_probs(self, text: str, top_k: int = 5):
        input = self.tokenizer.encode_plus(text, return_tensors="pt")

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**input)
            predictions = outputs.logits

        # Identify the masked index and get the top 5 likely token indices
        masked_index = torch.where(input["input_ids"] == self.tokenizer.mask_token_id)[
            1
        ].tolist()

        # Use the logits to get the top_k token predictions for the masked token
        probs = torch.nn.functional.softmax(predictions[0, masked_index[0]], dim=-1)
        top_k_weights, top_k_indices = torch.topk(probs, top_k, dim=-1)

        # Decode the top k indices to tokens and get their corresponding probabilities
        predicted_tokens = self.tokenizer.convert_ids_to_tokens(top_k_indices.tolist())
        predicted_probs = top_k_weights.tolist()

        # Print the result
        print("Masked sentence:", text)
        print(f"Top {top_k} predicted tokens and probabilities:")
        for token, prob in zip(predicted_tokens, predicted_probs):
            print(f"{token}: {prob:.4f}")  # Formats the probability to 4 decimal places
