from tqdm import tqdm
from transformers import BertForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling
import torch
from torch.utils.data import DataLoader
from data import load_validation_data


class Evaluate:
    def __init__(self, use_baseline: bool = False):
        if not use_baseline:
            self.model = BertForMaskedLM.from_pretrained("output/FinTwitBERT")
            self.tokenizer = AutoTokenizer.from_pretrained("output/FinTwitBERT")
        else:
            self.model = BertForMaskedLM.from_pretrained(
                "bert-base-uncased", cache_dir="baseline/"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                "bert-base-uncased", cache_dir="baseline/"
            )
        self.model.eval()

    def load_validation_data(self, length: int = None):
        # Load preprocessed data using your custom function
        dataset = load_validation_data()

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

        print(f"Perplexity: {perplexity}")
