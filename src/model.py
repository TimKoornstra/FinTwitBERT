# > Imports
# Third party
from transformers import (
    AutoTokenizer,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset


class FinTwitBERT:
    def __init__(self):
        self.model = BertForMaskedLM.from_pretrained("ProsusAI/finbert")
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")

        special_tokens = ["@USER"]
        self.tokenizer.add_tokens(special_tokens)

        self.model.resize_token_embeddings(len(self.tokenizer))

    def encode(self, data):
        return self.tokenizer(data["text"], truncation=True, padding="max_length")

    def train(self, data: Dataset):
        data = data.map(self.encode, batched=True)
        data.set_format(type="torch", columns=["input_ids", "attention_mask"])

        # Prepare for MLM training
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm_probability=0.15
        )

        # Training
        # https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/trainer#transformers.TrainingArguments
        training_args = TrainingArguments(
            output_dir="checkpoints/",
            overwrite_output_dir=True,
            num_train_epochs=5,
            per_device_train_batch_size=1,  # 64 on cluster
            save_steps=5_000,
            save_total_limit=2,
            # learning_rate=2e-5,
            fp16=True,
            # gradient_accumulation_steps=2,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=data,
        )

        # Train
        trainer.train()

        # Save the model
        trainer.save_model("output/FinTwitBERT")

        # Save the tokenizer
        self.tokenizer.save_pretrained("output/FinTwitBERT")
