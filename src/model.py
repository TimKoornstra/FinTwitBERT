# > Imports
# Third party
from transformers import AutoTokenizer, BertForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import Dataset


class FinTwitBERT:
    def __init__(self):
        self.model = BertForMaskedLM.from_pretrained("ProsusAI/finbert")
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")

        special_tokens = ["@USER"]
        num_added_tokens = self.tokenizer.add_tokens(special_tokens)

        self.model.resize_token_embeddings(len(self.tokenizer))

    def encode(self, data):
        return self.tokenizer(data["text"], truncation=True, padding="max_length")

    def train(self,
              data: Dataset):

        data = data.map(self.encode, batched=True)
        data.set_format(type="torch", columns=["input_ids", "attention_mask"])

        # Prepare for MLM training
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm_probability=0.15)

        # Training
        training_args = TrainingArguments(
            output_dir="checkpoints/",
            overwrite_output_dir=True,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            save_steps=10_000,
            save_total_limit=2,
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
