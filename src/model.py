# > Imports
import os

# Third party
import wandb
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
        self.model = BertForMaskedLM.from_pretrained(
            "yiyanghkust/finbert-pretrain", cache_dir="baseline/"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "yiyanghkust/finbert-pretrain", cache_dir="baseline/"
        )
        self.output_dir = "output/FinTwitBERT"

        special_tokens = ["@USER", "[URL]"]
        self.tokenizer.add_tokens(special_tokens)

        self.model.resize_token_embeddings(len(self.tokenizer))
        self.init_wandb()

    def init_wandb(self):
        with open("wandb_key.txt", "r") as file:
            wandb_api_key = file.read().strip()

        # Read the API key from the environment variable
        os.environ["WANDB_API_KEY"] = wandb_api_key

        # set the wandb project where this run will be logged
        os.environ["WANDB_PROJECT"] = "FinTwitBERT"

        # save your trained model checkpoint to wandb
        os.environ["WANDB_LOG_MODEL"] = "true"

        # turn off watch to log faster
        os.environ["WANDB_WATCH"] = "false"

    def encode(self, data):
        return self.tokenizer(data["text"], truncation=True, padding="max_length")

    def train(
        self,
        data: Dataset,
        validation: Dataset,
        batch_size: int = 64,
        num_train_epochs: int = 10,
        fold_num: int = 0,
    ):
        data = data.map(self.encode, batched=True)
        val = validation.map(self.encode, batched=True)

        data.set_format(type="torch", columns=["input_ids", "attention_mask"])
        val.set_format(type="torch", columns=["input_ids", "attention_mask"])

        # Prepare for MLM training
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm_probability=0.15
        )
        # Training
        # https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/trainer#transformers.TrainingArguments
        training_args = TrainingArguments(
            output_dir="checkpoints/",
            overwrite_output_dir=True,
            num_train_epochs=num_train_epochs,  # FinBERT uses 6
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            evaluation_strategy="steps",
            save_steps=10_000,
            eval_steps=10_000,
            # logging_steps=10_000, Use the default
            save_total_limit=2,
            learning_rate=2e-5,  # FinBERT uses 5e-5 to 2e-5
            fp16=True,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,  # Lower loss indicates better performance
            # gradient_accumulation_steps=1,  # FinBERT uses 1
            warmup_ratio=0.2,  # FinBERT uses 0.2
            save_safetensors=True,
            # weight_decay=0.01,  # FinBERT uses 0.01
            report_to="wandb",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=data,
            eval_dataset=val,
            data_collator=data_collator,
        )

        # Train
        trainer.train()

        # Change output dir if it's a fold
        if fold_num > 0:
            self.output_dir += f"_fold_{fold_num}"

        # Save the model
        trainer.save_model(self.output_dir)

        # Save the tokenizer
        self.tokenizer.save_pretrained(self.output_dir)
