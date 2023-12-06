# > Imports
import os

# Third party
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
from sklearn.metrics import accuracy_score


class FinTwitBERT:
    def __init__(self) -> None:
        self.model = BertForSequenceClassification.from_pretrained(
            "output/FinTwitBERT", num_labels=3
        )
        self.model.config.problem_type = "single_label_classification"
        self.tokenizer = AutoTokenizer.from_pretrained("output/FinTwitBERT")
        self.output_dir = "output/FinTwitBERT-sentiment"

        self.init_wandb()

    def init_wandb(self):
        with open("wandb_key.txt", "r") as file:
            wandb_api_key = file.read().strip()

        # Read the API key from the environment variable
        os.environ["WANDB_API_KEY"] = wandb_api_key

        # set the wandb project where this run will be logged
        os.environ["WANDB_PROJECT"] = "FinTwitBERT-sentiment"

        # save your trained model checkpoint to wandb
        os.environ["WANDB_LOG_MODEL"] = "true"

        # turn off watch to log faster
        os.environ["WANDB_WATCH"] = "false"

    def encode(self, data):
        # max_length is necessary for finetuning
        return self.tokenizer(
            data["text"],
            truncation=True,
            padding="max_length",
            max_length=512,  # 512 is max
        )

    def compute_metrics(self, pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        return {"accuracy": accuracy_score(labels, preds)}

    def calculate_steps(self, batch_size, base_batch_size=64, base_steps=500):
        return (base_batch_size * base_steps) // batch_size

    def train(
        self,
        data: Dataset,
        validation: Dataset,
        batch_size: int = 128,
        num_train_epochs: int = 10,
        fold_num: int = 0,
    ):
        data = data.map(self.encode, batched=True)
        val = validation.map(self.encode, batched=True)

        data.set_format(
            type="torch",
            columns=["input_ids", "token_type_ids", "attention_mask", "label"],
        )
        val.set_format(
            type="torch",
            columns=["input_ids", "token_type_ids", "attention_mask", "label"],
        )

        steps = self.calculate_steps(batch_size)

        # Training
        # https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/trainer#transformers.TrainingArguments
        training_args = TrainingArguments(
            output_dir="checkpoints/",
            overwrite_output_dir=True,
            num_train_epochs=num_train_epochs,  # FinBERT uses 6
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            evaluation_strategy="steps",
            save_steps=steps,
            eval_steps=steps,
            logging_steps=steps,
            save_total_limit=2,
            learning_rate=5e-5,  # FinBERT uses 5e-5 to 2e-5
            fp16=True,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,  # Higher accuracy is better
            # gradient_accumulation_steps=1,  # FinBERT uses 1
            warmup_ratio=0.4,  # FinBERT uses 0.2
            save_safetensors=True,
            weight_decay=0.01,  # FinBERT uses 0.01
            report_to="wandb",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=data,
            eval_dataset=val,
            compute_metrics=self.compute_metrics,
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
