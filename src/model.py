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
        self.model = BertForMaskedLM.from_pretrained(
            "ProsusAI/finbert-pretrain", cache_dir="baseline/"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "ProsusAI/finbert-pretrain", cache_dir="baseline/"
        )

        special_tokens = ["@USER", "[URL]"]
        self.tokenizer.add_tokens(special_tokens)

        self.model.resize_token_embeddings(len(self.tokenizer))

    def encode(self, data):
        return self.tokenizer(data["text"], truncation=True, padding="max_length")

    def gradual_unfreeze(self, unfreeze_last_n_layers: int):
        # Freeze all layers first
        for param in self.model.base_model.parameters():
            param.requires_grad = False

        # Count the number of layers in BertForMaskedLM model
        num_layers = len(self.model.base_model.encoder.layer)

        # Layers to unfreeze
        layers_to_unfreeze = num_layers - unfreeze_last_n_layers

        # Unfreeze the last n layers
        for layer in self.model.base_model.encoder.layer[layers_to_unfreeze:]:
            for param in layer.parameters():
                param.requires_grad = True

    def train(
        self,
        data: Dataset,
        validation: Dataset,
        batch_size: int = 32,
        num_train_epochs: int = 6,
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

        # Save the model
        trainer.save_model("output/FinTwitBERT")

        # Save the tokenizer
        self.tokenizer.save_pretrained("output/FinTwitBERT")
