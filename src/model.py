# > Imports
# Third party
from transformers import (
    AutoTokenizer,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from datasets import Dataset


class GradualUnfreezingCallback(TrainerCallback):
    def __init__(self, unfreeze_schedule):
        self.unfreeze_schedule = unfreeze_schedule

    def on_epoch_begin(self, args, state, control, model):
        # Calculate the number of layers to unfreeze based on the current epoch
        num_layers_to_unfreeze = self.unfreeze_schedule.get(state.epoch, None)

        if num_layers_to_unfreeze is not None:
            # Gradually unfreeze layers from the top (higher numbered layers)
            for i, layer in enumerate(model.bert.encoder.layer):
                # Layers are numbered in reverse (i.e., last layer is num_layers_to_unfreeze - 1)
                if i >= len(model.bert.encoder.layer) - num_layers_to_unfreeze:
                    for param in layer.parameters():
                        param.requires_grad = True


class EarlyStoppingCallback(TrainerCallback):
    """Early stopping callback for Hugging Face Trainer class."""

    def __init__(self, early_stopping_patience: int):
        self.early_stopping_patience = early_stopping_patience
        self.best_metric = None
        self.patience_counter = 0

    def on_evaluate(self, args, state, control, **kwargs):
        metric_value = kwargs["metrics"]["eval_loss"]
        if self.best_metric is None or metric_value < self.best_metric:
            self.best_metric = metric_value
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.early_stopping_patience:
                print(
                    f"No improvement on eval_loss for {self.early_stopping_patience} evaluations."
                )
                print("Early stopping...")
                control.should_training_stop = True


class FinTwitBERT:
    def __init__(self):
        self.model = BertForMaskedLM.from_pretrained("ProsusAI/finbert")
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")

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

    def train(self, data: Dataset, validation: Dataset, batch_size: int = 32):
        data = data.map(self.encode, batched=True)
        val = validation.map(self.encode, batched=True)

        data.set_format(type="torch", columns=["input_ids", "attention_mask"])
        val.set_format(type="torch", columns=["input_ids", "attention_mask"])

        # Freeze all layers initially, except the last one
        self.gradual_unfreeze(unfreeze_last_n_layers=1)

        # Prepare for MLM training
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm_probability=0.15
        )
        # Training
        # https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/trainer#transformers.TrainingArguments
        training_args = TrainingArguments(
            output_dir="checkpoints/",
            overwrite_output_dir=True,
            num_train_epochs=6,  # FinBERT uses 6
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
            weight_decay=0.01, #FinBERT uses 0.01
        )

        # Instantiate the EarlyStoppingCallback
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=3,
        )

        # Define your gradual unfreezing schedule
        # The key is the epoch number, and the value is the number of layers to unfreeze
        unfreeze_schedule = {
            1: 1,  # Unfreeze the last layer at epoch 1
            2: 3,  # Unfreeze the last 3 layers at epoch 2
            3: 6,  # Unfreeze the last 6 layers at epoch 3
            4: 12,  # Unfreeze all 12 encoder layers at epoch 4 (assuming 'bert-base' with 12 layers total)
        }

        # Instantiate the gradual unfreezing callback
        gradual_unfreezing_callback = GradualUnfreezingCallback(
            unfreeze_schedule=unfreeze_schedule
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=data,
            eval_dataset=val,
            data_collator=data_collator,
            callbacks=[gradual_unfreezing_callback],
        )

        # Train
        trainer.train()

        # Save the model
        trainer.save_model("output/FinTwitBERT")

        # Save the tokenizer
        self.tokenizer.save_pretrained("output/FinTwitBERT")
