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
    def __init__(self, model, num_layers, unfreeze_rate, total_steps):
        self.model = model
        self.num_layers = num_layers
        self.unfreeze_rate = unfreeze_rate
        self.total_steps = total_steps
        self.next_unfreeze_step = int(total_steps * unfreeze_rate)
        self.layers_unfrozen = False  # To track if the initial unfreezing has been done

    def on_train_begin(self, args, state, control, **kwargs):
        # Unfreeze the last layer from the beginning
        for param in self.model.bert.encoder.layer[-1].parameters():
            param.requires_grad = True

    def on_step_end(self, args, state, control, **kwargs):
        # Check if it is time to unfreeze the next layer
        if (
            state.global_step == self.next_unfreeze_step
            and self.layers_unfrozen < self.num_layers - 1
        ):
            # Calculate the layer index to unfreeze next
            layer_index = self.num_layers - 2 - self.layers_unfrozen
            for param in self.model.bert.encoder.layer[layer_index].parameters():
                param.requires_grad = True
            self.layers_unfrozen += 1  # Increment the count of unfrozen layers
            # Update the next step to unfreeze another layer
            self.next_unfreeze_step += int(self.total_steps * self.unfreeze_rate)

            # Log the unfreezing action
            print(f"Unfroze layer {layer_index} at step {state.global_step}")


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

    def train(
        self,
        data: Dataset,
        validation: Dataset,
        batch_size: int = 64,
        num_train_epochs: int = 6,
    ):
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

        # Instantiate the EarlyStoppingCallback
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=3,
        )

        # Calculate the total number of steps (assuming one epoch here for simplicity)
        total_dataset_size = len(data)  # Replace with your actual dataset size
        total_steps = (
            total_dataset_size // batch_size
        ) * num_train_epochs  # num_train_epochs is your total epochs

        # Instantiate the GradualUnfreezingCallback
        num_layers = len(self.model.bert.encoder.layer)
        gradual_unfreezing_callback = GradualUnfreezingCallback(
            model=self.model,
            num_layers=num_layers,
            unfreeze_rate=0.33,  # Adjust if you want different rates
            total_steps=total_steps,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=data,
            eval_dataset=val,
            data_collator=data_collator,
            # callbacks=[gradual_unfreezing_callback],
        )

        # Train
        trainer.train()

        # Save the model
        trainer.save_model("output/FinTwitBERT")

        # Save the tokenizer
        self.tokenizer.save_pretrained("output/FinTwitBERT")
