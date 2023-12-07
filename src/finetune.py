# > Imports
import os

# Third party
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score


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


class FinTwitBERT:
    def __init__(self) -> None:
        self.model = BertForSequenceClassification.from_pretrained(
            "output/FinTwitBERT",
            num_labels=3,
            id2label={0: "NEUTRAL", 1: "BULLISH", 2: "BEARISH"},
            label2id={"NEUTRAL": 0, "BULLISH": 1, "BEARISH": 2},
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
        acc = accuracy_score(labels, preds)
        f1 = f1_score(
            labels, preds, average="weighted"
        )  # 'weighted' can be replaced with 'binary' or 'macro' based on your specific needs
        return {"accuracy": acc, "f1": f1}

    def calculate_steps(self, batch_size, base_batch_size=64, base_steps=500):
        return (base_batch_size * base_steps) // batch_size

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
        batch_size: int = 128,
        num_train_epochs: int = 20,
        fold_num: int = 0,
        gradual_unfreeze: bool = True,
    ):
        callbacks = None

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

        if gradual_unfreeze:
            # Freeze all layers initially, except the last one
            self.gradual_unfreeze(unfreeze_last_n_layers=1)

            # Instantiate the GradualUnfreezingCallback
            gradual_unfreezing_callback = GradualUnfreezingCallback(
                model=self.model,
                num_layers=len(self.model.bert.encoder.layer),
                unfreeze_rate=0.33,  # Adjust if you want different rates
                total_steps=(len(data) // batch_size) * num_train_epochs,
            )

            callbacks = [gradual_unfreezing_callback]

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
            learning_rate=2e-4,  # FinBERT uses 5e-5 to 2e-5
            fp16=True,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,  # Higher accuracy is better
            # gradient_accumulation_steps=1,  # FinBERT uses 1
            warmup_ratio=0.2,  # FinBERT uses 0.2
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
            callbacks=callbacks,
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
