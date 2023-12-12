# > Imports
import os
import logging
import json

# Third party
from dotenv import load_dotenv
from transformers import (
    AutoTokenizer,
    BertForMaskedLM,
    BertForSequenceClassification,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from torch.optim import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
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


def compute_metrics(pred) -> dict:
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(
        labels, preds, average="weighted"
    )  # 'weighted' can be replaced with 'binary' or 'macro' based on your specific needs
    return {"accuracy": acc, "f1": f1}


class FinTwitBERT:
    def __init__(self, mode="pretrain"):
        if mode not in ["pretrain", "finetune"]:
            raise ValueError(f"Unsupported mode: {mode}")

        self.mode = mode

        # If the model will be pretrained
        if mode == "pretrain":
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

        # If the model will be finetuned
        elif mode == "finetune":
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
        # Check if a .env file exists
        if not os.path.exists("wandb.env"):
            logging.warning("No wandb.env file found")
            return

        # Load the .env file
        load_dotenv(dotenv_path="wandb.env")

        # Read the API key from the environment variable
        os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")

        # set the wandb project where this run will be logged
        os.environ["WANDB_PROJECT"] = (
            "FinTwitBERT" if self.mode == "pretrain" else "FinTwitBERT-sentiment"
        )

        # save your trained model checkpoint to wandb
        os.environ["WANDB_LOG_MODEL"] = "true"

        # turn off watch to log faster
        os.environ["WANDB_WATCH"] = "false"

    def encode(self, data):
        return self.tokenizer(
            data["text"], truncation=True, padding="max_length", max_length=512
        )

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

    def discriminative_lr(self, mode_args, data):
        # Define different learning rates
        lr = mode_args[self.mode]["learning_rate"]
        dft_rate = 1.2

        # Disable decay for these layers
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

        encoder_params = []
        for i in range(12):
            encoder_decay = {
                "params": [
                    p
                    for n, p in list(
                        self.model.bert.encoder.layer[i].named_parameters()
                    )
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
                "lr": lr / (dft_rate ** (12 - i)),
            }
            encoder_nodecay = {
                "params": [
                    p
                    for n, p in list(
                        self.model.bert.encoder.layer[i].named_parameters()
                    )
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": lr / (dft_rate ** (12 - i)),
            }
            encoder_params.append(encoder_decay)
            encoder_params.append(encoder_nodecay)

        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in list(self.model.bert.embeddings.named_parameters())
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
                "lr": lr / (dft_rate**13),
            },
            {
                "params": [
                    p
                    for n, p in list(self.model.bert.embeddings.named_parameters())
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": lr / (dft_rate**13),
            },
            {
                "params": [
                    p
                    for n, p in list(self.model.bert.pooler.named_parameters())
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
                "lr": lr,
            },
            {
                "params": [
                    p
                    for n, p in list(self.model.bert.pooler.named_parameters())
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": lr,
            },
            {
                "params": [
                    p
                    for n, p in list(self.model.classifier.named_parameters())
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
                "lr": lr,
            },
            {
                "params": [
                    p
                    for n, p in list(self.model.classifier.named_parameters())
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": lr,
            },
        ]

        optimizer_grouped_parameters.extend(encoder_params)

        # Initialize the optimizer
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr)  # , correct_bias=False)

        # You might need to adjust the scheduler setup as per your requirement
        total_steps = len(data) * mode_args[self.mode]["num_train_epochs"]
        warmup_steps = int(total_steps * mode_args[self.mode]["warmup_ratio"])
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        return optimizer, scheduler

    def train(
        self,
        data: Dataset,
        validation: Dataset,
        fold_num: int = 0,
    ):
        # Read model args from config.json
        with open("config.json", "r") as config_file:
            config = json.load(config_file)

        mode_args = {
            "pretrain": config["pretrain_args"],
            "finetune": config["finetune_args"],
        }

        data = data.map(self.encode, batched=True)
        val = validation.map(self.encode, batched=True)

        mode_columns = {
            "pretrain": ["input_ids", "attention_mask"],
            "finetune": ["input_ids", "token_type_ids", "attention_mask", "label"],
        }

        data.set_format(type="torch", columns=mode_columns[self.mode])
        val.set_format(type="torch", columns=mode_columns[self.mode])

        training_args = TrainingArguments(**mode_args[self.mode], **config["base_args"])

        # Add gradual unfreezing callback if enabled
        callbacks = []
        if config[f"{self.mode}_gradual_unfreeze"]:
            batch_size = mode_args[self.mode]["per_device_train_batch_size"]
            num_train_epochs = mode_args[self.mode]["num_train_epochs"]
            self.gradual_unfreeze(unfreeze_last_n_layers=1)
            total_steps = (len(data) // batch_size) * num_train_epochs
            callbacks.append(
                GradualUnfreezingCallback(
                    model=self.model,
                    num_layers=len(self.model.bert.encoder.layer),
                    unfreeze_rate=0.33,
                    total_steps=total_steps,
                )
            )

        # Use the MLM data collator when pretraining
        data_collator = None
        if self.mode == "pretrain":
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer, mlm_probability=0.15
            )

        # Compute F1 and accuracy scores when finetuning
        compute_metrics_fn = compute_metrics if self.mode == "finetune" else None

        optimizers = (None, None)
        if config[f"{self.mode}_discriminative_lr"]:
            optimizers = self.discriminative_lr(mode_args, data)

        # https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/trainer#transformers.TrainingArguments
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=data,
            eval_dataset=val,
            data_collator=data_collator,
            compute_metrics=compute_metrics_fn,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Train the model
        trainer.train()

        # Handle output directory for different folds
        output_dir = (
            f"{self.output_dir}_fold_{fold_num}" if fold_num > 0 else self.output_dir
        )

        # Save the model and tokenizer
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
