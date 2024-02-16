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
from datasets import Dataset, concatenate_datasets
from sklearn.metrics import accuracy_score, f1_score
import eval.finetune
import eval.pretrain

from data import (
    load_pretraining_data,
    load_finetuning_data,
    load_synthetic_data,
    load_tweet_eval,
    simple_oversample,
    synonym_oversample,
)


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
    def __init__(self):
        # Read model args from config.json
        with open("config.json", "r") as config_file:
            self.config = json.load(config_file)

        # Set the mode
        self.mode = self.config["mode"]
        if self.mode not in ["pretrain", "finetune", "pre-finetune"]:
            raise ValueError(f"Unsupported mode: {self.mode}")

        # Get the mode-specific args
        self.mode_args = self.config[self.mode][f"{self.mode}_args"]
        self.output_dir = self.config[self.mode]["output_dir"]

        # If the model will be pretrained
        if self.mode == "pretrain":
            self.model = BertForMaskedLM.from_pretrained(
                self.config[self.mode]["pretrained_model"], cache_dir="baseline/"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config[self.mode]["pretrained_tokenizer"], cache_dir="baseline/"
            )
            special_tokens = ["@USER", "[URL]"]
            self.tokenizer.add_tokens(special_tokens)
            self.model.resize_token_embeddings(len(self.tokenizer))
            self.data, self.validation = load_pretraining_data()

        elif self.mode == "pre-finetune":
            labels = ["NEUTRAL", "BULLISH", "BEARISH"]
            self.model = BertForSequenceClassification.from_pretrained(
                self.config[self.mode]["pretrained_model"],
                num_labels=len(labels),
                id2label={k: v for k, v in enumerate(labels)},
                label2id={v: k for k, v in enumerate(labels)},
            )
            self.model.config.problem_type = "single_label_classification"
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config[self.mode]["pretrained_tokenizer"]
            )
            self.data, self.validation = load_tweet_eval()

        # If the model will be finetuned
        elif self.mode == "finetune":
            labels = ["NEUTRAL", "BULLISH", "BEARISH"]
            self.model = BertForSequenceClassification.from_pretrained(
                self.config[self.mode]["pretrained_model"],
                num_labels=len(labels),
                id2label={k: v for k, v in enumerate(labels)},
                label2id={v: k for k, v in enumerate(labels)},
            )
            self.model.config.problem_type = "single_label_classification"
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config[self.mode]["pretrained_tokenizer"]
            )

            # special_tokens = ["[TICKER]"]
            # self.tokenizer.add_tokens(special_tokens)
            # self.model.resize_token_embeddings(len(self.tokenizer))

            synthetic = load_synthetic_data()
            gt_training, self.validation = load_finetuning_data()

            # Merge the synthetic and original datasets
            self.data = concatenate_datasets(
                [
                    synthetic, gt_training
                ]
            )

            # Oversample the data if enabled
            if self.config[self.mode]["oversampling"] == "simple":
                self.data = simple_oversample(self.data)
            elif self.config[self.mode]["oversampling"] == "synonym":
                self.data = synonym_oversample(self.data)

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
        os.environ["WANDB_PROJECT"] = self.output_dir.split("/")[-1]

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

    def get_layer_params(self, layer, learning_rate, no_decay):
        """Get parameters for a specific layer with and without decay."""
        params_with_decay = {
            "params": [
                p
                for n, p in layer.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,  # Should we use the config here?
            "lr": learning_rate,
        }
        params_without_decay = {
            "params": [
                p
                for n, p in layer.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "lr": learning_rate,
        }
        return [params_with_decay, params_without_decay]

    def discriminative_lr(self, data):
        # Define base learning rate and decay factor
        lr = self.mode_args["learning_rate"]
        dft_rate = self.config[self.mode]["dft_rate"]
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

        # Prepare encoder parameters with discriminative learning rates
        encoder_params = []
        for i in range(12):
            layer_lr = lr / (dft_rate ** (12 - i))
            encoder_layer_params = self.get_layer_params(
                self.model.bert.encoder.layer[i], layer_lr, no_decay
            )
            encoder_params.extend(encoder_layer_params)

        # Embeddings, pooler, and classifier parameters
        embeddings_params = self.get_layer_params(
            self.model.bert.embeddings, lr / (dft_rate**13), no_decay
        )
        pooler_params = self.get_layer_params(self.model.bert.pooler, lr, no_decay)
        classifier_params = self.get_layer_params(self.model.classifier, lr, no_decay)

        # Combine all parameter groups
        optimizer_grouped_parameters = (
            embeddings_params + pooler_params + classifier_params
        )
        optimizer_grouped_parameters.extend(encoder_params)

        # Initialize optimizer and scheduler
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
        total_steps = len(data) * self.mode_args["num_train_epochs"]
        warmup_steps = int(total_steps * self.mode_args["warmup_ratio"])
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        return optimizer, scheduler

    def get_gradual_unfreezing_callback(self, data):
        batch_size = self.mode_args["per_device_train_batch_size"]
        num_train_epochs = self.mode_args["num_train_epochs"]
        self.gradual_unfreeze(unfreeze_last_n_layers=1)
        total_steps = (len(data) // batch_size) * num_train_epochs
        return GradualUnfreezingCallback(
            model=self.model,
            num_layers=len(self.model.bert.encoder.layer),
            unfreeze_rate=0.33,  # Could add this to config
            total_steps=total_steps,
        )

    def encode_data(self, data: Dataset):
        mode_columns = {
            "pretrain": ["input_ids", "attention_mask"],
            "finetune": ["input_ids", "token_type_ids", "attention_mask", "label"],
            "pre-finetune": ["input_ids", "token_type_ids", "attention_mask", "label"],
        }

        data = data.map(self.encode, batched=True)
        data.set_format(type="torch", columns=mode_columns[self.mode])

        return data

    def train(
        self,
        fold_num: int = 0,
    ):
        # Define default values
        callbacks = []
        optimizers = (None, None)
        data_collator = None

        data = self.encode_data(self.data)
        val = self.encode_data(self.validation)

        training_args = TrainingArguments(**self.mode_args, **self.config["base_args"])

        # Add gradual unfreezing callback if enabled
        if self.config[self.mode]["gradual_unfreeze"]:
            callbacks.append(self.get_gradual_unfreezing_callback(data))

        # Use the MLM data collator when pretraining
        if self.mode == "pretrain":
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer, mlm_probability=0.15
            )

        # Compute F1 and accuracy scores when finetuning
        compute_metrics_fn = (
            compute_metrics if self.mode in ["finetune", "pre-finetune"] else None
        )

        # Enable discriminative learning rates when finetuning
        if self.config[self.mode]["discriminative_lr"]:
            optimizers = self.discriminative_lr(data)

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

        # TODO: pass val to evaluate_model
        if self.mode == "finetune":
            evaluate = eval.finetune.Evaluate()
            evaluate.evaluate_model()

        elif self.mode == "pretrain":
            evaluate = eval.pretrain.Evaluate()
            evaluate.calculate_perplexity()
            evaluate.calculate_masked_examples()
