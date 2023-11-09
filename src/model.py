# > Imports
# Third party
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
from sklearn.metrics import accuracy_score


class FinTwitBERT:
    def __init__(self):
        self.model = BertForMaskedLM.from_pretrained("ProsusAI/finbert")
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")

        special_tokens = ["@USER", "[URL]"]
        self.tokenizer.add_tokens(special_tokens)

        self.model.resize_token_embeddings(len(self.tokenizer))

    def compute_perplexity(self, eval_pred: tuple) -> dict:
        """
        Computes perplexity, which is a measure of how well a probability distribution or probability model predicts a sample.


        Parameters
        ----------
        eval_pred : tuple
            Tuple containing logits and labels

        Returns
        -------
        dict
            The perplexity of the model
        """
        logits, labels = eval_pred
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        perplexity = torch.exp(loss)
        return {"perplexity": perplexity.item()}

    def compute_accuracy(self, eval_pred: tuple) -> dict:
        """
        Computes the accuracy of the model.

        Parameters
        ----------
        eval_pred : tuple
            Tuple containing logits and labels

        Returns
        -------
        dict
            The accuracy of the model
        """
        predictions, labels = eval_pred
        # Flatten the output
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [p for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [l for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        # Calculate accuracy using the true predictions and labels
        true_predictions = [item for sublist in true_predictions for item in sublist]
        true_labels = [item for sublist in true_labels for item in sublist]

        accuracy = accuracy_score(true_labels, true_predictions)
        return {
            "accuracy": accuracy,
        }

    def encode(self, data):
        return self.tokenizer(data["text"], truncation=True, padding="max_length")

    def train(self, data: Dataset, validation: Dataset, batch_size: int = 8):
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
            num_train_epochs=10,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            evaluation_strategy="steps",
            save_steps=10_000,
            eval_steps=500,
            save_total_limit=2,
            fp16=True,
            load_best_model_at_end=True,
            early_stopping_patience=3,
            metric_for_best_model="accuracy",
            greater_is_better=False,  # Lower perplexity indicates better performance
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=data,
            eval_dataset=val,
            data_collator=data_collator,
            compute_metrics=self.compute_accuracy,
        )

        # Train
        trainer.train()

        # Save the model
        trainer.save_model("output/FinTwitBERT")

        # Save the tokenizer
        self.tokenizer.save_pretrained("output/FinTwitBERT")
