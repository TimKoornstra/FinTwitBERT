def calculate_steps(batch_size, base_batch_size=64, base_steps=500):
    return (base_batch_size * base_steps) // batch_size


# Used for pretraining and finetuning
base_args = {
    "output_dir": "checkpoints/",
    "overwrite_output_dir": True,
    "fp16": True,
    "load_best_model_at_end": True,
    "save_safetensors": True,
    "save_total_limit": 2,
    "report_to": "wandb",
}

pretraining_args = {
    "num_train_epochs": 10,
    "per_device_train_batch_size": 64,
    "per_device_eval_batch_size": 64,
    "evaluation_strategy": "steps",
    "save_steps": 10_000,
    "eval_steps": 10_000,
    "learning_rate": 2e-5,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
    "warmup_ratio": 0.2,
}

finetuning_args = {
    "num_train_epochs": 20,  # FinBERT uses 6
    "per_device_train_batch_size": 64,
    "per_device_eval_batch_size": 64,
    "evaluation_strategy": "steps",
    "save_steps": calculate_steps(64),
    "eval_steps": calculate_steps(64),
    "logging_steps": calculate_steps(64),
    "learning_rate": 2e-4,  # FinBERT uses 5e-5 to 2e-5
    "metric_for_best_model": "accuracy",
    "greater_is_better": True,  # Higher accuracy is better
    "warmup_ratio": 0.2,  # FinBERT uses 0.2
    "weight_decay": 0.01,  # FinBERT uses 0.01
    # gradient_accumulation_steps=1,  # FinBERT uses 1
    "lr_scheduler_type": "cosine_with_restarts",
}
