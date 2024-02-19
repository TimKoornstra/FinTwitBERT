#!/usr/bin/env python3

# > Imports
import logging
import os

import torch
from dotenv import load_dotenv
import wandb

from model import FinTwitBERT
import eval.finetune

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Create folders if they don't exist
    for folder in ["checkpoints", "output", "data"]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            logging.info(f"Created {folder} folder")

    # Display CUDA info
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    for i, device in enumerate(devices):
        logging.info(f"CUDA Device {i}: {device}")

    # Train the model
    logging.info("Training the model")
    # model = FinTwitBERT()
    # model.train()
    logging.info("Model trained and saved to output folder")

    # Remove this later
    # Load the .env file
    load_dotenv(dotenv_path="wandb.env")

    # Read the API key from the environment variable
    os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")

    # set the wandb project where this run will be logged
    os.environ["WANDB_PROJECT"] = "FinTwitBERT-sentiment"

    # save your trained model checkpoint to wandb
    os.environ["WANDB_LOG_MODEL"] = "true"

    # turn off watch to log faster
    os.environ["WANDB_WATCH"] = "false"

    wandb.init()

    evaluate = eval.finetune.Evaluate()
    evaluate.evaluate_model()
