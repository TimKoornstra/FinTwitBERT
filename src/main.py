#!/usr/bin/env python3

# > Imports
# Standard library
import logging
import os

# Third party
import torch

# Local
from data import load_pretraining_data, load_finetuning_data, kfold_pretraining_data
from model import FinTwitBERT
from eval.pretrain import Evaluate

KFOLD = False
PRETRAIN = False
FINETUNE = True


def do_kfold():
    df, val = kfold_pretraining_data()
    logging.info("Dataset loaded and preprocessed")

    # Display CUDA info
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    for i, device in enumerate(devices):
        logging.info(f"CUDA Device {i}: {device}")

    # Train the model
    for train, val in zip(df, val):
        logging.info("Training the model")
        model = FinTwitBERT()
        model.train(train, val)
        logging.info("Model trained and saved to output/FinTwitBERT")

        # Evaluate the new model
        logging.info("Evaluating the model")
        evaluate = Evaluate()
        evaluate.calculate_perplexity()
        evaluate.calculate_masked_examples()
        logging.info("Model perplexity calculated")


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

    # Load and preprocess the dataset
    logging.info("Loading and preprocessing the dataset")
    if KFOLD:
        do_kfold()

    if PRETRAIN:
        df, val = load_pretraining_data()
    if FINETUNE:
        df, val = load_finetuning_data()
    logging.info("Dataset loaded and preprocessed")

    # Display CUDA info
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    for i, device in enumerate(devices):
        logging.info(f"CUDA Device {i}: {device}")

    # Train the model
    logging.info("Training the model")
    model = FinTwitBERT(mode="pretrain" if PRETRAIN else "finetune")
    model.train(df, val)
    logging.info("Model trained and saved to output/FinTwitBERT")

    # Evaluate the new model
    logging.info("Evaluating the model")
    evaluate = Evaluate()
    evaluate.calculate_perplexity()
    evaluate.calculate_masked_examples()
    logging.info("Model perplexity calculated")
