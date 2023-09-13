#!/usr/bin/env python3

# > Imports
# Standard library
import logging
import os

# Third party
from data import load_preprocessed_data
from model import FinTwitBERT

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
    df = load_preprocessed_data()
    logging.info("Dataset loaded and preprocessed")

    # Train the model
    logging.info("Training the model")
    model = FinTwitBERT()
    model.train(df)
    logging.info("Model trained and saved to output/FinTwitBERT")
