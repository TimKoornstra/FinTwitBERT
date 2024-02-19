#!/usr/bin/env python3

# > Imports
import logging
import os

import torch

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

    # Display CUDA info
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    for i, device in enumerate(devices):
        logging.info(f"CUDA Device {i}: {device}")

    # Train the model
    logging.info("Training the model")
    model = FinTwitBERT()
    model.train()
    logging.info("Model trained and saved to output folder")
