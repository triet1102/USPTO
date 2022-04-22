import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchmetrics import MeanSquaredError
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from tqdm.auto import tqdm
import argparse
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import Callback
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from azureml.core import Run
import sys


def get_program_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model-name',
        type=str,
        default="bert-base-uncased",
        help='Path to a local model or name of Huggingface model'
    )
    parser.add_argument(
        '--val-size',
        type=float,
        default=0.25,
        help='Fraction of data for validation'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=192,
        help='Max sequence length for tokenizer'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size'
    )
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=6,
        help='Number of epoch for training'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=2e-5,
        help='Learning rate'
    )
    parser.add_argument(
        '--max-learning-rate',
        type=float,
        default=1e-3,
        help='Max learning rate'
    )
    parser.add_argument(
        '--accumulate',
        type=int,
        default=1,
        help='Accumulate'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=3,
        help='Patience'
    )
    parser.add_argument(
        '--monitor',
        type=str,
        default="val_loss",
        help='Monitor'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Seed'
    )
    parser.add_argument(
        '--debug',
        type=bool,
        default=False,
        help='Debug mode'
    )
    parser.add_argument(
        '--save-model',
        type=bool,
        default=False,
        help='Whether to save the trained model'
    )

    arguments, _ = parser.parse_known_args()
    return arguments


def main(model_name: str,
         val_size: float,
         max_length: int,
         batch_size: int,
         num_epoch: int,
         learning_rate: float,
         max_learning_rate: float,
         accumulate: int,
         patience: int,
         monitor: str,
         seed: int,
         debug: bool,
         save_model: bool):
    pass

if __name__ == "__main__":
    args = get_program_arguments()
    main(model_name=args.model_name,
         val_size=args.val_size,
         max_length=args.max_length,
         batch_size=args.batch_size,
         num_epoch=args.num_epoch,
         learning_rate=args.learning_rate,
         max_learning_rate=args.max_learning_rate,
         accumulate=args.accumulate,
         patience=args.patience,
         monitor=args.monitor,
         seed=args.seed,
         debug=args.debug,
         save_model=args.save_model)

    sys.exit(0)
