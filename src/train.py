import pandas as pd
import numpy as np
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
import os
from sim_dataset import PhraseSimilarityDataset
from sim_dataset import PhraseSimilarityTestset
from modeling import PhraseSimilarityModelImpl
from modeling import PhraseSimilarityModel
import matplotlib.pyplot as plt


def get_program_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--devices',
        type=int,
        default=4,
        help='Number of device'
    )
    parser.add_argument(
        '--num-nodes',
        type=int,
        default=2,
        help='Number of node'
    )

    parser.add_argument(
        '--data-folder',
        type=str,
        required=True,
        help='Path to data folder'
    )
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
        default=64,
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
        default=True,
        help='Whether to save the trained model'
    )

    arguments, _ = parser.parse_known_args()
    return arguments


def main(devices: int,
         num_nodes: int,
         data_folder: str,
         model_name: str,
         val_size: float,
         max_length: int,
         batch_size: int,
         num_epoch: int,
         learning_rate: float,
         accumulate: int,
         patience: int,
         monitor: str,
         seed: int,
         debug: bool,
         args: argparse.Namespace):
    """Training script

    :param devices: Number of GPUs per node
    :param num_nodes: Number of node
    :param data_folder: Mounted path of data folder from datastore to compute cluster
    :param model_name: Name or path of a pretrained model
    :param val_size: Percentage of validation
    :param max_length: Max sequence length for tokenizer
    :param batch_size: Batch size
    :param num_epoch: Number of epoch
    :param learning_rate: Learning rate
    :param accumulate: Accumulate
    :param patience: Patience
    :param monitor: Monitor
    :param seed: Seed
    :param debug: Debug
    :param args: Arguments
    """
    # Get PyTorch environment variables
    # world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    # local_rank = int(os.environ["LOCAL_RANK"])

    # Setup path
    train_path = os.path.join(data_folder, "train.csv")
    context_path = os.path.join(data_folder, "CPC_context.csv")
    test_path = os.path.join(data_folder, "test.csv")

    # Read and merge data
    context_data = pd.read_csv(context_path)
    context_data = context_data.rename(columns={"code": "context"})

    train_data = pd.read_csv(train_path)
    train_data = pd.merge(
        train_data, context_data[["context", "title"]], on="context", how="left")
    print(train_data.head())

    test_data = pd.read_csv(test_path)
    test_data = pd.merge(
        test_data, context_data[["context", "title"]], on="context", how="left")

    print(f"DEBUG: {debug}")
    if debug == True:
        train_data = train_data.iloc[:2000]

    scores = train_data.score.values
    train_data.drop("score", inplace=True, axis=1)
    train_data, val_data, train_labels, val_labels = train_test_split(train_data,
                                                                      scores,
                                                                      stratify=scores,
                                                                      test_size=val_size,
                                                                      random_state=seed)
    train_data["score"] = train_labels
    val_data["score"] = val_labels

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = PhraseSimilarityDataset(train_data, tokenizer, max_length)
    val_dataset = PhraseSimilarityDataset(val_data, tokenizer, max_length)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)

    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch * num_epoch
    # Use 10% of the total training steps for warmup learning rate
    num_warmups = int(total_steps * 0.1)
    num_decreases = total_steps - num_warmups
    total_devices = num_nodes * devices
    print(f"steps_per_epoch: {steps_per_epoch}")
    print(f"total_steps: {total_steps}")
    print(f"num_warmups: {num_warmups}")
    print(f"num_decreases: {num_decreases}")
    print(f"total_devices: {total_devices}")

    os.makedirs("./outputs", exist_ok=True)
    logger = CSVLogger(save_dir='./outputs',
                       name=model_name.split('/')[-1]+'_log')
    logger.log_hyperparams(args.__dict__)

    # Settup callbacks
    checkpoint_callback = ModelCheckpoint(dirpath="./outputs",
                                          monitor=monitor,
                                          save_top_k=1,
                                          save_last=True,
                                          save_weights_only=True,
                                          filename="{epoch:02d}-{val_loss:.4f}-{val_error:.4f}",
                                          verbose=False,
                                          mode="min")

    early_stop_callback = EarlyStopping(monitor=monitor,
                                        patience=patience,
                                        verbose=False,
                                        mode="min")

    # Set up driver for training
    model = PhraseSimilarityModelImpl(model_name)
    criterion = nn.HuberLoss(reduction='mean', delta=1.0)
    metric = MeanSquaredError()
    driver = PhraseSimilarityModel(model=model,
                                   lr=learning_rate,
                                   num_warmups=num_warmups,
                                   num_decreases=num_decreases,
                                   total_devices=total_devices,
                                   criterion=criterion,
                                   metric=metric)

    # Set up trainer
    trainer = Trainer(
        accelerator="gpu",
        devices=devices,
        num_nodes=num_nodes,
        strategy="ddp",
        max_epochs=num_epoch,
        accumulate_grad_batches=accumulate,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        weights_summary='top',
    )

    # Train our driver
    trainer.fit(driver, train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)

    # Plot metrics and learning rate
    if rank == 0:
        metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")

        # Plot Error
        train_acc = metrics["train_error"].dropna().reset_index(drop=True)
        val_acc = metrics["val_error"].dropna().reset_index(drop=True)

        fig = plt.figure(figsize=(7, 6))
        plt.grid(True)
        plt.plot(train_acc, color="r", marker="o", label="train/error")
        plt.plot(val_acc, color="b", marker="x", label="val/error")
        plt.xlabel("Epoch", fontsize=24)
        plt.ylabel("Error", fontsize=24)
        plt.legend(loc="lower right", fontsize=18)
        plt.savefig(f"{trainer.logger.log_dir}/acc.png")

        # Plot Loss
        train_loss = metrics["train_loss"].dropna().reset_index(drop=True)
        val_loss = metrics["val_loss"].dropna().reset_index(drop=True)

        fig = plt.figure(figsize=(7, 6))
        plt.grid(True)
        plt.plot(train_loss, color="r", marker="o", label="train/loss")
        plt.plot(val_loss, color="b", marker="x", label="val/loss")
        plt.xlabel("Epoch", fontsize=24)
        plt.ylabel("Loss", fontsize=24)
        plt.legend(loc="upper right", fontsize=18)
        plt.savefig(f"{trainer.logger.log_dir}/loss.png")

        # Plot Learning Rate
        fig = plt.figure(figsize=(7, 6))
        plt.grid(True)
        plt.plot(driver.lrs, color="g", marker="o", label='learning rate')
        plt.ylabel('LR', fontsize=24)
        plt.xlabel('Epoch', fontsize=24)
        plt.legend(loc='upper right', fontsize=18)
        plt.savefig(f'{trainer.logger.log_dir}/sch_lr.png')

        plt.show()


if __name__ == "__main__":
    args = get_program_arguments()
    print(f"DEBUG mode: {args.debug}")
    main(devices=args.devices,
         num_nodes=args.num_nodes,
         data_folder=args.data_folder,
         model_name=args.model_name,
         val_size=args.val_size,
         max_length=args.max_length,
         batch_size=args.batch_size,
         num_epoch=args.num_epoch,
         learning_rate=args.learning_rate,
         accumulate=args.accumulate,
         patience=args.patience,
         monitor=args.monitor,
         seed=args.seed,
         debug=args.debug,
         args=args)

    sys.exit(0)
