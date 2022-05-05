from src.model import PhraseSimilarityModelImpl, PhraseSimilarityModel
from src.sim_dataset import PhraseSimilarityTestset
from transformers import AutoTokenizer, AutoModel
import sys
import pandas as pd
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
import torch.nn as nn
from torchmetrics import MeanSquaredError
import yaml
import os


def main(model_ckpt, test_data_path, yaml_path):
    """Predict and fill submission.csv
    """
    os.chdir("/home/azureuser/cloudfiles/code/Users/triet.tran/USPTO")
    with open(yaml_path, 'r') as f:
        hparams = yaml.safe_load(f)
    criterion = nn.HuberLoss(reduction='mean', delta=1.0)
    metric = MeanSquaredError()
    model = PhraseSimilarityModelImpl("bert-base-uncased")

    driver = PhraseSimilarityModel(model=model,
                                   lr=hparams["learning_rate"],
                                   criterion=criterion,
                                   metric=metric)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    test_data = pd.read_csv(test_data_path)
    test_dataset = PhraseSimilarityTestset(test_data, tokenizer, max_length=64)
    test_dataloader = DataLoader(
        test_dataset, batch_size=32, shuffle=False)

    trainer = Trainer()
    predictions = trainer.predict(
        driver, dataloaders=test_dataloader, ckpt_path=model_ckpt)
    preds = []
    for batch in predictions:
        preds += batch.squeeze(1).tolist()

    submission_csv = pd.read_csv("data/sample_submission.csv")
    submission_csv["score"] = preds
    print(submission_csv.head())

    # bert_base_model = AutoModel.from_pretrained("bert-base-uncased")
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # bert_base_model.save_pretrained("saved_models/bert_base_uncased")
    # tokenizer.save_pretrained("saved_models/bert_base_uncased")


if __name__ == "__main__":
    MODEL_CKPT = r"saved_models/outputs/last.ckpt"
    TEST_DATA_PATH = r"data/test.csv"
    YAML_PATH = r"saved_models/outputs/bert-base-uncased_log/version_0/hparams.yaml"
    main(model_ckpt=MODEL_CKPT,
         test_data_path=TEST_DATA_PATH,
         yaml_path=YAML_PATH)
    sys.exit(0)
