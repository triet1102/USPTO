from src.model import PhraseSimilarityModelImpl, PhraseSimilarityModel
from src.sim_dataset import PhraseSimilarityTestset
from transformers import AutoTokenizer
import sys
import pandas as pd
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
import torch.nn as nn
from torchmetrics import MeanSquaredError
import yaml


def main(model_ckpt, test_data_path, yaml_path):
    """Predict and fill submission.csv
    """
    with open(yaml_path, 'r') as f:
        hparams = yaml.safe_load(f)
    criterion = nn.HuberLoss(reduction='mean', delta=1.0)
    metric = MeanSquaredError()
    model = PhraseSimilarityModelImpl("bert-base-uncased")
    print("OK1")
    # driver = PhraseSimilarityModel(model, lr, criterion, metric)
    print("OK2")

    driver = PhraseSimilarityModel.load_from_checkpoint(model_ckpt,
                                         model=model,
                                         lr=hparams["learning_rate"],
                                         criterion=criterion,
                                         metric=metric)
    print("OK3")

    # driver = PhraseSimilarityModel.load_from_checkpoint(
    #     model_ckpt)

    # print("ok")
    # tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    # test_data = pd.read_csv(test_data_path)
    # test_dataset = PhraseSimilarityTestset(test_data, tokenizer, max_length=64)
    # test_dataloader = DataLoader(
    #     test_dataset, batch_size=32, shuffle=False)


if __name__ == "__main__":
    MODEL_CKPT = r"saved_models/outputs/last.ckpt"
    TEST_DATA_PATH = r"data/test.csv"
    YAML_PATH = r"saved_models/outputs/bert-base-uncased_log/version_0/hparams.yaml"
    main(model_ckpt=MODEL_CKPT,
         test_data_path=TEST_DATA_PATH,
         yaml_path=YAML_PATH)
    sys.exit(0)
