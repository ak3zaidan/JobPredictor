import torch
import os

HF_REPO_ID = "akzaidan/JobPredictor2"
BASE_MODEL = "microsoft/deberta-v3-base"
MAX_LENGTH = 512
BATCH_SIZE = 4
DEVICE = torch.device("cpu")
