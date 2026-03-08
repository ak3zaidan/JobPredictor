import torch

HF_REPO_ID = "akzaidan/JobPredictor2"
BASE_MODEL = "microsoft/deberta-v3-base"
MAX_LENGTH = 512
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
