from .config import HF_REPO_ID, BASE_MODEL, MAX_LENGTH, DEVICE, BATCH_SIZE
from transformers import DebertaV2Tokenizer, AutoModel
from huggingface_hub import hf_hub_download
import torch.nn as nn
import numpy as np
import torch
import json

class JobRegressorModel(nn.Module):
    
    def __init__(self, model_name, dropout=0.2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head_experience = nn.Sequential(
            nn.Linear(hidden_size, 256), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(256, 1)
        )
        self.head_salary = nn.Sequential(
            nn.Linear(hidden_size, 256), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        cls_embedding = self.dropout(outputs.last_hidden_state[:, 0, :]).float()
        exp = self.head_experience(cls_embedding).squeeze(-1)
        salary = self.head_salary(cls_embedding).squeeze(-1)
        return exp, salary


def _denorm(val, col, norm_stats):
    return val * norm_stats[col]["std"] + norm_stats[col]["mean"]


def _denorm_salary(val, norm_stats):
    return np.expm1(_denorm(val, "expected_salary", norm_stats))


def _safe_int(val, default=0):
    """Convert to int, returning default if NaN or inf."""
    v = float(val)
    if np.isnan(v) or np.isinf(v):
        return default
    return max(0, round(v))


def load_predictor():
    """Load model, tokenizer, and normalization stats from HuggingFace."""
    print("Downloading model from HuggingFace...")
    weights_path = hf_hub_download(repo_id=HF_REPO_ID, filename="pytorch_model.bin")
    norm_stats_path = hf_hub_download(repo_id=HF_REPO_ID, filename="norm_stats.json")

    with open(norm_stats_path) as f:
        norm_stats = json.load(f)

    tokenizer = DebertaV2Tokenizer.from_pretrained(HF_REPO_ID)
    model = JobRegressorModel(BASE_MODEL)
    state_dict = torch.load(weights_path, map_location=DEVICE, weights_only=False)
    # Strip "module." prefix if checkpoint was saved from DataParallel/DDP
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    print("Model loaded successfully!")
    return model, tokenizer, norm_stats


def predict_batch(texts, model, tokenizer, norm_stats):
    """texts: list of raw job text strings (e.g. from parquet 'text' column)."""
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    token_type_ids = encoded.get("token_type_ids")
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(DEVICE)

    with torch.no_grad():
        pred_exp, pred_salary = model(
            encoded["input_ids"].to(DEVICE),
            encoded["attention_mask"].to(DEVICE),
            token_type_ids,
        )

    exp_years = pred_exp.cpu().numpy()
    salaries = pred_salary.cpu().numpy()

    return [
        {
            "experience_years": _safe_int(_denorm(exp_years[i], "expected_experience_years", norm_stats)),
            "expected_salary": _safe_int(_denorm_salary(salaries[i], norm_stats)),
        }
        for i in range(len(texts))
    ]


def predict_all(texts, model, tokenizer, norm_stats, batch_size=BATCH_SIZE):
    """Runs any number of text strings through the model in batches."""
    all_results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        all_results.extend(predict_batch(batch, model, tokenizer, norm_stats))
        print(f"  Processed {min(i + batch_size, len(texts))}/{len(texts)}", end="\r")
    return all_results
