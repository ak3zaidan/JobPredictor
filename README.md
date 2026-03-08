# Job Predictor — Experience & Salary Classification

A fine-tuned **DeBERTa-v3-base** model that predicts **expected years of experience** and **expected annual salary (USD)** from job posting text. Trained on ~750k real LinkedIn job postings.

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/akzaidan/JobPredictor2)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-orange)](https://huggingface.co/datasets/akzaidan/Job_Data_Parser)

## Performance

| Metric | Value |
|--------|-------|
| **Experience within 1 year** | **95%** of test examples |
| **Salary within $10k** | **80%** of test examples |
| Test set size | ~10,000 held-out examples |

The test set was **not used during training** — it serves purely as a benchmark.

<img width="903" height="730" alt="Screenshot 2026-03-07 at 5 50 26 PM" src="https://github.com/user-attachments/assets/bdac5d7f-c149-4854-839a-a8ce86a4ec26" />

## Quick Start

### Install

```bash
pip install torch transformers huggingface_hub pandas
```

### Inference

```python
from model.model import load_predictor, predict_batch

# Load model and tokenizer from Hugging Face
model, tokenizer, norm_stats = load_predictor()

# Job text in the expected format
text = "[LOCATION]: United States California [TITLE]: Senior Software Engineer [DESC]: We are seeking an experienced engineer with 5+ years of Python and cloud experience..."

predictions = predict_batch([text], model, tokenizer, norm_stats)
print(predictions[0])
# {'experience_years': 5, 'expected_salary': 145000}
```

## Input Format

Job postings must follow this structure:

```
[LOCATION]: <Remote | United States (State) | Country> [TITLE]: <job title> [DESC]: <job description>
```

Example:

```
[LOCATION]: United Kingdom [TITLE]: Senior Data Engineer [DESC]: Join our team to build scalable data pipelines. 5+ years experience with Spark and Python required. Competitive salary...
```

## Model & Dataset

| Resource | Link |
|----------|------|
| **Model** | [akzaidan/JobPredictor2](https://huggingface.co/akzaidan/JobPredictor2) |
| **Dataset** | [akzaidan/Job_Data_Parser](https://huggingface.co/datasets/akzaidan/Job_Data_Parser) |

Both are **open source** and available on Hugging Face.

## Training Details

| Setting | Value |
|---------|-------|
| Base model | `microsoft/deberta-v3-base` |
| Training data | ~750k job postings (LinkedIn) |
| Hardware | 4× NVIDIA RTX 5090 |
| Training time | ~5 hours |
| Epochs | 3 (best checkpoint) |
| Batch size | 32 |
| Learning rate | 5e-6 |
| Max sequence length | 512 |
| Optimizer | AdamW (weight decay 0.05) |
| Loss | MSE (experience + salary) |

<img width="1468" height="392" alt="Screenshot 2026-03-06 at 3 29 18 PM" src="https://github.com/user-attachments/assets/3b38579a-b77a-4a79-ab7b-91598830e0d8" />

<img width="315" height="274" alt="Screenshot 2026-03-07 at 5 28 55 PM" src="https://github.com/user-attachments/assets/80c9bf66-867e-422a-8ab5-584120b3cda7" />

## Architecture

- **Encoder**: DeBERTa-v3-base (12 layers, 768 hidden, 184M params)
- **Heads**: Two regression heads (768 → 256 → 1) with GELU and dropout (0.2)
- **Normalization**:
  - Experience: z-score
  - Salary: log1p then z-score

```python
# Denormalization at inference:
real_exp   = pred * std_exp   + mean_exp
real_salary = expm1(pred * std_salary + mean_salary)
```

## Project Structure

```
job_parser/
├── model/
│   ├── model.py      # JobRegressorModel, load_predictor, predict_batch
│   ├── config.py    # HF_REPO_ID, BASE_MODEL, MAX_LENGTH, etc.
│   └── predict.py   # CLI for batch prediction
├── model.ipynb      # Full training notebook
└── README.md
```

## Run Batch Prediction (CLI)

Ensure `test.parquet` (or your data file) is available, then:

```bash
python -m model.predict
```

This loads the model from Hugging Face and runs inference on a sample of rows.

## Requirements

- Python 3.8+
- PyTorch
- transformers
- huggingface_hub
- pandas

## License

Dataset and model are available under permissive licenses. See the respective Hugging Face pages for details.

## Data Distribution

<img width="316" height="632" alt="Screenshot 2026-03-04 at 12 53 54 PM" src="https://github.com/user-attachments/assets/553d16ff-7339-47d2-b22f-a5fd649354d3" />
<img width="229" height="360" alt="Screenshot 2026-03-04 at 12 53 43 PM" src="https://github.com/user-attachments/assets/9fef6815-174b-48fe-9988-d3dff46de208" />

