import sys
from pathlib import Path

# Add project root so 'model' is importable as a package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.model import load_predictor, predict_all
import pandas as pd

def main():
    path = Path(__file__).parent / "test.parquet"
    df = pd.read_parquet(path)

    # Sample 50 random rows
    sample = df.sample(n=50, random_state=42)
    texts = sample["text"].astype(str).tolist()
    expected_exp = sample["expected_experience_years"].tolist()
    expected_salary = sample["expected_salary"].tolist()

    print("Loading model...")
    model, tokenizer, norm_stats = load_predictor()

    print("Running batch prediction on 50 rows...")
    results = predict_all(texts, model, tokenizer, norm_stats)
    print()

    print("=" * 100)
    print("RESULTS (predicted vs expected | first 80 chars of text)")
    print("=" * 100)
    for i, pred in enumerate(results):
        exp_pred = pred["experience_years"]
        exp_actual = expected_exp[i]
        sal_pred = pred["expected_salary"]
        sal_actual = expected_salary[i]
        text_preview = texts[i][:80]
        print(
            f"exp: {exp_pred} vs {exp_actual} | "
            f"salary: ${sal_pred:,} vs ${sal_actual:,} | "
            f"{text_preview}"
        )

if __name__ == "__main__":
    main()
