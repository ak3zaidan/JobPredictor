from llm_clients.caller import LLMProvider, LLMModel

WORKER_COUNT = 50
SAVE_EVERY = 50000  # Save parquet after this many rows updated
CLASSIFY_FILE = "train_data/val.parquet"
MODEL_CONFIGS = [
    (LLMProvider.GROK, LLMModel.GROK_4_FAST_NON_REASONING),
    (LLMProvider.OPENAI, LLMModel.GPT_4O_MINI),
]
MODEL_WEIGHTS = [0.25, 0.75]  # Grok 25%, OpenAI 75%

SYSTEM_MSG_YEARS_ONLY = """Predict the expected years of experience for the job posting.

Rules:
- Explicit year mentions take priority over inferred seniority
- If ambiguous, guess the most likely number of years based on the job posting.
- Output a single integer number of years between 0 and 20.

Respond with ONLY the predicted experience years. No other text."""

SYSTEM_MSG_YEARS_AND_PRICE = """You are a compensation analyst. Given a job posting, predict the required experience and expected salary.

Output ONLY a valid JSON object with exactly these fields:
- "years": integer, minimum years of experience required between 0 and 20
- "expected_salary": integer, expected annual salary in USD

Example output:
{"years": 3, "expected_salary": 105000}

No markdown, no explanation, no extra fields. JSON only."""
