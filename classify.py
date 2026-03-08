"""
Classify job postings: fill missing expected_experience_years, expected_salary using LLM.
Loads parquet, processes rows where any of these is -1, uses YEARS_ONLY or YEARS_AND_PRICE based
on whether expected_salary is already valid. Saves parquet every 1000 rows updated.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from llm_clients.caller import LLMCaller
from pathlib import Path
import pandas as pd
import random
import signal
import time
import json
import re
import os

from config import (
    CLASSIFY_FILE,
    SYSTEM_MSG_YEARS_AND_PRICE,
    SYSTEM_MSG_YEARS_ONLY,
    WORKER_COUNT,
    MODEL_CONFIGS,
    MODEL_WEIGHTS,
    SAVE_EVERY
)

# Shared state for Ctrl+C handler to save progress and shutdown executor
_shared = {"df": None, "executor": None}


def _handle_sigint(sig, frame):
    print("\nInterrupt received. Saving progress...")
    if _shared["df"] is not None:
        save_parquet(CLASSIFY_FILE, _shared["df"])
        print(f"Saved to {CLASSIFY_FILE}")
    if _shared["executor"] is not None:
        try:
            _shared["executor"].shutdown(wait=False, cancel_futures=True)
        except TypeError:
            _shared["executor"].shutdown(wait=False)
    os._exit(1)


signal.signal(signal.SIGINT, _handle_sigint)


def is_missing(val) -> bool:
    """Value is missing if -1 or NaN."""
    return pd.isna(val) or val == -1


def has_valid_pay(row: pd.Series) -> bool:
    """Row has valid expected_salary."""
    return not is_missing(row.get("expected_salary"))


def parse_years_only(text: str) -> int | None:
    """Extract a single integer year (0-15) from response."""
    if not text or not isinstance(text, str):
        return None
    # Look for first integer in 0-15 range
    matches = re.findall(r"\b(\d{1,2})\b", text.strip())
    for m in matches:
        n = int(m)
        if 0 <= n <= 20:
            return n
    return None


def parse_years_and_price(text: str) -> dict | None:
    """Parse JSON with years, expected_salary."""
    if not text or not isinstance(text, str):
        return None
    # Strip markdown code blocks if present
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```\w*\n?", "", s)
        s = re.sub(r"\n?```\s*$", "", s)
    try:
        obj = json.loads(s)
        years = obj.get("years")
        salary = obj.get("expected_salary")
        if years is not None and isinstance(years, (int, float)):
            y = int(years)
            if 0 <= y <= 20:
                result = {"years": y}
                if salary is not None and not pd.isna(salary):
                    result["expected_salary"] = int(salary)
                return result
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    return None


def load_parquet(path: str) -> pd.DataFrame:
    """Load parquet file."""
    return pd.read_parquet(path)


def save_parquet(path: str, df: pd.DataFrame) -> None:
    """Save DataFrame to parquet."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def classify_one(caller: LLMCaller, idx: int, row: pd.Series) -> tuple[int, dict | None]:
    """
    Classify a single row. Returns (idx, result dict or None).
    result: {"years": int} for years-only, or {"years": int, "expected_salary": int}
    """
    model_config = random.choices(MODEL_CONFIGS, weights=MODEL_WEIGHTS, k=1)[0]
    text = (str(row.get("text", "") or ""))[:3500]

    use_years_only = has_valid_pay(row)
    system_msg = SYSTEM_MSG_YEARS_ONLY if use_years_only else SYSTEM_MSG_YEARS_AND_PRICE

    for attempt in range(4):
        try:
            response = caller.call_model(
                model_configs=[model_config],
                prompt=text,
                system_message=system_msg,
                temperature=0,
                max_tokens=256,
            )
            if use_years_only:
                years = parse_years_only(response)
                return (idx, {"years": years}) if years is not None else (idx, None)
            else:
                parsed = parse_years_and_price(response)
                return (idx, parsed) if parsed else (idx, None)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str and attempt < 3:
                time.sleep(5)
                continue
            print(f"  Warning: idx={idx} failed ({model_config[1].value}): {e}")
            return (idx, None)
    return (idx, None)


def main():
    print(f"Loading {CLASSIFY_FILE}...")
    df = load_parquet(CLASSIFY_FILE)
    _shared["df"] = df

    # Rows needing classification: any of expected_experience_years, expected_salary is -1 or NaN
    needs_work = (
        (df["expected_experience_years"] == -1)
        | df["expected_experience_years"].isna()
        | (df["expected_salary"] == -1)
        | df["expected_salary"].isna()
    )
    indices = df.index[needs_work].tolist()

    total = len(indices)
    print(f"Found {total} rows to classify (of {len(df)} total)")

    if total == 0:
        print("Nothing to classify.")
        return

    caller = LLMCaller()
    updated_count = 0
    failed = []
    executor = ThreadPoolExecutor(max_workers=WORKER_COUNT)
    _shared["executor"] = executor

    try:
        futures = {
            executor.submit(classify_one, caller, idx, df.loc[idx]): idx
            for idx in indices
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                _, result = future.result()
                if result is not None:
                    df.loc[idx, "expected_experience_years"] = result["years"]
                    if "expected_salary" in result:
                        df.loc[idx, "expected_salary"] = result["expected_salary"]
                    updated_count += 1
                    if updated_count % SAVE_EVERY == 0:
                        save_parquet(CLASSIFY_FILE, df)
                        print(f"  Saved after {updated_count} rows updated")
                else:
                    failed.append(idx)
            except Exception as e:
                print(f"  Warning: idx={idx} exception: {e}")
                failed.append(idx)
    finally:
        _shared["executor"] = None
        executor.shutdown(wait=True)

    if failed:
        df.drop(index=failed, inplace=True)
    save_parquet(CLASSIFY_FILE, df)
    print(f"Done. Updated {updated_count} rows. Removed {len(failed)} failed rows.")


if __name__ == "__main__":
    main()
