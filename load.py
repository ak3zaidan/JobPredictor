"""
Load jobs from parquet files, filter and clean, then create train/val/test datasets.
"""
from pathlib import Path
import pandas as pd
import unicodedata
import fasttext
import random
import json
import re

# New rows to add from raw_data
NEW_TRAIN_SIZE = 500_000
NEW_VAL_SIZE = 15_000

skip_substrings = [
    "cook",
    "chef",
    "line cook",
    "housekeeper",
    "barista",
    "delivery driver",
    "driver",
    "sandwich artist",
    "dishwasher",
    "restaurant manager",
    "cashier",
]

skip_jobs = [
    "Server",
    "Cook",
    "Bartender",
    "VENDEDOR",
    "Line Cook",
    "Housekeeper",
    "Barista",
    "Delivery Driver",
    "Driver",
    "Sandwich Artist",
    "Dishwasher",
    "Restaurant Manager",
    "Cashier",
    "Recepcionista",
    "Sous Chef",
    "Chef",
    "Cleaner",
    "Retail Sales Associate",
    "Chef de Partie",
    "Buyer",
    "Forklift Operator",
    "Kitchen Assistant",
    "Kitchen Porter",
    "Sandwich Artist ®",
    "Night Auditor",
    "Security Guard",
    "Executive Chef",
    "Shift Leader",
    "Cuisinier H/F",
    "Room Attendant",
    "Auxiliar de almacén",
    "Painter",
    "Prep Cook",
    "Host/Hostess",
    "Baker",
    "Merchandiser",
    "Estoquista",
    "Maintenance Assistant",
    "Reinigungskraft (m/w/d)",
    "Busser",
    "Kitchen Staff",
    "Warehouse Operative",
    "Custodian",
    "Commis Chef",
    "Housekeeping",
    "Cashier/Customer Service",
    "Restaurant Supervisor",
    "Waiter / Waitress",
    "Line Cook/Prep Cook",
    "Almacenista",
    "Concierge",
    "Food Runner",
    "Kitchen Manager",
    "General Laborer",
    "Assistant Restaurant Manager",
    "Cook",
    "Warehouse Worker",
    "Production Worker",
    "Front Desk Associate",
    "Maintenance Worker",
    "Brand Ambassador",
    "Phlebotomist",
    "Auxiliar de Logística",
    "Housekeeping Attendant",
    "Automotive Mechanic",
    "In Home Caregiver",
    "Laundry Attendant",
    "Janitor",
    "Steward",
    "Front of House Staff",
    "Bar Staff",
    "Retail Sales Assistant",
    "Hostess",
    "Commis de cuisine H/F",
    "Server/Bartender",
    "Auxiliaire de vie H/F",
    "Breakfast Attendant",
    "Bus Driver",
    "Ayudante de cocina",
    "Welder/Fabricator",
    "Assistant Teacher",
    "Bartender/Server",
    "Housekeeping Assistant",
    "Personal de limpieza",
    "Agent d'entretien H/F",
    "Cleaner (part-time)",
    "Kitchen Helper",
    "Front Office Assistant",
    "Kitchen Crew",
    "Warehouse Clerk",
    "Beauty Therapist",
    "Domestic Assistant",
    "Cleaning Technician",
    "Auxiliar administrativo/a",
    "CREW MEMBER",
    "Aide soignant H/F",
    "Chef de rang H/F",
    "Auxiliar de Expedição",
    "Auxiliar de Cozinha",
    "Food Service Supervisor",
    "Child caregiver - private home",
    "Home child care provider",
    "Food Server",
    "Restaurant Manager",
    "Shipping Clerk",
    "Dishwasher/Food Prep",
    "Server Assistant",
    "Store Assistant",
    "General Manager - Restaurant",
    "Kitchen Crew",
    "Porter",
    "Warehouse Operator",
    "Labourer",
    "Storekeeper",
    "Retail Associate",
    "Pastry Chef",
    "Barista/All Rounder",
    "Cocinero",
    "Cocinero/a",
    "Dietary Cook",
    "Groundskeeper",
    "Dog Groomer",
    "Spa Therapist",
    "Hair Stylist",
]

# ISO 3166-1 alpha-2 country code -> full name (common countries)
COUNTRY_NAMES = {
    "US": "United States", "CA": "Canada", "GB": "United Kingdom", "AU": "Australia",
    "DE": "Germany", "FR": "France", "IN": "India", "NL": "Netherlands", "IE": "Ireland",
    "ES": "Spain", "IT": "Italy", "BR": "Brazil", "MX": "Mexico", "JP": "Japan",
    "SG": "Singapore", "IL": "Israel", "PL": "Poland", "SE": "Sweden", "CH": "Switzerland",
    "BE": "Belgium", "AT": "Austria", "PT": "Portugal", "NZ": "New Zealand", "ZA": "South Africa",
    "NO": "Norway", "DK": "Denmark", "FI": "Finland", "RO": "Romania", "CZ": "Czech Republic",
    "HU": "Hungary", "GR": "Greece", "TR": "Turkey", "AE": "United Arab Emirates",
    "HK": "Hong Kong", "KR": "South Korea", "TW": "Taiwan", "PH": "Philippines",
    "AR": "Argentina", "CL": "Chile", "CO": "Colombia", "RU": "Russia", "UA": "Ukraine",
}

# US state/territory code -> full name
US_STATE_NAMES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire",
    "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York", "NC": "North Carolina",
    "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
    "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee",
    "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia",
}

def preprocess_text(text: str) -> str:
    """Clean and normalize text for model training."""
    if not text or not isinstance(text, str):
        return ""
    text = text.strip()
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = "".join(c for c in text if unicodedata.category(c)[0] != "C" or c in "\n\t")
    return text.strip()

def title_is_numeric_only(title: str) -> bool:
    """Return True if title contains only digits (and maybe whitespace)."""
    stripped = title.strip()
    if not stripped:
        return True
    return stripped.replace(" ", "").isdigit()

def is_english(text: str, model, threshold: float = 0.8) -> bool:
    text = text.replace("\n", " ").replace("\r", "")
    lang, prob = model.predict(text)
    return lang[0] == "__label__en" and prob[0] > threshold

def should_skip_title(title: str) -> bool:
    """Return True if title matches skip_jobs or contains any skip_substrings."""
    if title in skip_jobs:
        return True
    title_lower = title.lower()
    return any(sub in title_lower for sub in skip_substrings)

def main():
    raw_dir = Path("raw_data")
    parquet_files = sorted(raw_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {raw_dir}")

    random.seed(42)
    random.shuffle(parquet_files)

    print("Loading language model...")
    model = fasttext.load_model("text_model.bin")

    target_new = NEW_TRAIN_SIZE + NEW_VAL_SIZE
    clean_entries = []
    skipped_reasons = {"no_title_desc": 0, "title_numeric": 0, "not_english": 0, "skip_jobs": 0}

    for file_idx, parquet_path in enumerate(parquet_files):
        if len(clean_entries) >= target_new:
            break

        print(f"Loading {parquet_path.name} ({file_idx + 1}/{len(parquet_files)})...")
        df = pd.read_parquet(parquet_path)
        indices = list(range(len(df)))
        random.shuffle(indices)

        for i in indices:
            if len(clean_entries) >= target_new:
                break

            row = df.iloc[i]
            job_title = row.get("job_title", "") or ""
            job_description = row.get("description", "") or ""

            job_title = preprocess_text(str(job_title))
            job_description = preprocess_text(str(job_description))

            # Check 1: valid title and description (need both)
            if not job_title or not job_description:
                skipped_reasons["no_title_desc"] += 1
                continue

            # Check 2: title doesn't only contain numbers
            if title_is_numeric_only(job_title):
                skipped_reasons["title_numeric"] += 1
                continue

            # Check 3: title + description in English
            combined_text = f"{job_title} {job_description}"
            if not is_english(combined_text, model):
                skipped_reasons["not_english"] += 1
                continue

            # Check 4: skip jobs matching skip_jobs or skip_substrings
            if should_skip_title(job_title):
                skipped_reasons["skip_jobs"] += 1
                continue

            # Parse expected_salary from min/max annual salary
            pay_lower = row.get("min_annual_salary_usd")
            pay_upper = row.get("max_annual_salary_usd")
            if pd.isna(pay_lower):
                pay_lower = None
            if pd.isna(pay_upper):
                pay_upper = None
            if pay_lower is not None:
                pay_lower = int(pay_lower)
            if pay_upper is not None:
                pay_upper = int(pay_upper)

            # Compute expected_salary: average when both present, single value when one, -1 when neither
            if pay_lower is None and pay_upper is None:
                expected_salary = -1
            elif pay_lower is None:
                expected_salary = pay_upper
            elif pay_upper is None:
                expected_salary = pay_lower
            else:
                expected_salary = int((pay_lower + pay_upper) / 2)

            country_code = ""
            try:
                country_codes = json.loads(row.get("country_codes", "[]"))
                if country_codes:
                    country_code = country_codes[0]
            except Exception:
                pass

            is_remote = bool(row.get("remote", 0))
            state_code = row.get("state_code", "") or ""

            country_name = COUNTRY_NAMES.get(country_code, country_code) if country_code else ""
            state_name = US_STATE_NAMES.get(state_code.upper(), state_code) if state_code else ""

            if is_remote:
                text = f"[LOCATION]: Remote [TITLE]: {job_title} [DESC]: {job_description}"
            elif not country_code:
                text = f"[LOCATION]: Unknown [TITLE]: {job_title} [DESC]: {job_description}"
            elif country_code == "US":
                location = f"{country_name} {state_name}".strip() if state_name else country_name
                text = f"[LOCATION]: {location} [TITLE]: {job_title} [DESC]: {job_description}"
            else:
                text = f"[LOCATION]: {country_name} [TITLE]: {job_title} [DESC]: {job_description}"

            clean_entries.append({
                "text": text,
                "expected_salary": expected_salary,
                "expected_experience_years": -1,
            })

            if len(clean_entries) % 10_000 == 0 and len(clean_entries) > 0:
                print(f"  Collected {len(clean_entries)} / {target_new} clean entries...")

        del df  # free memory before loading next file

    print(f"\nCollected {len(clean_entries)} new clean entries")
    print(f"Skipped: no_title_desc={skipped_reasons['no_title_desc']}, "
          f"title_numeric={skipped_reasons['title_numeric']}, "
          f"not_english={skipped_reasons['not_english']}, "
          f"skip_jobs={skipped_reasons['skip_jobs']}")

    if len(clean_entries) < target_new:
        print(f"Warning: Only {len(clean_entries)} new entries (requested {target_new})")

    # Load existing train, val, test
    out_dir = Path("train_data")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.parquet"
    val_path = out_dir / "val.parquet"
    test_path = out_dir / "test.parquet"

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    # Split new entries: first NEW_TRAIN_SIZE for train, next NEW_VAL_SIZE for val
    new_train_entries = clean_entries[:NEW_TRAIN_SIZE]
    new_val_entries = clean_entries[NEW_TRAIN_SIZE : NEW_TRAIN_SIZE + NEW_VAL_SIZE]

    # Append new rows to existing
    new_train_df = pd.DataFrame(new_train_entries)
    new_val_df = pd.DataFrame(new_val_entries)

    train_df = pd.concat([train_df, new_train_df], ignore_index=True)
    val_df = pd.concat([val_df, new_val_df], ignore_index=True)

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    # test.parquet unchanged - not overwritten

    print(f"\nSaved train: {len(train_df)} total ({len(new_train_entries)} added) -> {train_path}")
    print(f"Saved val: {len(val_df)} total ({len(new_val_entries)} added) -> {val_path}")

if __name__ == "__main__":
    main()
