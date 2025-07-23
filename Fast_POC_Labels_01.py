#!/usr/bin/env python3
"""Review Sentiment & Product Classification Script
=================================================

Changes in this revision
-----------------------
* **UK‑style dates** – date column now parsed with `dayfirst=True` so
  formats like `25/07/24` are accepted.
* **Integer sentiment everywhere** – both per‑review scores **and** the
  monthly `avg_sentiment` are rounded to the nearest whole number.

What it does
------------
1. Adds `sentiment_score` (‑100…+100, int) & `product_category` per row.
2. Shows progress with *tqdm*.
3. Writes **enriched_reviews_*.csv** & **monthly_rollup_*.csv** (Year‑Month × product × company, with integer average sentiment & review count).

────────────────────────────  QUICK START  ────────────────────────────
```bash
pip install pandas textblob tqdm
python -m textblob.download_corpora
python review_sentiment_classifier.py
```
────────────────────────────────────────────────────────────────────────
"""

# ----------------------------- USER CONFIG --------------------------- #
INPUT_CSV: str = "BG TrustPilot All 09-24 to 07-25.csv"  # Input data
REVIEW_COL: str = "Review"          # Column with the review text
DATE_COL: str = "Date"               # Column with the review date (UK format: dd/mm/yy)
COMPANY_COL: str = "Company"          # Column with the company name
OUTPUT_DIR: str = "Unknown"           # Folder for output files

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "Boiler Services": [
        "boiler", "heating", "heating", "radiator",
        "asv", "service", "heat", "engineer", "homecare"
    ],
    "Plumbing & Drains": [
        "plumbing", "plumber", "drain", "pipe", "leak", "burst",
        "blockage", "toilet", "sink", "sewer"
    ],
    "Home Electrical": [
        "electrical", "socket", "wiring", "fuse", "circuit", "lighting",
        "breaker", "power cut", "powercut", "rewire"
    ],
    "Appliance Cover": [
        "appliance", "washing machine", "dishwasher", "fridge", "freezer",
        "oven", "hob", "dryer", "tumble dryer", "microwave"
    ],
    "Energy": [
        "energy", "electricity", "electric", "supplier",
        "meter", "tariff", "bill", "price", "standing charge",
        "direct debit", "account"
    ],
}
DEFAULT_CATEGORY: str = "other"
# --------------------------------------------------------------------- #

# ------------------------------ IMPORTS ------------------------------ #
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import pandas as pd
from textblob import TextBlob

try:
    from tqdm import tqdm
except ImportError:  # graceful fallback if tqdm missing
    tqdm = None
# --------------------------------------------------------------------- #

# ---------------------------  FUNCTIONS ------------------------------ #

def scale_sentiment(polarity: float) -> int:
    """Convert TextBlob polarity (‑1..1) ➜ integer (‑100..100)."""
    return int(round(polarity * 100))


def get_sentiment(text: str) -> int:
    """Return scaled sentiment for *text* (0 when NaN)."""
    if pd.isna(text):
        return 0
    polarity = TextBlob(str(text)).sentiment.polarity
    return scale_sentiment(polarity)


def classify_review(text: str, cats: Dict[str, List[str]], default: str) -> str:
    """Return first category whose keyword appears in *text* (case‑insensitive)."""
    if pd.isna(text):
        return default
    text_lower = str(text).lower()
    for cat, kws in cats.items():
        for kw in kws:
            if kw in text_lower:
                return cat
    return default
# --------------------------------------------------------------------- #

# ------------------------------ MAIN --------------------------------- #

def main() -> None:
    # 1️⃣ Load data ---------------------------------------------------- #
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        raise SystemExit(f"❌ Input file not found: {INPUT_CSV}")

    # Sanity checks
    for col in (REVIEW_COL, DATE_COL, COMPANY_COL):
        if col not in df.columns:
            raise SystemExit(f"❌ Column '{col}' not found in {INPUT_CSV}")

    # 2️⃣ Parse UK‑formatted dates ----------------------------------- #
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], dayfirst=True, errors="coerce")
    missing_dates = df[DATE_COL].isna().sum()
    if missing_dates:
        print(f"⚠️  {missing_dates} rows have invalid dates → dropped from roll‑up.")

    # 3️⃣ Progress‑aware enrichment ---------------------------------- #
    if tqdm is not None:
        tqdm.pandas(desc="🔍 Scoring sentiment")
        df["sentiment_score"] = df[REVIEW_COL].progress_apply(get_sentiment)
        tqdm.pandas(desc="🔗 Classifying product")
        df["product_category"] = df[REVIEW_COL].progress_apply(
            classify_review, args=(CATEGORY_KEYWORDS, DEFAULT_CATEGORY)
        )
    else:
        print("ℹ️  tqdm not installed – running without progress bars.")
        df["sentiment_score"] = df[REVIEW_COL].apply(get_sentiment)
        df["product_category"] = df[REVIEW_COL].apply(
            classify_review, args=(CATEGORY_KEYWORDS, DEFAULT_CATEGORY)
        )

    # 4️⃣ Write enriched CSV ----------------------------------------- #
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    enriched_path = out_dir / f"Processed Reviews {timestamp}.csv"
    df.to_csv(enriched_path, index=False)
    print(f"✅ Processed file → {enriched_path}")

    # 5️⃣ Monthly roll‑up -------------------------------------------- #
    df_valid = df.dropna(subset=[DATE_COL]).copy()
    df_valid["Year-Month"] = (
        df_valid[DATE_COL].dt.to_period("M").dt.to_timestamp()
    )

    agg = (
        df_valid.groupby(["Year-Month", "product_category", COMPANY_COL])
        .agg(avg_sentiment=("sentiment_score", "mean"),
             reviews_count=("sentiment_score", "size"))
        .reset_index()
    )

    # Round average sentiment to nearest int
    agg["avg_sentiment"] = agg["avg_sentiment"].round().astype(int)
    agg = agg.rename(columns={'avg_sentiment': 'Sentiment Score'})

    rollup_path = out_dir / f"Monthly_SA_{timestamp}.csv"
    agg.to_csv(rollup_path, index=False)
    print(f"✅ Monthly roll‑up → {rollup_path}")

# --------------------------------------------------------------------- #

if __name__ == "__main__":
    main()
