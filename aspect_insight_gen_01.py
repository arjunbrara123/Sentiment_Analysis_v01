"""
================================================================================
Aspect‑Level LLM Insights Generator – Dashboard‑Ready Version
================================================================================
PURPOSE
Automate deep‑dive, **dashboard‑ready** insights for each customer‑experience aspect
(appointment scheduling, billing, etc.) by combining existing sentiment data with
LLM reasoning.  The script outputs one **CSV row per British Gas pain‑point** with
clean columns that flow straight into Streamlit, Power BI, or Tableau.

VALUE
• Turns unstructured reviews into actionable, competitor‑benchmarked actions
• Outputs tidy data schema (no free‑text blobs) – ready for visual cards, tables,
  and commentary panes
• Adds sentiment‑gap metrics so the dashboard can rank opportunities by impact

KEY STEPS
1. **Identify top competitor** for each aspect (highest sentiment ≥ 100 mentions).
2. **Select reviews** (most relevant) for British Gas & competitor.
3. **Ask GPT‑4o** for JSON‑formatted issues (summary, quotes, explanation, action).
4. **Parse JSON** → explode into rows & add sentiment‑gap columns.
5. **Save CSV** in `output/aspect_insight_llm_output.csv`.

REQUIREMENTS
- Python 3.9+
- `pip install openai pandas tqdm python‑dotenv`
- Previous script outputs in `output/` folder:
  • `reviews_audit_sentiment_calcs.csv` (granular)
  • `LLM SA Monthly Data.csv` (aggregated)
- OpenAI API key in environment (`OPENAI_API_KEY=`…)
================================================================================
"""

# === SECTION 1 · USER INPUTS ===
import os, json, time
from dotenv import load_dotenv

# How many high‑relevance reviews to feed the model per company & aspect
TOP_N_REVIEWS: int = 100   # try 20–100 depending on cost ↔ detail trade‑off

# Max issues the LLM should return per aspect
MAX_ISSUES: int = 2

# File paths
INPUT_FILE  = "output/reviews_audit_sentiment_calcs.csv"
AGG_FILE    = "output/LLM SA Monthly Data.csv"
OUTPUT_FILE = "output/aspect_insight_llm_output.csv"

# OpenAI settings
OPENAI_MODEL   = "gpt-4.1" #'o3-2025-04-16', 'gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-2025-04-14'
TEMPERATURE    = 0.1
MAX_TOKENS     = 1600                  # generous to fit JSON & reasoning
RATE_LIMIT_SEC = 1                     # sleep between calls

# Load env vars
load_dotenv()

# === SECTION 2 · IMPORTS & SDK SET‑UP ===
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

ASPECTS = [
    "Appointment Scheduling",
    "Customer Service",
    "Energy Readings",
    "Meter Installations",
    "Value For Money",
    "Accounts and Billing",
]

# === SECTION 3 · DATA LOADING & VALIDATION ===

def load_data():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Missing {INPUT_FILE}. Run the sentiment script first.")
    if not os.path.exists(AGG_FILE):
        raise FileNotFoundError(f"Missing {AGG_FILE}. Run the sentiment script first.")
    return pd.read_csv(INPUT_FILE), pd.read_csv(AGG_FILE)

# === SECTION 4 · COMPETITOR SELECTION ===

def top_competitor_per_aspect(agg: pd.DataFrame) -> dict:
    """Return {aspect: competitor} with highest sentiment (≥100 mentions)."""
    mapping = {}
    for aspect in ASPECTS:
        score_col = f"{aspect}_sentiment_score"
        vol_col   = f"{aspect}_Volume"
        subset = agg[(agg[vol_col] >= 100) & agg[score_col].notna()]
        if subset.empty:
            continue
        best_row = subset.loc[subset[score_col].idxmax()]
        mapping[aspect] = best_row["Company"]
    return mapping

def aspect_gap(agg: pd.DataFrame, aspect: str, competitor: str):
    """Return BG sentiment, competitor sentiment, and gap."""
    col = f"{aspect}_sentiment_score"
    bg = agg[agg["Company"] == "British Gas"][col].mean()
    comp = agg[agg["Company"] == competitor][col].mean()
    return round(bg, 2), round(comp, 2), round((comp - bg), 2)

# === SECTION 5 · REVIEW SELECTION ===

def get_reviews(df: pd.DataFrame, aspect: str, company: str, n: int):
    rel_col = f"Relevance_{aspect}"
    mask = (df["Company"] == company) & (df[rel_col] > 0)
    return df.loc[mask].sort_values(rel_col, ascending=False).head(n)["Review"].tolist()

# === SECTION 6 · LLM CALL ===

def craft_prompt(aspect: str, comp: str, bg_reviews: list[str], comp_reviews: list[str]) -> str:
    return f"""
You are a senior energy‑sector CX analyst.

Compare **British Gas** and **{comp}** on the aspect **{aspect}**.

Use the quotes below (bullet‑pointed). Identify up to {MAX_ISSUES} clear British Gas pain points and how {comp} outperforms them.

---  
British Gas Reviews:\n""" + "\n".join(f"- {r}" for r in bg_reviews) + "\n" + f"""
---  
{comp} Reviews:\n""" + "\n".join(f"- {r}" for r in comp_reviews) + "\n" + f"""
---

Return **valid JSON** exactly in this schema:
{{
  "issues": [
    {{
      "summary": "<short summary>",
      "bg_quote": "<exact BG quote>",
      "competitor_quote": "<exact competitor quote>",
      "explanation": "<why competitor wins>",
      "action": "<concise pilot action>"
    }}
  ]
}}
Do not wrap the JSON in markdown. No additional text.
"""

def query_llm(prompt: str) -> dict | None:
    try:
        resp = openai_client.chat.completions.create(
            model       = OPENAI_MODEL,
            messages    = [
                {"role": "system", "content": "You are a commercially‑savvy CX strategist."},
                {"role": "user",   "content": prompt},
            ],
            temperature = TEMPERATURE,
            max_tokens  = MAX_TOKENS,
        )
        raw = resp.choices[0].message.content.strip()
        return json.loads(raw)
    except Exception as e:
        print(f"⚠️  LLM error: {e}")
        return None

# === SECTION 7 · PIPELINE ===

def run_pipeline():
    detailed, agg = load_data()
    top_comp_map = top_competitor_per_aspect(agg)
    rows = []

    for aspect in tqdm(ASPECTS, desc="Aspects"):
        competitor = top_comp_map.get(aspect)
        if not competitor:
            continue

        bg_reviews   = get_reviews(detailed, aspect, "British Gas", TOP_N_REVIEWS)
        comp_reviews = get_reviews(detailed, aspect, competitor,   TOP_N_REVIEWS)

        prompt = craft_prompt(aspect, competitor, bg_reviews, comp_reviews)
        data   = query_llm(prompt)
        time.sleep(RATE_LIMIT_SEC)
        if not data or "issues" not in data:
            continue

        bg_sent, comp_sent, gap = aspect_gap(agg, aspect, competitor)

        for idx, issue in enumerate(data["issues"], start=1):
            rows.append({
                "Aspect": aspect,
                "Issue #": idx,
                "Summary": issue.get("summary", ""),
                "BG Quote": issue.get("bg_quote", ""),
                "Competitor Quote": issue.get("competitor_quote", ""),
                "Explanation": issue.get("explanation", ""),
                "Action": issue.get("action", ""),
                "Top Competitor": competitor,
                "BG Sentiment": bg_sent,
                "Competitor Sentiment": comp_sent,
                "Sentiment Gap": gap,
            })

    pd.DataFrame(rows).to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Dashboard‑ready insights saved to →  {OUTPUT_FILE}")

# === SECTION 8 · MAIN ===
if __name__ == "__main__":
    run_pipeline()
