import pandas as pd
from transformers import pipeline
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from datetime import datetime
import re

print(f"Starting script at {datetime.now()}")

# 1) Load the data
df = pd.read_csv("BG_Trustpilot_AI_Sentiment_Predictive_Analysis_Clean.csv")  # Must have a 'review_text' column

# ----------------------------------------------------------------------------
# STEP A: Remove Exact Duplicates
# ----------------------------------------------------------------------------
df.drop_duplicates(subset='Review', inplace=True)
print(f"Duplicates Removed -  {datetime.now()}")

# ----------------------------------------------------------------------------
# STEP B: SPAM DETECTION
# ----------------------------------------------------------------------------
spam_classifier = pipeline(
    "text-classification",
    model="mrm8488/bert-tiny-finetuned-sms-spam-detection",
    truncation=True,      # Truncate inputs to max_length
    max_length=512,       # The standard max for many BERT models
    padding=True          # Optional: add padding to align batches
)

reviews = df["Review"].tolist()
batch_size = 256

spam_labels = []
spam_scores = []

for i in range(0, len(reviews), batch_size):
    batch = reviews[i: i + batch_size]
    results = spam_classifier(batch)
    for r in results:
        spam_labels.append(r["label"])  # "spam" or "ham"
        spam_scores.append(r["score"])  # Confidence

df["spam_label"] = spam_labels
df["spam_confidence"] = spam_scores

# Decide which spam you want to exclude or mark as definitely spam
SPAM_CONFIDENCE_THRESHOLD = 0.9
df["is_spam"] = df.apply(
    lambda row: True if (row["spam_label"] == "spam" and row["spam_confidence"] > SPAM_CONFIDENCE_THRESHOLD)
    else False,
    axis=1
)
print(f"Spam Classification Complete -  {datetime.now()}")


# ----------------------------------------------------------------------------
# STEP C: LANGUAGE DETECTION
# ----------------------------------------------------------------------------
def detect_language(text):
    try:
        lang = detect(text)
        return lang
    except LangDetectException:
        return "unknown"


df["detected_lang"] = df["Review"].apply(detect_language)

# If you only care about English, you can mark a review as English or not:
df["is_english"] = df["detected_lang"].apply(lambda x: True if x == "en" else False)
print(f"Languages Listed -  {datetime.now()}")

# ----------------------------------------------------------------------------
# STEP D: JUSTIFICATION CHECK (Refined Justification)
# ----------------------------------------------------------------------------
# A simple approach: detect cause-effect words/phrases like "because", "since", " as ", "due to", etc.
# This is still naive, but it's automated and very fast.

justification_keywords = ["because", "since", " as ", "due to", "reason", "why", "so that"]


justification_regex = re.compile(r"\b(because|since|as|due to|reason|why|so that)\b", re.IGNORECASE)

def has_refined_justification(text):
    return bool(justification_regex.search(str(text)))


df["has_justification"] = df["Review"].apply(has_refined_justification)
print(f"Justification Check Complete -  {datetime.now()}")


# ----------------------------------------------------------------------------
# STEP E: Create a Final 0–1 Weight
# ----------------------------------------------------------------------------
# We'll create a simple formula for the final weighting:
#
#   - If it's classified as spam => weight = 0
#   - If not spam => base_weight = 1
#   - If not English => multiply by 0.5 (for example)
#   - If has justification => multiply by 1, else 0.9 (some mild penalty if no justification)
#
# You can tweak these multipliers as you see fit.

def calculate_review_weight(row):
    # If spam, weight is 0
    if row["is_spam"]:
        return 0.0

    # Start with a base weight
    weight = 1.0

    # Language factor
    if not row["is_english"]:
        weight *= 0.5  # example: half-weight if not English

    # Justification factor
    if row["has_justification"]:
        weight *= 1.0  # no penalty
    else:
        weight *= 0.9  # slight penalty if no justification

    return weight


df["review_weight"] = df.apply(calculate_review_weight, axis=1)
print(f"Weightings Created -  {datetime.now()}")

# ----------------------------------------------------------------------------
# STEP F: (Optional) Filter out spam or keep it for separate analysis
# ----------------------------------------------------------------------------
output_columns = [
    "Date",
    "Year-Month",
    "Final Product Category",
    "Review",
    "spam_label",
    "spam_confidence",
    "detected_lang",
    "is_english",
    "has_justification",
    "review_weight"
]
# Filter columns
df_output = df[output_columns]
#df_clean = df_output[df_output["review_weight"] > 0]  # effectively removes spam

# Save or proceed with analysis
df_output.to_csv("BG_cleaned_reviews_data.csv", index=False)
print(f"Data cleaning and scoring complete! -  {datetime.now()}")
