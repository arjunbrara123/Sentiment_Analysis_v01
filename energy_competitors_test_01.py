"""
Script: annotate_reviews.py
Purpose:
  • Load a CSV of customer reviews (“Review” column)
  • Compute sentiment (TextBlob & VADER), length, payment type,
    justification, switching intent/status, topic relevance (keywords & TF–IDF),
    promotional flags, contact channels, regulatory references,
    and an overall quality score.
  • Measure and report time taken for each processing section.

Inputs:
  • CSV file with at least a “Review” column (e.g. energy_reviews.csv)

Outputs:
  • annotated_reviews.csv containing original data plus new columns:
    sentiment scores, normalized sentiment, word_count, length_cat,
    payment_method, justification, switching, kw_<topic>, tfidf_<topic>,
    promo_flag, channel_<x>, reg_ref_flag, topic_cov_n, length_score,
    quality_score.

Dependencies:
  • Python 3.x
  • pandas
  • nltk (vader_lexicon)
  • textblob
  • scikit-learn
"""

import time
import nltk
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from joblib import Parallel, delayed

# -------------------------------------------------------------------
# Timing start
# -------------------------------------------------------------------
start_time = time.time()
prev_time = start_time

# -------------------------------------------------------------------
# Section 1: Download & Initialization
# Purpose: set up NLP resources and sentiment analyzers
# -------------------------------------------------------------------
nltk.download('vader_lexicon', quiet=True)
sid = SentimentIntensityAnalyzer()
section_end = time.time()
print(f"Section 1 (Download & Initialization) took {section_end - prev_time:.2f} seconds")
prev_time = section_end

# -------------------------------------------------------------------
# Section 2: Data Loading
# Purpose: load review dataset
# -------------------------------------------------------------------
df = pd.read_csv("C:/Users/arjun/OneDrive/CIT data - energy reviews.csv")
section_end = time.time()
print(f"Section 2 (Data Loading) took {section_end - prev_time:.2f} seconds")
prev_time = section_end

# -------------------------------------------------------------------
# Section 3: Sentiment Scoring
# Purpose: quantify review tone using two methods for cross-validation.
#  - TextBlob polarity reflects overall positive(>0)/negative(<0) sentiment at sentence-level.
#  - VADER compound score is optimized for social/media text, weights intensifiers and negations.
#  Normalizing both to [0,1] allows direct comparison and aggregation.
#
# Columns:
# - sent_tb      : TextBlob polarity score in [-1,1]
# - sent_vader   : VADER compound score in [-1,1]
# - sent_tb_n    : normalized TextBlob in [0,1]
# - sent_vader_n : normalized VADER in [0,1]
# -------------------------------------------------------------------
def tb_sent(text):
    try:
        return TextBlob(text).sentiment.polarity
    except:
        return 0.0

def vd_sent(text):
    try:
        return sid.polarity_scores(text)['compound']
    except:
        return 0.0

try:
    # use all available cores
    n_jobs = -1

    reviews = df['Review'].fillna('').tolist()

    tb_scores = Parallel(n_jobs=n_jobs)(
        delayed(tb_sent)(r) for r in reviews
    )
    vader_scores = Parallel(n_jobs=n_jobs)(
        delayed(vd_sent)(r) for r in reviews
    )

    # assign, round, normalize
    df['sent_tb'] = pd.Series(tb_scores).round(3)
    df['sent_vader'] = pd.Series(vader_scores).round(3)
    print("Parallelisation successfully used")
except:
    df['sent_tb']    = df['Review'].apply(tb_sent).round(3)
    df['sent_vader'] = df['Review'].apply(vd_sent).round(3)
    print("Parallelism failed. Went old school here")

# normalize to [0,1]
df['sent_tb_n']    = ((df['sent_tb'] + 1) / 2).round(3)
df['sent_vader_n'] = ((df['sent_vader'] + 1) / 2).round(3)
section_end = time.time()
print(f"Section 3 (Sentiment Scoring) took {section_end - prev_time:.2f} seconds")
prev_time = section_end

# -------------------------------------------------------------------
# Section 4: Review Length
# Purpose: bucket reviews by verbosity—longer often = more detail.
#
# Columns:
# - word_count : total number of words (higher implies more elaboration).
# - length_cat : 'Short'<50 words, 'Medium'50–150 (moderate detail), 'Long'>150 (in-depth).
# -------------------------------------------------------------------
df['word_count'] = df['Review'].str.split().str.len().fillna(0).astype(int)
df['length_cat'] = pd.cut(
    df['word_count'],
    bins=[0, 50, 150, df['word_count'].max() + 1],
    labels=['Short', 'Medium', 'Long'],
    right=False
)
section_end = time.time()
print(f"Section 4 (Review Length) took {section_end - prev_time:.2f} seconds")
prev_time = section_end

# -------------------------------------------------------------------
# Section 5: Payment Method
# Purpose: identify customer payment channel, which often correlates with complaint types.
#
# Column:
# - payment_method : 'Direct Debit' (auto-pay, usually stable billing),
#                    'Prepayment Meter' (pay-as-you-go, often higher cost),
#                    'Standard Credit' (manual payment, variable).
# -------------------------------------------------------------------
def payment_method(text):
    t = text.lower()
    if 'direct debit' in t:
        return 'Direct Debit'
    if 'prepayment' in t or 'pre-paid' in t:
        return 'Prepayment Meter'
    return 'Standard Credit'

df['payment_method'] = df['Review'].apply(payment_method)
section_end = time.time()
print(f"Section 5 (Payment Method) took {section_end - prev_time:.2f} seconds")
prev_time = section_end

# -------------------------------------------------------------------
# Section 6: Fuel Type Classification
# Purpose: tag each review by the energy product(s) mentioned, so you can segment sentiment and issues by supply type.
#
# Columns:
# - fuel_type : one of:
#     • 'Gas'        — mentions gas only (e.g. “my gas meter”).
#     • 'Electricity' — mentions electric(electricity) only (e.g. “electric bill”).
#     • 'Dual Fuel'  — explicitly mentions both gas & electricity (e.g. “dual fuel tariff”).
#     • 'Unknown'    — no clear reference to fuel type.
#
# Interpretation: allows comparing, for instance, gas-specific complaints (e.g. heating outages) vs electricity issues (e.g. meter errors), or evaluating overall feedback across dual-fuel users.
# -------------------------------------------------------------------
def fuel_type_category(text):
    t = text.lower()
    if 'dual fuel' in t:
        return 'Dual Fuel'
    elif 'electric' in t:
        return 'Electricity'
    elif 'gas' in t:
        return 'Gas'
    return 'Unknown'

df['fuel_type'] = df['Review'].apply(fuel_type_category)
section_end = time.time()
print(f"Section 6 (Fuel Type Classification) took {section_end - prev_time:.2f} seconds")
prev_time = section_end

# -------------------------------------------------------------------
# Section 7: Tariff Type Classification
# Purpose: identify which tariff structure the review references, to analyze satisfaction by plan type.
#
# Columns:
# - tariff_type : one of:
#     • 'Fixed'    — mentions 'fixed' indicating a locked-in rate (e.g. “fixed for 12 months”).
#     • 'Variable' — mentions 'variable' or synonyms (e.g. “standard variable tariff”).
#     • 'Unknown'  — no tariff-type keyword found.
#
# Interpretation: lets you compare feedback and sentiment for rate-guaranteed customers vs those on variable plans, highlighting differences in billing concerns.
# -------------------------------------------------------------------
def tariff_type_category(text):
    t = text.lower()
    if 'fixed' in t:
        return 'Fixed'
    if 'variable' in t:
        return 'Variable'
    return 'Unknown'

df['tariff_type'] = df['Review'].apply(tariff_type_category)
section_end = time.time()
print(f"Section 7 (Tariff Type Classification) took {section_end - prev_time:.2f} seconds")
prev_time = section_end

# -------------------------------------------------------------------
# Section 8: Justification Flag
# Purpose: flag reviews where customer explains 'why' (richer feedback).
#
# Column:
# - justification : 1 if contains connectives like because, as, since, so that;
#                   these reviews often provide actionable detail.
# -------------------------------------------------------------------
justifiers    = ['because', 'as', 'since', 'so that']
pattern_just  = r'\b(?:' + '|'.join(justifiers) + r')\b'
df['justification'] = df['Review']\
    .str.contains(pattern_just, flags=re.IGNORECASE, na=False)\
    .astype(int)
section_end = time.time()
print(f"Section 8 (Justification Flag) took {section_end - prev_time:.2f} seconds")
prev_time = section_end

# -------------------------------------------------------------------
# Section 9: Switching Intent/Outcome
# Purpose: distinguish between customers planning to switch vs those who have switched.
#
# Column:
# - switching : 'Intent' if future switching keywords (e.g. will switch),
#               'Outcome' if past switch verbs (e.g. switched to),
#               'None' otherwise.
# -------------------------------------------------------------------
def switching_category(text):
    t = text.lower()
    if any(k in t for k in ['will switch', 'looking to switch', 'consider switching']):
        return 'Intent'
    if any(k in t for k in ['switched to', 'moved to', 'changed to']):
        return 'Outcome'
    return 'None'

df['switching'] = df['Review'].apply(switching_category)
section_end = time.time()
print(f"Section 9 (Switching Category) took {section_end - prev_time:.2f} seconds")
prev_time = section_end

# -------------------------------------------------------------------
# Section 10: Topic Relevance Scores
# Purpose: measure strength of five key themes via keyword frequency
#          and TF–IDF (to downweight very common words).
#
# Columns for each topic (e.g. pricing):
# - kw_pricing    : raw count of theme keywords (higher = more mentions)
# - tfidf_pricing : sum of TF–IDF weights (higher = more distinctive relevance)
# -------------------------------------------------------------------
topics = {
    'pricing':   ['bill', 'overcharged', 'tariff', 'price', 'cost'],
    'service':   ['call centre', 'support', 'agent', 'rude', 'helpful'],
    'churn':     ['switch', 'exit fee', 'moving home', 'competitive quote'],
    'technical': ['outage', 'meter', 'smart meter', 'error'],
    'green':     ['green', 'renewable', 'eco', 'carbon offset']
}
for name, kws in topics.items():
    pat = '|'.join(map(re.escape, kws))
    df[f'kw_{name}'] = df['Review'].str.lower().str.count(pat)
    vect = TfidfVectorizer(vocabulary=kws)
    tfidf = vect.fit_transform(df['Review'].fillna('')).sum(axis=1)
    df[f'tfidf_{name}'] = pd.Series(tfidf.A1).round(3)
section_end = time.time()
print(f"Section 10 (Topic Scores) took {section_end - prev_time:.2f} seconds")
prev_time = section_end

# -------------------------------------------------------------------
# Section 11: Promotional Mentions
# Purpose: flag reviews referencing short-term deals or exit fees.
#
# Column:
# - promo_flag : 1 if mentions words like introductory, exit fee, cheapest deal;
#                highlights sensitivity to pricing traps.
# -------------------------------------------------------------------
promo_kw = ['introductory', 'exit fee', 'cheapest deal', 'fixed for']
df['promo_flag'] = df['Review']\
    .str.contains('|'.join(map(re.escape, promo_kw)), flags=re.IGNORECASE, na=False)\
    .astype(int)
section_end = time.time()
print(f"Section 11 (Promotional Mentions) took {section_end - prev_time:.2f} seconds")
prev_time = section_end

# -------------------------------------------------------------------
# Section 12: Contact Channel Flags
# Purpose: identify which support channel is discussed (app/website/email/phone/chat).
#
# Columns:
# - channel_<x> : 1 if review mentions the channel; helps trace digital vs offline issues.
# - channel_app, channel_website, channel_email, channel_phone, channel_chat : 1 or 0
# -------------------------------------------------------------------
channels = ['app', 'website', 'email', 'phone', 'chat']
for ch in channels:
    df[f'channel_{ch}'] = df['Review']\
        .str.contains(rf'\b{re.escape(ch)}\b', flags=re.IGNORECASE, na=False)\
        .astype(int)
section_end = time.time()
print(f"Section 12 (Contact Channels) took {section_end - prev_time:.2f} seconds")
prev_time = section_end

# -------------------------------------------------------------------
# Section 13: Regulatory References
# Purpose: flag formal complaints or contract mentions.
#
# Column:
# - reg_ref_flag : 1 if mentions 'Ofgem','T&C','contract','terms','conditions'
# -------------------------------------------------------------------
reg_kw = ['ofgem', 't&c', 'contract', 'terms', 'conditions']
df['reg_ref_flag'] = df['Review']\
    .str.contains('|'.join(map(re.escape, reg_kw)), flags=re.IGNORECASE, na=False)\
    .astype(int)
section_end = time.time()
print(f"Section 13 (Regulatory References) took {section_end - prev_time:.2f} seconds")
prev_time = section_end

# -------------------------------------------------------------------
# Section 14: Quality Score
# Purpose: aggregate key normalized metrics into a single “quality” indicator.
#
# Columns:
# - topic_cov_n  : normalized sum of all kw_* counts in [0,1]
# - length_score : normalized word_count/150 in [0,1]
# - quality_score: mean of [sent_tb_n, sent_vader_n, topic_cov_n,
#                          length_score, justification, promo_flag, reg_ref_flag]
# -------------------------------------------------------------------
kw_cols = [f'kw_{n}' for n in topics]
max_kw  = df[kw_cols].sum(axis=1).max() or 1
df['topic_cov_n']  = (df[kw_cols].sum(axis=1) / max_kw).round(3)
df['length_score'] = (df['word_count'] / 150).clip(0, 1).round(3)

factors = [
    'sent_tb_n', 'sent_vader_n', 'topic_cov_n',
    'length_score', 'justification', 'promo_flag', 'reg_ref_flag'
]
df['quality_score'] = df[factors].mean(axis=1).round(3)
section_end = time.time()
print(f"Section 14 (Quality Score) took {section_end - prev_time:.2f} seconds")
prev_time = section_end

# -------------------------------------------------------------------
# Section 15: Summary Statistics
# Purpose: provide quick counts and averages to interpret results
# -------------------------------------------------------------------
print("\n=== Summary Statistics ===")
total = len(df)

# Sentiment breakdown (counts and % of total)
pos_tb = (df['sent_tb'] > 0).sum()
neut_tb = (df['sent_tb'] == 0).sum()
neg_tb = (df['sent_tb'] < 0).sum()
pos_tb_pct = round(pos_tb/total*100)
neut_tb_pct = round(neut_tb/total*100)
neg_tb_pct = round(neg_tb/total*100)
print(f"TextBlob: {pos_tb} positive ({pos_tb_pct}%), {neut_tb} neutral ({neut_tb_pct}%), {neg_tb} negative ({neg_tb_pct}%)")

pos_v = (df['sent_vader'] > 0).sum()
neut_v = (df['sent_vader'] == 0).sum()
neg_v = (df['sent_vader'] < 0).sum()
pos_v_pct = round(pos_v/total*100)
neut_v_pct = round(neut_v/total*100)
neg_v_pct = round(neg_v/total*100)
print(f"VADER:    {pos_v} positive ({pos_v_pct}%), {neut_v} neutral ({neut_v_pct}%), {neg_v} negative ({neg_v_pct}%)")

# Review length categories
print("Review lengths:")
for cat, cnt in df['length_cat'].value_counts().items():
    pct = round(cnt/total*100)
    print(f"  {cat}: {cnt} reviews ({pct}%)")

# Payment methods
print("Payment methods:")
for pm, cnt in df['payment_method'].value_counts().items():
    pct = round(cnt/total*100)
    print(f"  {pm}: {cnt} reviews ({pct}%)")

# Fuel types
print("\nFuel types:")
for ft, cnt in df['fuel_type'].value_counts().items():
    pct = round(cnt/total*100)
    print(f"  {ft}: {cnt} reviews ({pct}%)")

# Tariff types
print("\nTariff types:")
for tt, cnt in df['tariff_type'].value_counts().items():
    pct = round(cnt/total*100)
    print(f"  {tt}: {cnt} reviews ({pct}%)")

# Switching categories
print("Switching categories:")
for sw, cnt in df['switching'].value_counts().items():
    pct = round(cnt/total*100)
    print(f"  {sw}: {cnt} reviews ({pct}%)")

# Justification & promotions
just_cnt = df['justification'].sum()
promo_cnt = df['promo_flag'].sum()
just_pct = round(just_cnt/total*100)
promo_pct = round(promo_cnt/total*100)
print(f"Justifications provided: {just_cnt} reviews ({just_pct}%)")
print(f"Promotional mentions:   {promo_cnt} reviews ({promo_pct}%)")

# Topic averages (unchanged)
print("Topic averages:")
for name in topics:
    kw_avg = df[f'kw_{name}'].mean().round(2)
    tf_avg = df[f'tfidf_{name}'].mean().round(3)
    print(f"  {name.title()}: Avg keywords {kw_avg}, Avg TF-IDF {tf_avg}")

# Contact channels
print("Contact channel mentions:")
for ch in channels:
    cnt = df[f'channel_{ch}'].sum()
    pct = round(cnt/total*100)
    print(f"  {ch}: {cnt} mentions ({pct}%)")

# Regulatory references
reg_cnt = df['reg_ref_flag'].sum()
reg_pct = round(reg_cnt/total*100)
print(f"Regulatory references: {reg_cnt} reviews ({reg_pct}%)")

# Quality score distribution (unchanged)
print("Quality score summary:")
print(df['quality_score'].describe().round(3))

# -------------------------------------------------------------------
# Total run time
# -------------------------------------------------------------------
total_time = time.time() - start_time
print(f"Total run time: {total_time:.2f} seconds")

# Save annotated DataFrame
df.to_csv('annotated_reviews.csv', index=False)
