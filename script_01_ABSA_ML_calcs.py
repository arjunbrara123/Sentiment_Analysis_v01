"""
================================================================================
Sentiment & Relevance Analysis for Customer Reviews
================================================================================
PURPOSE:
This script analyzes customer reviews to calculate one overall sentiment score
and a "relevance score" for several key business aspects. It is designed to be
fast, accurate, and easily understood by non-technical leaders.

OUTPUTS:
1. A detailed CSV file ('review_level_data.csv') with a sentiment and
   relevance score for every single review, providing a full audit trail.
2. A summary CSV file ('monthly_aggregated_data.csv') that aggregates
   these scores by month and company for high-level business intelligence.

--------------------------------------------------------------------------------
SCRIPT STRUCTURE:
This file is organized into the following sections for clarity:
1.  IMPORTS: Loads necessary software libraries.
2.  CONFIGURATION: **The only section a user needs to edit.**
3.  KEYWORD DEFINITIONS: The business aspects and their related keywords.
4.  CORE ANALYTICAL ENGINE: The functions that perform the analysis, broken down into:
    4.1 - Data Ingestion & Validation
    4.2 - Natural Language Pre-Processing
    4.3 - Sentiment & Relevance Calculation
    4.4 - Report Generation
5.  MAIN WORKFLOW: Executes the entire process in a clear, step-by-step manner.
--------------------------------------------------------------------------------
"""

# === SECTION 1: IMPORTS ===
# Loads the software libraries required for the script to run.
#
import pandas as pd
from tqdm.auto import tqdm
import os
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


# === SECTION 2: CONFIGURATION ===
# This is the only section where a user should change settings.
#
# 1. FILE NAMES
INPUT_FILENAME = 'CIT Energy 2024 TrustPilot v2.csv' # Input file containing the customer reviews
OUTPUT_FILENAME_DETAILED = 'output/review_level_data.csv' # Original input with appended calculation columns for auditability
OUTPUT_FILENAME_AGGREGATED = 'output/monthly_aggregated_data.csv' # Calculation output aggregated by month

# 2. COLUMN NAMES
REVIEW_COLUMN_NAME = 'Review'
DATE_COLUMN_NAME = "Date"
COMPANY_COLUMN_NAME = "Company"
# The script will look for this column. If not found, it will create it
# and fill it with the default value specified below.
PRODUCT_COLUMN_NAME = "Product"
DEFAULT_PRODUCT_NAME = "Energy"


# === SECTION 3: KEYWORD DEFINITIONS ===
# This dictionary defines the keywords used to identify different aspects of a
# review. The script will calculate a 'Relevance' score for each of these aspects.
#
ASPECTS_KEYWORDS = {
    'Appointment Scheduling': ['appointment', 'booking', 'schedule', 'engineer', 'visit', 'reschedule', 'cancel', 'time', 'slot', 'booked'],
    'Customer Service': ['service', 'support', 'helpline', 'call', 'centre', 'agent', 'representative', 'chat', 'email', 'complaint', 'resolution'],
    'Energy Readings': ['reading', 'meter', 'submit', 'smart', 'estimate', 'usage'],
    'Meter Installations': ['meter', 'installation', 'smart', 'engineer', 'install', 'new', 'exchange'],
    'Value For Money': ['price', 'cost', 'tariff', 'bill', 'expensive', 'cheap', 'value', 'direct', 'debit', 'payment', 'refund'],
    'Accounts and Billing': ['account', 'bill', 'billing', 'payment', 'direct', 'debit', 'statement', 'refund', 'credit', 'charge', 'tariff']
}


# === SECTION 4: CORE ANALYTICAL ENGINE ===
# This section contains the functions that perform the analysis. It is the
# "engine" of the script. It has been broken down into logical stages.
# There is no need to edit anything here.
#

# --- 4.1: Data Ingestion & Validation ---
# PURPOSE:  To load the raw data from the source file and ensure it's in the
#           correct format before any analysis begins.
# PROCESS:  The script reads the specified CSV file and confirms that the essential
#           columns exist. If a 'Product' column isn't found, it is intelligently
#           created using the default value from the configuration.
# VALUE:    This is a critical quality control step that ensures the integrity
#           of our input data, leading to more reliable and robust results.
#
def download_nltk_resources():
    """Checks for necessary NLTK packages and downloads them if missing."""
    resources = [('tokenizers/punkt', 'punkt'), ('sentiment/vader_lexicon', 'vader_lexicon')]
    print("Setting up NLTK models...")
    for path, name in resources:
        try:
            nltk.data.find(path)
            print(f"- '{name}' is already downloaded.")
        except LookupError:
            print(f"-> Downloading missing resource: '{name}'...")
            nltk.download(name)
    print("Setup complete.")

def load_and_validate_data(filename, review_col, date_col, company_col, product_col, default_product):
    """Loads the input CSV, validates columns, and handles the product column dynamically."""
    print(f"\nLoading data from {filename}...")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"ERROR: Input file not found at '{filename}'. Please check the path.")
    
    df = pd.read_csv(filename)
    df.dropna(subset=[review_col], inplace=True)

    for col in [review_col, date_col, company_col]:
        if col not in df.columns:
            raise ValueError(f"ERROR: The required column '{col}' was not found in the CSV. Available columns are: {list(df.columns)}")
    
    if product_col not in df.columns:
        print(f"- Product column '{product_col}' not found. Creating it with default value: '{default_product}'.")
        df[product_col] = default_product
    else:
        print(f"- Using existing product column: '{product_col}'.")

    df['Month'] = pd.to_datetime(df[date_col]).dt.to_period('M').dt.to_timestamp()
    return df

# --- 4.2: Natural Language Pre-Processing ---
# PURPOSE:  To transform raw, unstructured review text into a structured format
#           that our algorithms can analyze accurately and efficiently.
# PROCESS:  This is a one-time "heavy lifting" step. For each review, the script
#           1. Isolates every individual word (a process called 'Tokenization').
#           2. Algorithmically reduces each word to its core concept or 'stem'
#              (e.g., 'billing' and 'billed' both become 'bill').
#           This is done once per review to maximize speed.
# VALUE:    This sophisticated step ensures that our analysis captures all relevant
#           feedback, regardless of the specific tense or phrasing the customer uses.
#
def preprocess_text_data(df, review_col):
    """Performs tokenization and stemming on each review ONCE to speed up the script."""
    print("\nPreprocessing text data for advanced analysis...")
    stemmer = PorterStemmer()
    tqdm.pandas(desc="Analyzing word structure")
    
    def stem_text(text):
        if not isinstance(text, str): return []
        try:
            words = word_tokenize(text.lower())
            return [stemmer.stem(word) for word in words]
        except Exception:
            return []
            
    df['_stemmed_tokens'] = df[review_col].progress_apply(stem_text)
    return df

# --- 4.3: Sentiment & Relevance Calculation ---
# PURPOSE:  To score every review for its emotional tone and its relevance to
#           key business drivers.
# PROCESS:  First, a lexicon-based sentiment engine (VADER) reads each review
#           to assign an 'Overall_Sentiment' score. This score is then scaled
#           to the business-standard range of -100 to +100. Then, our proprietary
#           relevance algorithm counts keywords to score each aspect.
# VALUE:    This gives us two critical dimensions for each review: how the customer
#           felt (on a simple 100-point scale), and what specific part of the
#           business they were talking about.
#
def calculate_scores(df, review_col):
    """Calculates sentiment and relevance, then scales and rounds sentiment."""
    print("\nCalculating Overall Sentiment for each review...")
    sid = SentimentIntensityAnalyzer()
    tqdm.pandas(desc="Scoring sentiment")
    
    df['Overall_Sentiment'] = df[review_col].progress_apply(
        lambda text: sid.polarity_scores(text)['compound'] if isinstance(text, str) else None
    )
    df['Overall_Sentiment'] = (df['Overall_Sentiment'] * 100).round(1)

    print("\nCalculating Relevance Scores for each business aspect...")
    stemmed_keywords = {aspect: [PorterStemmer().stem(kw) for kw in kws] for aspect, kws in ASPECTS_KEYWORDS.items()}
    for aspect, keywords in stemmed_keywords.items():
        tqdm.pandas(desc=f"Scoring Relevance: {aspect}")
        df[f'Relevance_{aspect}'] = df['_stemmed_tokens'].progress_apply(
            lambda tokens: sum(1 for token in tokens if token in keywords)
        )
    return df

# --- 4.4: Report Generation ---
# PURPOSE:  To synthesize the processed data into the two final business-ready
#           CSV reports: one for detailed auditing and one for high-level trends.
# PROCESS:  The script constructs and saves the two DataFrames as CSV files with
#           the appropriate formatting, naming, and column order.
# VALUE:    This provides the final, tangible outputs needed for strategic
#           analysis, reporting, and decision-making.
#
def generate_detailed_output(df, filename):
    """Creates the detailed, review-level CSV file for auditing."""
    print(f"\nGenerating detailed audit file: '{filename}'...")
    output_cols = [REVIEW_COLUMN_NAME, DATE_COLUMN_NAME, COMPANY_COLUMN_NAME, PRODUCT_COLUMN_NAME, 'Overall_Sentiment'] + [f'Relevance_{aspect}' for aspect in ASPECTS_KEYWORDS.keys()]
    df[output_cols].to_csv(filename, index=False)
    print("...Done.")

def generate_aggregated_output(df, filename):
    """Creates the high-level summary CSV, aggregated by month and company."""
    print(f"\nGenerating aggregated monthly report: '{filename}'...")
    
    group_by_cols = ['Month', COMPANY_COLUMN_NAME, PRODUCT_COLUMN_NAME]
    
    df_agg_base = df.groupby(group_by_cols).size().reset_index(name='Reviews Count')
    df_overall_sent = df.groupby(group_by_cols)['Overall_Sentiment'].mean().reset_index().rename(columns={'Overall_Sentiment': 'Overall_SA'})
    df_agg = pd.merge(df_agg_base, df_overall_sent, on=group_by_cols)

    for aspect in ASPECTS_KEYWORDS.keys():
        relevance_col = f'Relevance_{aspect}'
        df_aspect = df[df[relevance_col] > 0].copy()
        
        aspect_volume = df_aspect.groupby(group_by_cols).size().reset_index(name=f'{aspect}_Volume')
        aspect_sent = df_aspect.groupby(group_by_cols)['Overall_Sentiment'].mean().reset_index().rename(columns={'Overall_Sentiment': f'{aspect}_SA'})
        
        df_agg = pd.merge(df_agg, aspect_volume, on=group_by_cols, how='left')
        df_agg = pd.merge(df_agg, aspect_sent, on=group_by_cols, how='left')

    df_agg.fillna(0, inplace=True)
    
    sa_cols = [col for col in df_agg.columns if col.endswith('_SA')]
    for col in sa_cols:
        df_agg[col] = df_agg[col].round(1)

    df_agg.rename(columns={PRODUCT_COLUMN_NAME: 'Final Product Category'}, inplace=True)
    
    # Define final column order
    final_cols_order = ['Month', 'Company', 'Final Product Category', 'Reviews Count', 'Overall_SA'] + sorted([col for col in df_agg.columns if '_SA' in col and 'Overall' not in col or '_Volume' in col])
    df_agg = df_agg[final_cols_order]

    df_agg.to_csv(filename, index=False, date_format='%d/%m/%Y')
    print("...Done.")


# === SECTION 5: MAIN WORKFLOW ===
# This section executes the entire process using the functions defined above.
# It provides a high-level summary of the script's operational flow.
#

"""Main workflow to run the entire data processing pipeline."""
# Ensure all required linguistic models are downloaded before starting.
download_nltk_resources()

# STEP 1: Ingest and Validate Data (as defined in Section 4.1)
data_df = load_and_validate_data(INPUT_FILENAME, REVIEW_COLUMN_NAME, DATE_COLUMN_NAME, COMPANY_COLUMN_NAME, PRODUCT_COLUMN_NAME, DEFAULT_PRODUCT_NAME)

# STEP 2: Perform one-time text pre-processing for speed (Section 4.2)
data_df = preprocess_text_data(data_df, REVIEW_COLUMN_NAME)

# STEP 3: Calculate all sentiment and relevance scores (Section 4.3)
data_df = calculate_scores(data_df, REVIEW_COLUMN_NAME)

# STEP 4: Generate the final business-ready reports (Section 4.4)
generate_detailed_output(data_df, OUTPUT_FILENAME_DETAILED)
generate_aggregated_output(data_df, OUTPUT_FILENAME_AGGREGATED)

print("\nProcessing complete.")
