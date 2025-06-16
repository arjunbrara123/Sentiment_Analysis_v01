# process_reviews_final.py

"""
================================================================================
Sentiment & Relevance Analysis for Customer Reviews
================================================================================
PURPOSE:
This script is an enterprise-grade analytical engine that transforms raw customer
feedback into specific, quantifiable aspect based sentiment scores. It uses Natural
Language Processing (NLP) to derive insights on customer sentiment and the specific
business aspects being discussed.

OUTPUTS:
1. DETAILED AUDIT TRAIL ('reviews_audit_sentiment_calcs.csv'): A granular,
   review-by-review breakdown of all calculated scores, providing full
   transparency for data validation.
2. MONTHLY SENTIMENTS ('LLM SA Monthly Data.csv'): A high-level monthly report
   that aggregates all metrics by company and product, ready for dashboards
   and strategic review.

--------------------------------------------------------------------------------
SYSTEM ARCHITECTURE:
This file is organized into the following sections for clarity. An end user can
easily understand the flow of the process by reviewing the purpose of these stages.

1.  IMPORTS & STYLING: Loads necessary software libraries and sets up console colors.
2.  USER INPUTS: **The only section a user needs to edit.**
3.  KEYWORD DEFINITIONS: The business aspects and their related keywords.
4.  CORE ANALYTICAL ENGINE: The functions that perform the analysis, broken down into:
    4.1 - System Initialization & Validation: Confirms all settings are valid and that
          the required linguistic models are operational before any processing begins.
    4.2 - Data Ingestion & Quality Assessment: Loads the source CSV file and performs
          a rigorous quality check, reporting on data integrity.
    4.3 - Natural Language Pre-Processing: Transforms raw review text into a standardized,
          machine-readable format to enable consistent and accurate analysis.
    4.4 - Aspect Relevance Scoring Engine: Deploys a high-performance algorithm to
          quantify the focus of each review on key business topics.
    4.5 - Sentiment Scoring Engine: Applies a validated lexicon-based model to
          score the emotional tone of reviews on a -100 to +100 scale.
    4.6 - Report Generation: Synthesizes millions of data points into the
          two final, business-ready CSV reports.
5.  MAIN WORKFLOW: Executes the entire process in a clear, step-by-step manner.
--------------------------------------------------------------------------------
"""

# === SECTION 1: IMPORTS & STYLING ===
import pandas as pd
from tqdm.auto import tqdm
import os
import sys
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from datetime import datetime
import gc

class Txt:
    GREEN, YELLOW, RED, BLUE, RESET = '\033[92m', '\033[93m', '\033[91m', '\033[94m', '\033[0m'

# === SECTION 2: USER INPUTS ===
USE_PRECISE_METHOD = False
INPUT_FILENAME = 'CIT Energy 2024 TrustPilot v2.csv'
OUTPUT_DIRECTORY = 'output'
OUTPUT_FILENAME_DETAILED = os.path.join(OUTPUT_DIRECTORY, 'reviews_audit_sentiment_calcs.csv')
OUTPUT_FILENAME_AGGREGATED = os.path.join(OUTPUT_DIRECTORY, 'LLM SA Monthly Data.csv')
REVIEW_COLUMN_NAME, DATE_COLUMN_NAME, COMPANY_COLUMN_NAME, PRODUCT_COLUMN_NAME, DEFAULT_PRODUCT_NAME = 'Review', "Date", "Company", "Product", "Energy"
MIN_REVIEW_LENGTH, MIN_RELEVANCE_SCORE = 15, 1

# === SECTION 3: KEYWORD DEFINITIONS ===
ASPECTS_KEYWORDS = {
    "Appointment Scheduling": ['appointment', 'booking', 'schedule', 'engineer', 'visit', 'reschedule', 'cancel', 'time', 'slot', 'booked', 'arrival', 'arrived', 'late', 'no-show'],
    "Customer Service": ['service', 'support', 'helpline', 'call', 'centre', 'agent', 'representative', 'advisor', 'team', 'chat', 'email', 'complaint'],
    "Energy Readings": ['reading', 'readings', 'submit', 'submitting', 'estimate', 'usage', 'consumption'],
    "Meter Installations": ['meter', 'installation', 'smart', 'engineer', 'install', 'new', 'exchange', 'fitted', 'fitting'],
    "Value For Money": ['price', 'cost', 'tariff', 'expensive', 'cheap', 'value', 'overcharge', 'deal', 'saving', 'affordable', 'competitive'],
    "Accounts and Billing": ['account', 'bill', 'billing', 'payment', 'payments', 'paid', 'direct debit', 'statement', 'refund', 'credit', 'charge', 'invoice'],
    # "Response Speed": ['fast', 'quick', 'slow', 'wait', 'waiting', 'delay', 'delayed', 'prompt', 'immediate', 'response'],
    # "Solution Quality": ['solved', 'fixed', 'resolved', 'unresolved', 'helpful', 'useless', 'effective', 'ineffective', 'knowledgeable', 'resolution'],
}

# === SECTION 4: CORE ANALYTICAL ENGINE ===

# --- 4.1: System Initialization & Validation ---
# PURPOSE:  To ensure the script is properly configured and all required
#           data and models are available before starting the analysis.
# PROCESS:  This stage validates configuration settings, checks for the input
#           file, and downloads any necessary linguistic models from NLTK.
# VALUE:    This prevents errors and wasted time by catching setup issues early,
#           ensuring the reliability of the entire process.
#
def initialize_system():
    """Downloads NLP models and validates the configuration settings."""
    resources = [('tokenizers/punkt', 'punkt'), ('sentiment/vader_lexicon', 'vader_lexicon')]
    print(f"\n{Txt.BLUE}🚀 Initialising Natural Language Processing subsystems...{Txt.RESET}")
    for path, name in resources:
        try: nltk.data.find(path)
        except LookupError: nltk.download(name, quiet=True)
    print(f"   {Txt.GREEN}✓ All NLP systems operational{Txt.RESET}")

    print(f"\n{Txt.BLUE}🔍 Validating system configuration...{Txt.RESET}")
    errors = []
    if not os.path.exists(INPUT_FILENAME): errors.append(f"Input file not found at '{INPUT_FILENAME}'")
    if MIN_REVIEW_LENGTH < 0: errors.append("MIN_REVIEW_LENGTH cannot be negative.")
    if errors:
        for error in errors: print(f"   {Txt.RED}❌ CRITICAL ERROR: {error}{Txt.RESET}")
        sys.exit(1)
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    print(f"   {Txt.GREEN}✓ Configuration validated successfully{Txt.RESET}")

# --- 4.2: Data Ingestion & Quality Assessment ---
# PURPOSE:  To load the source file while running a rigorous quality check
#           on the incoming data to ensure analytical integrity.
# PROCESS:  The script reads the entire source CSV into memory, then validates columns,
#           handles date errors, and filters out low-quality reviews.
# VALUE:    This provides clear metrics on data integrity and builds confidence
#           in the final results by transparently reporting on filtered data.
#
def load_and_validate_data(filename, **cols):
    """Loads input data and performs a comprehensive quality assessment."""
    print(f"\n{Txt.BLUE}📊 Loading and assessing data from {filename}...{Txt.RESET}")
    df = pd.read_csv(filename)
    initial_count = len(df)
    
    df.columns = df.columns.str.strip()
    for key in ['review_col', 'date_col', 'company_col']:
        if cols[key] not in df.columns: raise ValueError(f"Required column '{cols[key]}' not found.")
    
    df['Month'] = pd.to_datetime(df[cols['date_col']], errors='coerce').dt.to_period('M').dt.to_timestamp()
    df['_review_length'] = df[cols['review_col']].fillna('').str.len()
    
    initial_rows = len(df)
    df.dropna(subset=['Month'], inplace=True)
    invalid_dates = initial_rows - len(df)
    df = df[df['_review_length'] >= MIN_REVIEW_LENGTH].copy()
    short_reviews = initial_rows - invalid_dates - len(df)
    
    if cols['product_col'] not in df.columns:
        df[cols['product_col']] = cols['default_product']
    
    final_count = len(df)
    quality_score = (final_count / initial_count * 100) if initial_count > 0 else 0
    
    print(f"\n{Txt.BLUE}📈 Data Quality Report:{Txt.RESET}")
    print(f"   - Total reviews loaded: {initial_count:,}")
    if invalid_dates > 0: print(f"   - Reviews removed (invalid date): {invalid_dates:,}")
    if short_reviews > 0: print(f"   - Reviews removed (too short): {short_reviews:,}")
    print(f"   - {Txt.GREEN}✓ Reviews passing quality checks: {final_count:,}{Txt.RESET}")
    print(f"   - {Txt.GREEN}✓ Data quality score: {quality_score:.1f}%{Txt.RESET}")
    
    return df, quality_score

# --- 4.3: Natural Language Pre-Processing ---
# PURPOSE:  To transform raw, unstructured review text into a standardized,
#           machine-readable format for accurate and efficient analysis.
# PROCESS:  This is a one-time "heavy lifting" step. It isolates every individual word
#           ('Tokenization') and algorithmically reduces it to its core 'stem'
#           ('Stemming'). This is done once to maximize speed.
# VALUE:    This sophisticated step ensures our analysis captures all relevant
#           feedback, regardless of the specific tense or phrasing a customer uses.
#
def preprocess_text_data(df, review_col, stemmer):
    """Performs tokenization and stemming on each review ONCE to speed up the script."""
    print(f"\n{Txt.BLUE}🧠 Applying advanced Natural Language Processing...{Txt.RESET}")
    tqdm.pandas(desc=f"   {Txt.BLUE}Analysing linguistic patterns{Txt.RESET}")
    def intelligent_tokenization(text):
        if not isinstance(text, str): return set()
        try: return {stemmer.stem(word) for word in word_tokenize(text.lower()) if word.isalpha()}
        except Exception: return set()
    df['_stemmed_tokens'] = df[review_col].progress_apply(intelligent_tokenization)
    return df

# --- 4.4: Aspect Relevance Scoring Engine ---
# PURPOSE:  To quantify how strongly each review focuses on our key business drivers.
#           (Note: This step is only used in the 'Fast' analysis mode).
# PROCESS:  A high-performance algorithm counts keyword occurrences by comparing
#           sets of words (a computationally optimal method) to generate a
#           'Relevance' score for each business aspect.
# VALUE:    This acts as a "signal-to-noise" filter, allowing us to analyze
#           reviews that are highly relevant to a specific part of the business.
#
def calculate_relevance_scores(df, stemmed_keywords):
    """Calculates relevance scores based on keyword counts using set intersection."""
    print(f"\n{Txt.BLUE}🎯 Calculating Business Aspect Relevance...{Txt.RESET}")
    for aspect, keywords_set in stemmed_keywords.items():
        tqdm.pandas(desc=f"   {Txt.BLUE}Scoring Relevance: {aspect}{Txt.RESET}")
        df[f'Relevance_{aspect}'] = df['_stemmed_tokens'].progress_apply(
            lambda tokens: len(tokens.intersection(keywords_set)) if tokens else 0)
    return df

# --- 4.5: Sentiment Scoring Engine ---
# PURPOSE:  To score the emotional tone of every review. This engine operates
#           in two modes: 'Fast' for speed or 'Precise' for accuracy.
# PROCESS:  Using a lexicon-based model (a dictionary of words scored by human
#           experts), it assigns a sentiment score from -100 to +100. 'Precise
#           Mode' isolates relevant sentences first for a more nuanced score.
# VALUE:    This provides the core emotional KPI, converting subjective
#           customer feelings into objective, measurable data.
#
def calculate_sentiment_scores(df, review_col, stemmed_keywords, precise_mode=False):
    """Calculates sentiment scores based on the selected analysis mode."""
    sid = SentimentIntensityAnalyzer()
    stemmer = PorterStemmer()
    print(f"\n{Txt.BLUE}💡 Scoring Emotional Tone...{Txt.RESET}")
    if not precise_mode:
        tqdm.pandas(desc=f"   {Txt.BLUE}Quantifying sentiment (Fast Mode){Txt.RESET}")
        df['Overall_Sentiment'] = df[review_col].progress_apply(lambda text: sid.polarity_scores(text)['compound'] if isinstance(text, str) else 0)
    else:
        def _get_precise_aspect_sentiment(text, keywords):
            if not isinstance(text, str): return None
            try:
                sentences = sent_tokenize(text)
                review_tokens = {stemmer.stem(word) for word in word_tokenize(text.lower())}
                if not review_tokens.intersection(keywords): return None
                relevant_sentences = [s for s in sentences if {stemmer.stem(w) for w in word_tokenize(s.lower())}.intersection(keywords)]
                if not relevant_sentences: return None
                return sid.polarity_scores(". ".join(relevant_sentences))['compound']
            except Exception: return None
        for aspect, keywords_set in stemmed_keywords.items():
            tqdm.pandas(desc=f"   {Txt.BLUE}Scoring sentiment (Precise): {aspect}{Txt.RESET}")
            df[f'Sentiment_{aspect}'] = df[review_col].progress_apply(lambda text: _get_precise_aspect_sentiment(text, keywords_set))
    
    sentiment_cols = [col for col in df.columns if 'Sentiment' in col]
    for col in sentiment_cols: df[col] = (df[col] * 100).round(1)
    return df

# --- 4.6: Report Generation ---
# PURPOSE:  To synthesize all processed data into the two final business-ready reports.
# PROCESS:  The script constructs and saves two CSV files (detailed and aggregated)
#           with executive-friendly formatting, naming, and column order.
# VALUE:    This provides the final, tangible outputs needed for strategic
#           analysis, reporting, and decision-making.
#
def generate_final_reports(df, detailed_filename, aggregated_filename, quality_score, precise_mode=False, **cols):
    """Generates the two final output CSV files based on the analysis mode."""
    print(f"\n{Txt.BLUE}📑 Generating final intelligence reports...{Txt.RESET}")
    os.makedirs(os.path.dirname(detailed_filename), exist_ok=True)
    
    if precise_mode:
        output_cols_detailed = [cols['date_col'], cols['review_col'], cols['company_col'], cols['product_col']] + [f'Sentiment_{aspect}' for aspect in ASPECTS_KEYWORDS.keys() if f'Sentiment_{aspect}' in df.columns]
    else:
        output_cols_detailed = [cols['date_col'], cols['review_col'], cols['company_col'], cols['product_col'], 'Overall_Sentiment'] + [f'Relevance_{aspect}' for aspect in ASPECTS_KEYWORDS.keys()]
    df[output_cols_detailed].to_csv(detailed_filename, index=False)
    print(f"   {Txt.GREEN}✓ Detailed audit file saved to '{detailed_filename}'{Txt.RESET}")

    group_by_cols = ['Month', cols['company_col'], cols['product_col']]
    if precise_mode:
        agg_spec = {f'Sentiment_{aspect}': ['mean', 'std'] for aspect in ASPECTS_KEYWORDS.keys() if f'Sentiment_{aspect}' in df.columns}
        df_agg = df.groupby(group_by_cols).agg(agg_spec).reset_index()
        df_agg.columns = ['_'.join(col).strip('_') for col in df_agg.columns.values]
        df_volume = df.groupby(group_by_cols).size().reset_index(name='Reviews Count')
        df_agg = pd.merge(df_agg, df_volume, on=group_by_cols, how='left')
        for aspect in ASPECTS_KEYWORDS.keys():
            if f'Sentiment_{aspect}_mean' in df_agg.columns: df_agg.rename(columns={f'Sentiment_{aspect}_mean': f'{aspect}_sentiment_score', f'Sentiment_{aspect}_std': f'{aspect}_volatility'}, inplace=True)
    else:
        df_agg_base = df.groupby(group_by_cols).agg({'_review_length': 'count', 'Overall_Sentiment': ['mean', 'std']}).reset_index()
        df_agg_base.columns = ['Month', cols['company_col'], cols['product_col'], 'Reviews Count', 'Sentiment Score', 'Sentiment Volatility']
        df_agg = df_agg_base
        for aspect in ASPECTS_KEYWORDS.keys():
            relevance_col = f'Relevance_{aspect}'
            aspect_df = df[df[relevance_col] >= MIN_RELEVANCE_SCORE].copy()
            if not aspect_df.empty:
                aspect_agg = aspect_df.groupby(group_by_cols).agg({'Overall_Sentiment': ['mean', 'std'], '_review_length': 'count'}).reset_index()
                aspect_agg.columns = ['Month', cols['company_col'], cols['product_col'], f'{aspect}_sentiment_score', f'{aspect}_volatility', f'{aspect}_Volume']
                df_agg = pd.merge(df_agg, aspect_agg, on=group_by_cols, how='left')
    
    df_agg.fillna(0, inplace=True)
    for col in df_agg.columns:
        if pd.api.types.is_numeric_dtype(df_agg[col]):
            if col == 'Reviews Count' or '_Volume' in col or '_Reviews' in col or 'Sentiment' in col: df_agg[col] = df_agg[col].astype(int)
            elif 'score' in col or 'volatility' in col: df_agg[col] = df_agg[col].round(1)
    
    df_agg.rename(columns={'Month': 'Year-Month', cols['product_col']: 'Final Product Category'}, inplace=True)
    df_agg['Data_Quality_Score'] = round(quality_score, 1)
    
    first_cols = ['Year-Month', 'Company', 'Final Product Category', 'Reviews Count']
    if 'Sentiment Score' in df_agg.columns: first_cols.extend(['Sentiment Score', 'Sentiment Volatility'])
    first_cols.append('Data_Quality_Score')
        
    aspect_cols_ordered = []
    for aspect in ASPECTS_KEYWORDS.keys():
        for suffix in ['_Volume', '_sentiment_score', '_volatility', '_Reviews']:
            col_name = f'{aspect}{suffix}'
            if col_name in df_agg.columns: aspect_cols_ordered.append(col_name)
    final_cols = first_cols + aspect_cols_ordered
    df_agg = df_agg[[col for col in final_cols if col in df_agg.columns]]
    df_agg.to_csv(aggregated_filename, index=False, date_format='%Y-%m')
    print(f"   {Txt.GREEN}✓ Aggregated executive summary saved to '{aggregated_filename}'{Txt.RESET}")

# === SECTION 5: MAIN WORKFLOW ===
def main():
    """Main function that orchestrates the entire analytical pipeline."""
    start_time = datetime.now()
    print(f"{Txt.BLUE}================================================================================{Txt.RESET}")
    print(f"{Txt.BLUE}CUSTOMER REVIEW INTELLIGENCE SYSTEM - INITIATED at {start_time.strftime('%Y-%m-%d %H:%M:%S')}{Txt.RESET}")
    print(f"{Txt.BLUE}================================================================================{Txt.RESET}")
    try:
        initialize_system()
        
        cols_map = {'review_col': REVIEW_COLUMN_NAME, 'date_col': DATE_COLUMN_NAME, 'company_col': COMPANY_COLUMN_NAME, 'product_col': PRODUCT_COLUMN_NAME, 'default_product': DEFAULT_PRODUCT_NAME}
        
        data_df, quality_score = load_and_validate_data(INPUT_FILENAME, **cols_map)
        
        stemmer = PorterStemmer()
        stemmed_keywords = {aspect: {stemmer.stem(kw.lower()) for kw in keywords} for aspect, keywords in ASPECTS_KEYWORDS.items()}
        
        if USE_PRECISE_METHOD:
            print(f"\n{Txt.YELLOW}🔬 PRECISION MODE ACTIVATED. Expect surgical accuracy with longer processing time.{Txt.RESET}")
        else:
            print(f"\n{Txt.YELLOW}⚡ FAST MODE ACTIVATED. Expect rapid results with comprehensive coverage.{Txt.RESET}")
            data_df = preprocess_text_data(data_df, REVIEW_COLUMN_NAME, stemmer)
            data_df = calculate_relevance_scores(data_df, stemmed_keywords)
        
        data_df = calculate_sentiment_scores(data_df, REVIEW_COLUMN_NAME, stemmed_keywords, precise_mode=USE_PRECISE_METHOD)
        
        if '_stemmed_tokens' in data_df.columns:
            del data_df['_stemmed_tokens']; gc.collect()

        generate_final_reports(data_df, OUTPUT_FILENAME_DETAILED, OUTPUT_FILENAME_AGGREGATED, quality_score, precise_mode=USE_PRECISE_METHOD, **cols_map)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        reviews_per_second = len(data_df) / processing_time if processing_time > 0 else 0
        print(f"\n{Txt.GREEN}================================================================================{Txt.RESET}")
        print(f"{Txt.GREEN}✅ ANALYSIS COMPLETE - All processes finished without critical errors.{Txt.RESET}")
        print(f"{Txt.GREEN}================================================================================{Txt.RESET}")
        print(f"  - Total processing time: {processing_time:.2f} seconds")
        print(f"  - Average processing speed: {reviews_per_second:.1f} reviews/second")
        print(f"  - Final reports generated at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  - Data Quality Score: {quality_score:.1f}%")

    except Exception as e:
        print(f"\n\n{Txt.RED}================================================================================{Txt.RESET}")
        print(f"{Txt.RED}❌ A CRITICAL ERROR occurred during execution: {str(e)}{Txt.RESET}")
        print(f"{Txt.RED}================================================================================{Txt.RESET}")
        raise

main()
