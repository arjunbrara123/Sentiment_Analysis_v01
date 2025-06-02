# This script performs sentiment analysis on customer reviews.
# It calculates overall sentiment and sentiment for specific aspects (topics)
# defined by keywords, with an enhanced method for aspect sentiment differentiation.
# Results are saved in CSV files, with monthly summaries broken down by product and company.
# No LLMs are used; sentiment is based on NLTK VADER and relevance on TF-IDF.

################################################################################
# Config Block: Please edit the values below to configure the script.
# Ensure this script and your input CSV file are in a location where you have
# permission to write files, as output files will be saved here.
################################################################################

# INPUT_CSV_PATH: Path to the raw CSV file containing customer reviews.
#INPUT_CSV_PATH = "BG All TrustPilot Data_Prod_Cat_4.1_v01.csv"  # <<< --- !!! EDIT THIS LINE TO YOUR FILE PATH !!!
INPUT_CSV_PATH = "CIT Energy Trustpilot All.csv"

# --- Configurable Column Names from your Input CSV ---
# These MUST match the column headers in your CSV file.
REVIEW_COLUMN_NAME = "Review"  # Column containing the review text
DATE_COLUMN_NAME = "Date"  # Column with review dates (dd/mm/yyyy)
PRODUCT_COLUMN_NAME = "prod_category"  # Column for product/service category
COMPANY_COLUMN_NAME = "Company"  # Column for company identifier

# ASPECT_KEYWORDS: Define your aspects (topics of interest) and their keywords.
# Each aspect is a key (string), and the value is a list of keyword strings.
ASPECT_KEYWORDS = {
    "Appointment Scheduling": ["appointment", "schedule", "booking", "availability", "slot", "reschedule",
                               "booking system", "appointment time", "booked", "cancelled appointment",
                               "manage booking"],
    "Customer Service": ["customer service", "support", "agent", "representative", "helpline", "complaint", "polite",
                         "rude", "helpful", "unhelpful", "assistance", "communication", "staff", "query", "enquiry",
                         "contact centre"],
    "Response Speed": ["response", "speed", "quick", "fast", "slow", "wait time", "prompt", "delay", "waiting",
                       "quickly", "timely", "turnaround", "responsiveness"],
    "Engineer Experience": ["engineer", "technician", "visit", "on-site", "professional", "knowledgeable", "skill",
                            "attitude", "courteous", "efficient staff", "workmanship", "expertise", "site visit"],
    "Solution Quality": ["solution", "quality", "fix", "resolve", "issue", "problem", "effective", "reliable", "worked",
                         "failed", "outcome", "resolution", "successful", "poor quality", "fixed"],
    "Value For Money": ["value", "price", "cost", "expensive", "cheap", "affordable", "worth", "tariff", "charges",
                        "pricing", "overpriced", "good value", "fair price", "economical", "cost-effective"],
    "Meter Installations": ["meter", "installation", "install", "fitting", "new meter", "smart meter", "meter exchange",
                            "engineer visit for meter", "meter setup", "meter replaced", "commissioning"],
    "Energy Readings": ["reading", "energy usage", "consumption", "meter reading", "estimate", "actual", "usage data",
                        "smart meter reading", "inaccurate reading", "submit reading", "view readings"],
    "Accounts & Billing": ["account", "billing", "invoice", "payment", "charge", "refund", "statement", "direct debit",
                           "overcharge", "billing error", "final bill", "credit", "debt", "manage account"]
}

# ASPECT_PROCESSING_FLAGS: Control which aspects are processed.
# Keys MUST EXACTLY MATCH those in ASPECT_KEYWORDS. Set to False to skip an aspect.
ASPECT_PROCESSING_FLAGS = {
    "Appointment Scheduling": True,
    "Customer Service": True,
    "Response Speed": True,
    "Engineer Experience": True,
    "Solution Quality": True,
    "Value For Money": True,
    "Meter Installations": True,
    "Energy Readings": True,
    "Accounts & Billing": True
}

# SENTIMENT_METHOD: Method for overall sentiment calculation.
# Currently fixed to "vader".
SENTIMENT_METHOD = "vader"

# RELEVANCE_METHOD: Method for calculating review-aspect relevance.
# Currently fixed to "tfidf_cosine".
RELEVANCE_METHOD = "tfidf_cosine"

# THRESHOLD_RULE: Rule for filtering reviews in the monthly summary for aspect averages.
# Currently fixed to "median_per_aspect".
THRESHOLD_RULE = "median_per_aspect"

# END OF CONFIG BLOCK
################################################################################

print("Advanced Sentiment Analyzer (GG Edition) - Starting...")
print("-" * 70)

# 1. Import libraries and check dependencies
print("\nSECTION 1: Initializing and checking requirements...")

import sys
import os
import importlib.util
from datetime import datetime
from pathlib import Path
import time  # For basic timing if needed, not for tqdm replacement logic

# tqdm (optional progress bar) setup
TQDM_AVAILABLE = False
try:
    from tqdm.auto import tqdm

    tqdm.pandas()  # Enable progress_apply for pandas
    TQDM_AVAILABLE = True
    print("Progress bars (tqdm) available and enabled.")
except ImportError:
    print("Optional package 'tqdm' not found. Progress bars will not be shown for some operations.")


    # Dummy tqdm function if not available
    def tqdm(iterable, *args, **kwargs):  # type: ignore
        # Remove desc from kwargs if it exists, as print doesn't use it like tqdm
        _ = kwargs.pop('desc', None)
        return iterable


def optional_iterable_progress(iterable_object, description="Processing items"):
    """Wraps an iterable with tqdm if available, otherwise prints a start message."""
    if TQDM_AVAILABLE:
        return tqdm(iterable_object, desc=description, leave=False)
    else:
        if description:
            print(f"{description}...")
        return iterable_object


# Package dependency check
REQUIRED_PACKAGES = {
    "pandas": "pandas",
    "numpy": "numpy",
    "nltk": "nltk",
    "sklearn": "scikit-learn"
}
missing_packages_list = []
for package_name_check, install_name in REQUIRED_PACKAGES.items():
    if importlib.util.find_spec(package_name_check) is None:
        missing_packages_list.append(install_name)

if missing_packages_list:
    error_msg = "Error: The following required Python packages are missing:\n"
    for pkg_install in missing_packages_list:
        error_msg += f"  - {pkg_install} (install with: pip install {pkg_install})\n"
    if "nltk" in [p_name.lower() for p_name in missing_packages_list]:
        error_msg += "\nAfter installing nltk (pip install nltk), this script will attempt to download "
        error_msg += "the VADER lexicon. If it fails, you might need to run Python and execute: "
        error_msg += "import nltk; nltk.download('vader_lexicon')"
    print(error_msg)
    sys.exit("Exiting due to missing packages. Please install them and re-run the script.")

import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("All required packages are present.")
print("-" * 70)

# 2. Set up sentiment and relevance tools
print("\nSECTION 2: Setting up analysis tools...")

# Download VADER lexicon if not already present
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
    print("NLTK VADER lexicon found.")
except nltk.downloader.DownloadError:
    print("NLTK VADER lexicon not found. Attempting to download (one-time setup)...")
    try:
        nltk.download('vader_lexicon', quiet=True)  # quiet to reduce console noise
        nltk.data.find('sentiment/vader_lexicon.zip')  # Verify download
        print("NLTK VADER lexicon downloaded successfully.")
    except Exception as e_nltk_download:
        print(f"Error: Could not download NLTK VADER lexicon automatically: {e_nltk_download}")
        print("Please try downloading it manually. Open a Python console and run:")
        print("import nltk")
        print("nltk.download('vader_lexicon')")
        sys.exit("Exiting due to missing VADER lexicon.")

# Initialize VADER sentiment analyzer
vader_analyzer = SentimentIntensityAnalyzer()
print(f"Sentiment analysis method: {SENTIMENT_METHOD} (using NLTK VADER).")

# Initialize TF-IDF Vectorizer for relevance
tfidf_vec = TfidfVectorizer(stop_words='english', min_df=1)  # min_df=1 for short keyword lists
print(f"Relevance calculation method: {RELEVANCE_METHOD} (using TF-IDF and Cosine Similarity).")
print("-" * 70)

# 3. Load input data
print("\nSECTION 3: Loading and preparing input data...")
print(f"Attempting to load data from: {INPUT_CSV_PATH}")

if not Path(INPUT_CSV_PATH).is_file():
    print(f"Error: Input CSV file not found at '{INPUT_CSV_PATH}'.")
    print("Please check the INPUT_CSV_PATH in the Config block and ensure the file exists.")
    sys.exit("Exiting due to missing input file.")

try:
    # This step loads the raw data exported by your team.
    df_main = pd.read_csv(INPUT_CSV_PATH)
    original_columns = df_main.columns.tolist()  # Store original column order BEFORE any modifications
    print(f"Successfully loaded {len(df_main)} rows and {len(df_main.columns)} columns.")
except Exception as e_csv_load:
    print(f"Error: Could not read the CSV file. Please ensure it's a valid CSV. Details: {e_csv_load}")
    sys.exit("Exiting due to CSV loading error.")

# Validate essential columns based on Config
essential_input_cols = [REVIEW_COLUMN_NAME, DATE_COLUMN_NAME, PRODUCT_COLUMN_NAME, COMPANY_COLUMN_NAME]
missing_essential_cols = [col for col in essential_input_cols if col not in df_main.columns]
if missing_essential_cols:
    print(f"Error: The CSV file is missing these configured required columns: {', '.join(missing_essential_cols)}.")
    print("Please check your CSV file or the Config block (REVIEW_COLUMN_NAME, etc.).")
    sys.exit("Exiting due to missing essential configured columns.")

# Handle missing values in essential columns by dropping affected rows
initial_rows = len(df_main)
df_main.dropna(subset=essential_input_cols, inplace=True)
rows_dropped = initial_rows - len(df_main)
if rows_dropped > 0:
    print(
        f"Dropped {rows_dropped} rows due to missing values in one or more essential columns: {', '.join(essential_input_cols)}.")

# Ensure correct data types for key columns
df_main[REVIEW_COLUMN_NAME] = df_main[REVIEW_COLUMN_NAME].astype(str)
df_main[PRODUCT_COLUMN_NAME] = df_main[PRODUCT_COLUMN_NAME].astype(str)
df_main[COMPANY_COLUMN_NAME] = df_main[COMPANY_COLUMN_NAME].astype(str)

# Parse 'Date' column. Invalid date formats will result in NaT.
df_main[DATE_COLUMN_NAME] = pd.to_datetime(df_main[DATE_COLUMN_NAME], format='%d/%m/%Y', errors='coerce')
initial_rows_after_essential_nan = len(df_main)
df_main.dropna(subset=[DATE_COLUMN_NAME], inplace=True)  # Drop rows where date parsing failed
rows_dropped_invalid_dates = initial_rows_after_essential_nan - len(df_main)
if rows_dropped_invalid_dates > 0:
    print(
        f"Dropped {rows_dropped_invalid_dates} rows due to invalid date formats in '{DATE_COLUMN_NAME}'. Expected dd/mm/yyyy.")

if df_main.empty:
    print("Error: No valid data remaining after cleaning (checking missing values and invalid dates).")
    print("Please check your input CSV file content and format.")
    sys.exit("Exiting due to no processable data.")

print(f"Data loaded and preprocessed. {len(df_main)} reviews remaining for analysis.")
print("-" * 70)

# 4. Compute overall sentiment
print("\nSECTION 4: Calculating overall sentiment for each review...")


# This function applies VADER to get the compound sentiment score.
def calculate_overall_sentiment(text_data):
    if pd.isna(text_data) or not isinstance(text_data, str) or not text_data.strip():
        return np.nan
    return vader_analyzer.polarity_scores(text_data)['compound']


# Applying to the review column. Result is between -1 (negative) and +1 (positive).
overall_sentiment_col_name = 'overall_sentiment'  # Internal standard name
if TQDM_AVAILABLE:
    df_main[overall_sentiment_col_name] = df_main[REVIEW_COLUMN_NAME].progress_apply(calculate_overall_sentiment)
else:
    print("Calculating overall sentiment (this may take some time)...")
    df_main[overall_sentiment_col_name] = df_main[REVIEW_COLUMN_NAME].apply(calculate_overall_sentiment)
df_main.dropna(subset=[overall_sentiment_col_name], inplace=True)  # Drop if sentiment calculation failed

print("Overall sentiment calculation complete.")
print("-" * 70)

# 5. Compute aspect relevance and then aspect sentiment (using Normalization by Maximum Relevance)
print("\nSECTION 5: Calculating aspect relevance and sentiment...")

# Determine which aspects are enabled for processing
enabled_aspects = [aspect for aspect, flag in ASPECT_PROCESSING_FLAGS.items() if flag and aspect in ASPECT_KEYWORDS]

actual_relevance_cols = []  # Will store names of relevance columns successfully created
actual_aspect_sentiment_cols = []  # Will store names of aspect sentiment columns successfully created

if not enabled_aspects:
    print("No aspects are enabled for processing in ASPECT_PROCESSING_FLAGS or defined in ASPECT_KEYWORDS.")
else:
    print(f"Enabled aspects for processing: {', '.join(enabled_aspects)}")
    # Fit TF-IDF model on all review texts. This learns word importance.
    print("Fitting TF-IDF model on review texts...")
    if not df_main[REVIEW_COLUMN_NAME].empty:
        try:
            review_tfidf_matrix_fitted = tfidf_vec.fit_transform(df_main[REVIEW_COLUMN_NAME])
            print("TF-IDF model fitted successfully.")
        except Exception as e_tfidf_fit:
            print(f"Error fitting TF-IDF model: {e_tfidf_fit}. Aspect analysis will be skipped.")
            enabled_aspects = []  # Disable further aspect processing
    else:
        print("No review texts available to fit TF-IDF model. Skipping aspect analysis.")
        enabled_aspects = []

    if enabled_aspects:  # Proceed if TF-IDF fitting was successful and aspects are enabled
        for aspect_name in optional_iterable_progress(enabled_aspects, description="Calculating aspect relevance"):
            keywords = ASPECT_KEYWORDS[aspect_name]
            relevance_col = f'relevance_{aspect_name}'  # e.g., relevance_Customer Service

            if not keywords:
                print(f"  - Warning: Aspect '{aspect_name}' has no keywords defined. Assigning zero relevance.")
                df_main[relevance_col] = 0.0
            else:
                keywords_as_string = ' '.join(keywords)
                try:
                    # Transform keywords using the same fitted TF-IDF model
                    keyword_tfidf_vector = tfidf_vec.transform([keywords_as_string])
                    # Calculate cosine similarity between each review and the aspect's keywords
                    relevance_scores_for_aspect = cosine_similarity(review_tfidf_matrix_fitted,
                                                                    keyword_tfidf_vector).flatten()
                    df_main[relevance_col] = relevance_scores_for_aspect
                except Exception as e_relevance:
                    print(f"  - Error calculating relevance for aspect '{aspect_name}': {e_relevance}. Assigning NaN.")
                    df_main[relevance_col] = np.nan  # Mark as NaN if error

            if relevance_col in df_main.columns:  # If column was created (even if all NaN or 0)
                actual_relevance_cols.append(relevance_col)

        print("Aspect relevance calculation complete.")

        # New Aspect Sentiment Calculation (Normalization by Maximum Relevance)
        print("Calculating aspect sentiment scores using Normalization by Maximum Relevance method...")

        if actual_relevance_cols:  # Only if there are relevance scores to process
            # Calculate max relevance per review across all enabled aspects that had relevance calculated
            df_main['max_relevance_for_review'] = df_main[actual_relevance_cols].max(axis=1)

            for aspect_name in optional_iterable_progress(enabled_aspects,
                                                          description="Calculating aspect sentiment scores"):
                relevance_col = f'relevance_{aspect_name}'
                # Construct aspect sentiment column name (e.g., "Customer Service_sentiment_score")
                aspect_sentiment_col = f'{aspect_name}_sentiment_score'  # This matches review-level naming

                if relevance_col not in df_main.columns:  # Should exist if in actual_relevance_cols
                    df_main[aspect_sentiment_col] = 0.0  # Default if relevance calculation failed for this aspect
                else:
                    # Apply the formula: overall_sentiment * (relevance_AspectX / Max_Relevance_for_Review)
                    # Ensure to handle division by zero if max_relevance_for_review is 0
                    df_main[aspect_sentiment_col] = np.where(
                        df_main['max_relevance_for_review'] > 0,  # Condition
                        df_main[overall_sentiment_col_name] * (
                                    df_main[relevance_col].fillna(0) / df_main['max_relevance_for_review']),
                        # Value if true, fill NaN relevance with 0
                        0.0  # Value if false (max relevance is 0, so this aspect's contribution is 0)
                    )
                actual_aspect_sentiment_cols.append(
                    aspect_sentiment_col)  # Add to list regardless of calculation outcome

            df_main.drop(columns=['max_relevance_for_review'], inplace=True)  # Clean up temporary column
            print("Aspect sentiment score calculation complete.")
        else:
            print("No relevance columns were successfully generated to calculate aspect sentiment scores.")
print("-" * 70)

# 6. Write review-level CSV
print("\nSECTION 6: Preparing and writing detailed review-level output CSV...")

# Create 'output' subfolder in the same directory as the input CSV
input_file_path_obj = Path(INPUT_CSV_PATH).resolve()
output_dir_path = input_file_path_obj.parent / "output"
try:
    output_dir_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory ensured at: {output_dir_path}")
except Exception as e_mkdir:
    print(f"Warning: Could not create output directory '{output_dir_path}'. Error: {e_mkdir}")
    print(f"Output files will be saved in the script's current directory: {Path.cwd()}")
    output_dir_path = Path.cwd()  # Fallback

# Generate a timestamp for unique filenames
current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
input_filename_stem = input_file_path_obj.stem

# Define column order for review-level output
# 1. Original columns (as read from input)
# 2. Overall sentiment
# 3. All relevance columns (sorted alphabetically by aspect name for consistency)
# 4. All aspect sentiment score columns (sorted alphabetically by aspect name)

# Use the 'original_columns' list captured at the beginning
# Filter to ensure these columns still exist in df_main (though they should)
final_original_cols_in_df = [col for col in original_columns if col in df_main.columns]

# Get generated relevance and aspect sentiment columns that actually exist in df_main
# Sort them alphabetically for consistent grouping
sorted_actual_relevance_cols = sorted([col for col in actual_relevance_cols if col in df_main.columns])
sorted_actual_aspect_sentiment_cols = sorted([col for col in actual_aspect_sentiment_cols if col in df_main.columns])

# Assemble the final column order
ordered_review_level_columns = final_original_cols_in_df + \
                               ([overall_sentiment_col_name] if overall_sentiment_col_name in df_main.columns else []) + \
                               sorted_actual_relevance_cols + \
                               sorted_actual_aspect_sentiment_cols

# Ensure no duplicate columns and all desired columns are present, preserving order
seen_cols_review_lvl = set()
final_ordered_unique_cols_review_lvl = []
for col_review_lvl in ordered_review_level_columns:
    if col_review_lvl not in seen_cols_review_lvl and col_review_lvl in df_main.columns:
        final_ordered_unique_cols_review_lvl.append(col_review_lvl)
        seen_cols_review_lvl.add(col_review_lvl)

# Add any other columns that might have been created in df_main but not in the planned order (should be rare)
for col_review_lvl in df_main.columns:
    if col_review_lvl not in seen_cols_review_lvl:
        final_ordered_unique_cols_review_lvl.append(col_review_lvl)

review_level_csv_name = f"{input_filename_stem}_review_level_analysis_GG_{current_timestamp}.csv"  # GG tag added
review_level_output_full_path = output_dir_path / review_level_csv_name

try:
    df_main_output_review_lvl = df_main[final_ordered_unique_cols_review_lvl]  # Reorder DataFrame
    df_main_output_review_lvl.to_csv(review_level_output_full_path, index=False, encoding='utf-8-sig',
                                     float_format='%.2f')
    print(f"Review-level data successfully written to: {review_level_output_full_path}")
except Exception as e_review_csv:
    print(f"Error: Could not write review-level CSV file. Details: {e_review_csv}")
print("-" * 70)

# 7. Aggregate to monthly summary (by Product and Company)
print("\nSECTION 7: Aggregating data for monthly summary (by Product & Company)...")

if df_main.empty:
    print("No data available for monthly aggregation. Skipping monthly summary.")
    df_monthly_summary = pd.DataFrame()
else:
    df_for_monthly_agg = df_main.copy()
    df_for_monthly_agg['Month_Start'] = df_for_monthly_agg[DATE_COLUMN_NAME].dt.to_period('M').dt.start_time

    grouping_cols_monthly = ['Month_Start', PRODUCT_COLUMN_NAME, COMPANY_COLUMN_NAME]

    monthly_group_stats = df_for_monthly_agg.groupby(grouping_cols_monthly).agg(
        total_reviews_in_group=(REVIEW_COLUMN_NAME, 'count'),
        # This will be renamed to 'Sentiment Score' for the output
        avg_overall_sentiment_for_dashboard=(overall_sentiment_col_name, 'mean')
    ).reset_index()
    monthly_group_stats.rename(columns={'avg_overall_sentiment_for_dashboard': 'Sentiment Score'}, inplace=True)

    if not enabled_aspects or not actual_aspect_sentiment_cols:  # If no aspects processed
        print(
            "No aspects processed or no aspect sentiment columns generated; monthly summary will only contain total reviews and overall sentiment scores.")
        df_monthly_summary = monthly_group_stats
    else:
        print(f"Aggregating monthly aspect data using threshold rule: '{THRESHOLD_RULE}'")
        aspect_median_relevances = {}
        if THRESHOLD_RULE == "median_per_aspect":
            for aspect_name_median in enabled_aspects:
                relevance_col_median = f'relevance_{aspect_name_median}'
                if relevance_col_median in df_for_monthly_agg.columns and df_for_monthly_agg[
                    relevance_col_median].notna().any():
                    aspect_median_relevances[aspect_name_median] = df_for_monthly_agg[relevance_col_median].median()
                else:
                    aspect_median_relevances[aspect_name_median] = np.nan
                    print(
                        f"  - Warning: Median relevance for '{aspect_name_median}' cannot be computed. Thresholding may be affected.")

        all_monthly_aspect_data_list = []
        for group_keys, group_df in optional_iterable_progress(df_for_monthly_agg.groupby(grouping_cols_monthly),
                                                               description=f"Aggregating by Month, {PRODUCT_COLUMN_NAME}, {COMPANY_COLUMN_NAME}"):
            month_val, product_val, company_val = group_keys
            current_group_aspect_summary = {'Month_Start': month_val, PRODUCT_COLUMN_NAME: product_val,
                                            COMPANY_COLUMN_NAME: company_val}

            for aspect_name_agg in enabled_aspects:
                # This is the individual score column from df_main (calculated with Normalization by Max Relevance)
                individual_aspect_sent_col_agg = f'{aspect_name_agg}_sentiment_score'
                relevance_col_agg = f'relevance_{aspect_name_agg}'

                # Dashboard output column name for average aspect sentiment
                dashboard_avg_aspect_sent_col_name = f'{aspect_name_agg}_sentiment_score'
                dashboard_aspect_count_col_name = f'count_relevant_reviews_{aspect_name_agg}'

                if individual_aspect_sent_col_agg not in group_df.columns or relevance_col_agg not in group_df.columns:
                    current_group_aspect_summary[dashboard_avg_aspect_sent_col_name] = np.nan
                    current_group_aspect_summary[dashboard_aspect_count_col_name] = 0
                    continue

                relevant_reviews_for_aspect_avg = group_df
                if THRESHOLD_RULE == "median_per_aspect" and aspect_name_agg in aspect_median_relevances and pd.notna(
                        aspect_median_relevances[aspect_name_agg]):
                    current_threshold = aspect_median_relevances[aspect_name_agg]
                    relevant_reviews_for_aspect_avg = group_df[
                        (group_df[relevance_col_agg].fillna(
                            0) >= current_threshold) &  # FillNa for comparison robustness
                        (group_df[individual_aspect_sent_col_agg].notna())
                        ]
                elif THRESHOLD_RULE == "median_per_aspect":
                    relevant_reviews_for_aspect_avg = pd.DataFrame(columns=group_df.columns)
                else:
                    relevant_reviews_for_aspect_avg = group_df[group_df[individual_aspect_sent_col_agg].notna()]

                if not relevant_reviews_for_aspect_avg.empty:
                    current_group_aspect_summary[dashboard_avg_aspect_sent_col_name] = relevant_reviews_for_aspect_avg[
                        individual_aspect_sent_col_agg].mean()
                    current_group_aspect_summary[dashboard_aspect_count_col_name] = len(relevant_reviews_for_aspect_avg)
                else:
                    current_group_aspect_summary[dashboard_avg_aspect_sent_col_name] = np.nan
                    current_group_aspect_summary[dashboard_aspect_count_col_name] = 0
            all_monthly_aspect_data_list.append(current_group_aspect_summary)

        if all_monthly_aspect_data_list:
            df_monthly_aspect_summary_part = pd.DataFrame(all_monthly_aspect_data_list)
            df_monthly_summary = pd.merge(monthly_group_stats, df_monthly_aspect_summary_part, on=grouping_cols_monthly,
                                          how='left')
        else:
            df_monthly_summary = monthly_group_stats

    # Format 'Month_Start' to 'Month' as dd/mm/yyyy string for final output
    if 'Month_Start' in df_monthly_summary.columns:
        df_monthly_summary['Month'] = df_monthly_summary['Month_Start'].dt.strftime('%d/%m/%Y')
        # Drop Month_Start only if 'Month' was successfully created
        if 'Month' in df_monthly_summary.columns:
            df_monthly_summary.drop(columns=['Month_Start'], inplace=True, errors='ignore')

    # --- Final column order for monthly summary - NEW LOGIC for Grouping ---
    if not df_monthly_summary.empty:  # Proceed with ordering if DataFrame is not empty
        base_cols_monthly = ['Month', PRODUCT_COLUMN_NAME, COMPANY_COLUMN_NAME, 'total_reviews_in_group',
                             'Sentiment Score']

        # Filter base_cols_monthly to only include those present in df_monthly_summary
        # This handles cases like 'Month' not being created if 'Month_Start' was missing.
        actual_base_cols_monthly = [col for col in base_cols_monthly if col in df_monthly_summary.columns]

        aspect_count_cols_to_add = []
        aspect_sentiment_cols_to_add_for_dashboard = []

        # Use a consistent order for aspects, e.g., from ASPECT_KEYWORDS or sorted enabled_aspects
        processed_aspect_names_in_order_monthly = [
            aspect for aspect in ASPECT_KEYWORDS.keys() if aspect in enabled_aspects
        ]

        for aspect_name_ordered_monthly in processed_aspect_names_in_order_monthly:
            count_col_monthly = f'count_relevant_reviews_{aspect_name_ordered_monthly}'
            if count_col_monthly in df_monthly_summary.columns:
                aspect_count_cols_to_add.append(count_col_monthly)

            sentiment_col_monthly_dashboard = f'{aspect_name_ordered_monthly}_sentiment_score'  # Dashboard name
            if sentiment_col_monthly_dashboard in df_monthly_summary.columns:
                aspect_sentiment_cols_to_add_for_dashboard.append(sentiment_col_monthly_dashboard)

        final_monthly_cols_ordered = actual_base_cols_monthly + aspect_count_cols_to_add + aspect_sentiment_cols_to_add_for_dashboard
        final_monthly_cols_to_select = [col for col in final_monthly_cols_ordered if col in df_monthly_summary.columns]

        df_monthly_summary = df_monthly_summary[final_monthly_cols_to_select]

print("Monthly aggregation by Product & Company complete.")
print("-" * 70)

# 8. Write monthly CSV and finish
print("\nSECTION 8: Writing monthly summary CSV and finishing up...")

monthly_summary_output_full_path_str = "Not generated (no data or empty summary)."
if not df_monthly_summary.empty:
    monthly_summary_csv_name = f"{input_filename_stem}_monthly_summary_GG_{current_timestamp}.csv"  # GG tag added
    monthly_summary_output_full_path = output_dir_path / monthly_summary_csv_name
    monthly_summary_output_full_path_str = str(monthly_summary_output_full_path)
    try:
        df_monthly_summary.to_csv(monthly_summary_output_full_path, index=False, encoding='utf-8-sig',
                                  float_format='%.3f')
        print(
            f"Monthly summary data (by Product & Company) successfully written to: {monthly_summary_output_full_path}")
    except Exception as e_monthly_csv:
        print(f"Error: Could not write monthly summary CSV file. Details: {e_monthly_csv}")
        monthly_summary_output_full_path_str = f"Error during generation: {e_monthly_csv}"
else:
    print("No monthly summary data was generated to write to a file.")

print("-" * 70)
print("\n🎉 Advanced Sentiment Analyzer (GG Edition) Script Finished! 🎉")
print("Summary of outputs:")
# Check if path variable exists and if the file itself was created for review-level
review_level_path_exists = 'review_level_output_full_path' in locals() and Path(
    str(review_level_output_full_path)).exists()
print(
    f"  - Review-level details: {review_level_output_full_path if review_level_path_exists else 'Not generated or error.'}")

# Check for monthly summary path and if the df was non-empty (implying an attempt to write)
monthly_summary_path_exists_and_valid = 'monthly_summary_output_full_path' in locals() and Path(
    str(monthly_summary_output_full_path)).exists() and not df_monthly_summary.empty
if monthly_summary_path_exists_and_valid:
    print(f"  - Monthly aggregated summary: {monthly_summary_output_full_path_str}")
else:
    print(f"  - Monthly aggregated summary: {monthly_summary_output_full_path_str}")

if output_dir_path:  # Check if output_dir_path was successfully defined
    if input_file_path_obj:  # Check if input_file_path_obj was successfully defined
        print(
            f"\nOutput files are located in the '{output_dir_path.name}' subfolder, typically within the directory of your input file ({input_file_path_obj.parent}).")
    else:
        print(f"\nOutput files are located in: {output_dir_path}")
print("You can now open these CSV files using spreadsheet software like Excel or load them into your dashboard.")
print("-" * 70)