#!/usr/bin/env python3
"""
review_synthesis_generator.py

This script leverages a Large Language Model (LLM) to transform raw customer reviews
into strategic business intelligence, specifically for PRIMARY_COMPANY_NAME.
It generates two key types of analysis outputs, now saved incrementally for robustness:

1.  Market Comparison Analysis: Compares PRIMARY_COMPANY_NAME's performance for each product
    against the aggregated performance of all *other* companies (market aggregate) within the
    dataset for the same product and year. The analysis focuses on relative Strengths,
    Weaknesses, or areas for Improvement for PRIMARY_COMPANY_NAME, presented in a rich,
    themed HTML format. Results are appended to the CSV as they are generated.

2.  Direct Competitor Comparison Analysis (Optional): Generates detailed head-to-head comparisons
    between PRIMARY_COMPANY_NAME and specific, named competitors for common products across key
    business aspects. This analysis incorporates pre-calculated sentiment scores. Results are
    appended to the CSV as they are generated.

The output is formatted in HTML (within the 'Analysis' fields) for easy integration.

HOW TO USE:
  (Instructions remain largely the same, user should be aware that output files
   will be created early and populated as the script runs.)
  ... (rest of instructions)

SCRIPT SECTIONS OVERVIEW:
  (Overview remains the same)
  ... (rest of overview)
"""

import pandas as pd
import time
from datetime import datetime
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv
import os
import sys
from tqdm import tqdm
import logging
import numpy as np  # For NaN handling if needed

# --- SECTION 1: CONFIGURATION & SETUP ---
# Purpose: Define user-configurable parameters.
# ------------------------------------------

# --- Input Data Configuration (Customer Reviews) ---
INPUT_FILE = "test_reviews_01.csv"  # Ensure this file exists and contains reviews for multiple companies
INPUT_FILE = "annotated_reviews_energy_nollm_2024_trustpilot.csv"
COL_DATE = 'Date'  # Column name for the review date
COL_REVIEW = 'Review'  # Column name for the actual review content
COL_PRODUCT = 'Product'  # Column name for the product/service category the review refers to
COL_REVIEW_COMPANY = 'Company'  # Column name identifying the company the review is about

# --- LLM & API Configuration ---
MODEL = "gpt-4.1-mini"
API_KEY = ""
REQUEST_DELAY_SECONDS = 1

# --- Analysis Scope Configuration ---
PRIMARY_COMPANY_NAME = "British Gas"  # This is "our" company. Adjusted based on user feedback.
MARKET = "Energy"

MARKET_COMPARE_FOCUS_TYPES = ["Strength", "Weakness", "Improvement"]
MAX_REVIEWS_FOR_SYNTHESIS = 50

# --- Direct Competitor Comparison Insights Configuration ---
GENERATE_DIRECT_COMPETITOR_INSIGHTS = True

# --- Sentiment Data Input Configuration (for Direct Competitor Insights) ---
SENTIMENT_INPUT_FILE = "sentiment_scores_input.csv"
SENTIMENT_INPUT_FILE = "energy/LLM SA Monthly Data.csv"  # Example: "company_sentiment_scores.csv"
COL_SENTIMENT_YEAR_MONTH = 'Year-Month'
COL_SENTIMENT_COMPANY = 'Company'
COL_SENTIMENT_PRODUCT = 'Final Product Category'

if MARKET == "Services":
    DIRECT_COMPETITOR_ASPECTS = [
        "Appointment Scheduling", "Customer Service", "Response Speed",
        "Engineer Experience", "Solution Quality", "Value For Money"
    ]
elif MARKET == "Energy":  # Adjusted
    DIRECT_COMPETITOR_ASPECTS = [
        "Appointment Scheduling", "Customer Service", "Energy Readings",
        "Engineer Experience", "Meter Installations", "Value For Money"
    ]
else:
    DIRECT_COMPETITOR_ASPECTS = []

# --- Output Configuration ---
NO_INSIGHT_PLACEHOLDER = "No specific insights found in the provided reviews for this category/comparison."

# --- Constants ---
ASPECT_STYLES_EXAMPLES_FOR_PROMPT = {
    "Appointment Scheduling": '<span style="background: lightcyan">⌚ Appointment Scheduling</span>',
    "Customer Service": '<span style="background: mistyrose">📞 Customer Service</span>',
    "Response Speed": '<span style="background: PapayaWhip">🥇 Response Speed</span>',
    "Engineer Experience": '<span style="background: lightcyan">🧑‍🔧 Engineer Experience</span>',
    "Solution Quality": '<span style="background: lavenderblush">🧠 Solution Quality</span>',
    "Value For Money": '<span style="background: Honeydew">💵 Value For Money</span>',
    "Energy Readings": '<span style="background: lightyellow">⚡ Energy Readings</span>',
    "Meter Installations": '<span style="background: lightblue">🛠️ Meter Installations</span>',
}

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')


# --- SECTION 2: VALIDATION FUNCTIONS ---
def validate_config():
    """Validates critical configuration settings."""
    if not INPUT_FILE or not os.path.isfile(INPUT_FILE):
        logging.error(f"Configuration Error: Review input file '{INPUT_FILE}' not found or not specified.")
        sys.exit(1)
    if not COL_DATE or not COL_REVIEW or not COL_PRODUCT or not COL_REVIEW_COMPANY:
        logging.error(
            "Configuration Error: All review input column names (COL_DATE, COL_REVIEW, COL_PRODUCT, COL_REVIEW_COMPANY) must be specified.")
        sys.exit(1)
    if not MODEL:
        logging.error("Configuration Error: LLM model (MODEL) is not specified.")
        sys.exit(1)
    if not PRIMARY_COMPANY_NAME:
        logging.error("Configuration Error: PRIMARY_COMPANY_NAME must be specified.")
        sys.exit(1)
    if not isinstance(MAX_REVIEWS_FOR_SYNTHESIS, int) or MAX_REVIEWS_FOR_SYNTHESIS <= 0:
        logging.error("Configuration Error: MAX_REVIEWS_FOR_SYNTHESIS must be a positive whole number.")
        sys.exit(1)
    if not isinstance(REQUEST_DELAY_SECONDS, (int, float)) or REQUEST_DELAY_SECONDS < 0:
        logging.error("Configuration Error: REQUEST_DELAY_SECONDS must be a non-negative number.")
        sys.exit(1)
    if not isinstance(GENERATE_DIRECT_COMPETITOR_INSIGHTS, bool):
        logging.error("Configuration Error: GENERATE_DIRECT_COMPETITOR_INSIGHTS must be set to True or False.")
        sys.exit(1)

    if GENERATE_DIRECT_COMPETITOR_INSIGHTS:
        if not SENTIMENT_INPUT_FILE or not os.path.isfile(SENTIMENT_INPUT_FILE):
            logging.error(
                f"Configuration Error: Sentiment input file '{SENTIMENT_INPUT_FILE}' not found or not specified, but GENERATE_DIRECT_COMPETITOR_INSIGHTS is True.")
            sys.exit(1)
        if not COL_SENTIMENT_YEAR_MONTH or not COL_SENTIMENT_COMPANY or not COL_SENTIMENT_PRODUCT:
            logging.error(
                "Configuration Error: All sentiment input column names (COL_SENTIMENT_YEAR_MONTH, COL_SENTIMENT_COMPANY, COL_SENTIMENT_PRODUCT) must be specified if GENERATE_DIRECT_COMPETITOR_INSIGHTS is True.")
            sys.exit(1)
        if not DIRECT_COMPETITOR_ASPECTS:
            logging.warning(
                "Configuration Warning: GENERATE_DIRECT_COMPETITOR_INSIGHTS is True, but DIRECT_COMPETITOR_ASPECTS list is empty. No direct competitor comparison insights will be generated.")

    if not MARKET_COMPARE_FOCUS_TYPES:
        logging.warning(
            "Configuration Warning: MARKET_COMPARE_FOCUS_TYPES list is empty. No market comparison analysis will be generated.")
    logging.info("Configuration settings validated successfully.")


# --- SECTION 3: API & CLIENT INITIALIZATION ---
def get_api_key_from_env_or_config():
    """Retrieves the OpenAI API key."""
    if API_KEY:
        logging.info("Using OpenAI API key from script configuration (API_KEY variable).")
        return API_KEY
    load_dotenv()
    env_key = os.getenv('OPENAI_API_KEY')
    if not env_key:
        logging.error(
            "API Key Error: OpenAI API key not found. Set API_KEY variable or environment variable 'OPENAI_API_KEY'.")
        sys.exit(1)
    logging.info("Using OpenAI API key from environment variable 'OPENAI_API_KEY'.")
    return env_key


def initialize_openai_client(api_key_value):
    """Initializes the OpenAI client."""
    try:
        client = OpenAI(api_key=api_key_value)
        logging.info("OpenAI client initialized successfully.")
        return client
    except OpenAIError as e:
        logging.error(f"API Initialization Error: Failed to initialize OpenAI client: {e}")
        sys.exit(1)


# --- SECTION 4: DATA LOADING & PREPARATION ---
def load_and_validate_input_df(filepath, date_col, review_col, product_col, review_company_col):
    """Loads, validates, and cleans review data from CSV."""
    try:
        df = pd.read_csv(filepath)
        logging.info(f"Successfully loaded review input CSV: {filepath}")
    except FileNotFoundError:
        logging.error(f"File Error: Review input file '{filepath}' not found.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"File Error: Could not read review CSV file '{filepath}': {e}")
        sys.exit(1)

    required_input_cols = [date_col, review_col, product_col, review_company_col]
    missing_cols = [col for col in required_input_cols if col not in df.columns]
    if missing_cols:
        logging.error(f"Review Data Error: Input CSV is missing required columns: {', '.join(missing_cols)}.")
        sys.exit(1)

    df.rename(columns={
        date_col: 'Date', review_col: 'Review', product_col: 'Product', review_company_col: 'ReviewedCompany'
    }, inplace=True)

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    initial_rows = len(df)
    df.dropna(subset=['Date'], inplace=True)
    if len(df) < initial_rows:
        logging.warning(f"Data Cleaning (Reviews): Removed {initial_rows - len(df)} rows with invalid dates.")

    essential_cols = ['Review', 'Product', 'ReviewedCompany']
    initial_rows = len(df)
    df.dropna(subset=essential_cols, inplace=True)
    for col in essential_cols:  # Ensure no empty strings masquerading as data
        df = df[df[col].astype(str).str.strip() != '']
    if len(df) < initial_rows:
        logging.warning(
            f"Data Cleaning (Reviews): Removed {initial_rows - len(df)} rows with empty essential fields (Review, Product, ReviewedCompany).")

    if df.empty:
        logging.error("Review Data Error: No valid review data found after cleaning. Cannot proceed.")
        sys.exit(1)

    df['Year'] = df['Date'].dt.year
    df['ReviewedCompanyLower'] = df['ReviewedCompany'].str.strip().str.lower()  # For matching
    df['ProductLower'] = df['Product'].str.strip().str.lower()  # For matching
    logging.info("Review input CSV loaded, validated, and cleaned.")
    return df


def load_and_prepare_sentiment_data(filepath, year_month_col, company_col, product_col, aspects_list):
    """Loads, validates, and prepares sentiment data from CSV for lookup."""
    try:
        df_sentiment = pd.read_csv(filepath)
        logging.info(f"Successfully loaded sentiment input CSV: {filepath}")
    except FileNotFoundError:
        logging.error(f"File Error: Sentiment input file '{filepath}' not found.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"File Error: Could not read sentiment CSV file '{filepath}': {e}")
        sys.exit(1)

    required_sentiment_cols = [year_month_col, company_col, product_col]
    missing_cols = [col for col in required_sentiment_cols if col not in df_sentiment.columns]
    if missing_cols:
        logging.error(f"Sentiment Data Error: Sentiment CSV is missing required columns: {', '.join(missing_cols)}.")
        sys.exit(1)

    df_sentiment.rename(columns={
        year_month_col: '_SentimentYearMonth',
        company_col: '_SentimentCompany',
        product_col: '_SentimentProduct'
    }, inplace=True)

    try:
        df_sentiment['_SentimentDate'] = pd.to_datetime(df_sentiment['_SentimentYearMonth'], dayfirst=True,
                                                        errors='coerce')
        if df_sentiment['_SentimentDate'].isnull().sum() > len(df_sentiment) / 2:
            df_sentiment['_SentimentDate'] = pd.to_datetime(df_sentiment['_SentimentYearMonth'], errors='coerce')
        df_sentiment['_SentimentYear'] = df_sentiment['_SentimentDate'].dt.year
    except Exception as e:
        logging.error(
            f"Sentiment Data Error: Could not parse dates in '{year_month_col}'. Ensure format is DD/MM/YYYY or MM/DD/YYYY. Error: {e}")
        sys.exit(1)

    df_sentiment.dropna(subset=['_SentimentYear', '_SentimentCompany', '_SentimentProduct'], inplace=True)
    df_sentiment['_SentimentCompanyLower'] = df_sentiment['_SentimentCompany'].str.strip().str.lower()
    df_sentiment['_SentimentProductLower'] = df_sentiment['_SentimentProduct'].str.strip().str.lower()

    expected_aspect_score_cols = {
        aspect: f"{aspect.replace(' ', '_')}_sentiment_score" for aspect in aspects_list
    }
    found_aspect_cols_map = {}
    actual_cols_to_melt = []

    for aspect_name, col_name_candidate in expected_aspect_score_cols.items():
        if col_name_candidate in df_sentiment.columns:
            found_aspect_cols_map[aspect_name] = col_name_candidate
            actual_cols_to_melt.append(col_name_candidate)
        else:
            col_name_candidate_no_space = f"{aspect_name}_sentiment_score"  # Try matching aspect names like "ValueForMoney_sentiment_score"
            if col_name_candidate_no_space in df_sentiment.columns:
                found_aspect_cols_map[aspect_name] = col_name_candidate_no_space
                actual_cols_to_melt.append(col_name_candidate_no_space)
            else:
                logging.warning(
                    f"Sentiment Data Warning: Score column for aspect '{aspect_name}' (expected like '{col_name_candidate}' or '{col_name_candidate_no_space}') not found in sentiment file.")

    id_vars = ['_SentimentYear', '_SentimentCompanyLower', '_SentimentProductLower']
    df_pivot = pd.DataFrame(columns=id_vars + ['Aspect', 'SentimentScore'])

    if actual_cols_to_melt:
        for col in actual_cols_to_melt:
            df_sentiment[col] = pd.to_numeric(df_sentiment[col], errors='coerce')

        df_melted = df_sentiment.melt(id_vars=id_vars,
                                      value_vars=actual_cols_to_melt,
                                      var_name='AspectColumnName',
                                      value_name='SentimentScore')

        col_to_aspect_map = {v: k for k, v in found_aspect_cols_map.items()}
        df_melted['Aspect'] = df_melted['AspectColumnName'].map(col_to_aspect_map)

        df_melted.dropna(subset=['Aspect', 'SentimentScore'], inplace=True)

        if not df_melted.empty:
            df_pivot = df_melted.groupby(id_vars + ['Aspect'])['SentimentScore'].mean().reset_index()
        else:
            logging.warning("Sentiment Data Warning: After processing, no valid aspect sentiment scores to pivot.")
    else:
        logging.warning(
            "Sentiment Data Warning: No aspect-specific sentiment score columns found or matched in the sentiment file.")

    if not df_pivot.empty:
        df_pivot.set_index(['_SentimentYear', '_SentimentCompanyLower', '_SentimentProductLower', 'Aspect'],
                           inplace=True)
    else:
        df_pivot = pd.DataFrame(columns=['SentimentScore'])
        df_pivot.index = pd.MultiIndex(levels=[[]] * 4, codes=[[]] * 4,
                                       names=['_SentimentYear', '_SentimentCompanyLower', '_SentimentProductLower',
                                              'Aspect'])

    logging.info("Sentiment data loaded and prepared for lookup.")
    return df_pivot


def get_sentiment_score(df_sentiment_pivot, year, company_name, product_name, aspect_name):
    """Looks up an aspect's sentiment score from the prepared sentiment DataFrame."""
    if df_sentiment_pivot.empty:
        return None
    try:
        score = df_sentiment_pivot.loc[
            (year, company_name.strip().lower(), product_name.strip().lower(), aspect_name), 'SentimentScore']
        return score if pd.notna(score) else None
    except KeyError:
        return None
    except Exception as e:
        return None


def format_reviews_for_prompt(review_list, company_label=""):
    """Formats a list of review strings into a single, structured string for an LLM prompt."""
    if not review_list:
        return "No relevant reviews provided for this group."
    if company_label:
        return "\n".join(
            [f"Review {i + 1} ({company_label}): {str(review).strip()}" for i, review in enumerate(review_list)])
    else:
        return "\n".join([f"Review {i + 1}: {str(review).strip()}" for i, review in enumerate(review_list)])


# --- SECTION 5: CORE LLM INTERACTION ---
def construct_market_comparison_prompt(primary_company_name, product_category, focus_type, year,
                                       primary_company_reviews_text, market_aggregate_reviews_text,
                                       aspect_styles_examples):
    """Creates a prompt for market comparison analysis."""
    prompt = f"You are a strategic market analyst for {primary_company_name}.\n"
    prompt += f"Your task is to analyze customer reviews for the product category '{product_category}' in the year {year}.\n"
    prompt += f"Specifically, identify and elaborate on the main '{focus_type}' of {primary_company_name} *in comparison to the aggregated market reviews* from other companies.\n"
    prompt += f"The goal is to understand {primary_company_name}'s competitive positioning regarding '{focus_type}' in the '{product_category}' market for {year}, presenting insights in a detailed, themed HTML format.\n\n"

    prompt += "OUTPUT FORMATTING REQUIREMENTS (HTML-like, single analysis block):\n"
    # ... (rest of prompt same as before)
    prompt += "- Your entire response MUST consist ONLY of the HTML-formatted analysis text. Do not add explanations before or after.\n"
    prompt += f"- Identify 2-4 key themes that illustrate {primary_company_name}'s '{focus_type}' relative to the market aggregate.\n"
    prompt += "- For each theme:\n"
    prompt += "  1. Start with an appropriate emoji and a bolded, underlined title (e.g., '💡 <b><u>Key Strength vs Market</u></b>:', '📉 <b><u>Weakness Compared to Competitors</u></b>:', '🛠️ <b><u>Improvement Opportunity from Market Analysis</u></b>:').\n"
    prompt += f"  2. Explain the theme by comparing {primary_company_name}'s performance (related to '{focus_type}') against trends observed in the Market Aggregate reviews. Use evidence from *both* sets of provided reviews.\n"
    prompt += f"  3. Include short, impactful quotes, clearly attributed (e.g., '<i>\"{primary_company_name} Review: ...\"</i>' or '<i>\"Market Aggregate Review: ...\"</i>').\n"
    prompt += "  4. If a specific point strongly relates to a service aspect (examples below), add the corresponding HTML span tag *at the end* of that point's description.\n"
    for aspect, style_example in aspect_styles_examples.items():
        prompt += f"      - Example style for {aspect}: {style_example}\n"
    if focus_type == "Improvement":
        prompt += f"  5. For 'Improvement' focus: Clearly state an 'Actionable Item:' for {primary_company_name} based on market gaps or competitor strengths revealed in this theme.\n"
    prompt += "- Use '<br><br>' for paragraph breaks between themes to ensure readability.\n"
    prompt += f"- If insufficient data exists for a meaningful comparison regarding '{focus_type}' for '{product_category}' in {year}, your entire response must be ONLY the text: '{NO_INSIGHT_PLACEHOLDER}'\n\n"

    prompt += f"REVIEWS FOR {primary_company_name} (Year: {year}, Product: '{product_category}'):\n"
    prompt += "---------------------------------------------------------------------------\n"
    prompt += f"{primary_company_reviews_text if primary_company_reviews_text else 'No specific reviews provided for ' + primary_company_name}\n"
    prompt += "---------------------------------------------------------------------------\n\n"

    prompt += f"REVIEWS FOR MARKET AGGREGATE (Other Companies) (Year: {year}, Product: '{product_category}'):\n"
    prompt += "------------------------------------------------------------------------------------\n"
    prompt += f"{market_aggregate_reviews_text if market_aggregate_reviews_text else 'No specific reviews provided for Market Aggregate'}\n"
    prompt += "------------------------------------------------------------------------------------\n\n"

    prompt += f"Generate the market comparison analysis focusing on '{focus_type}' for {primary_company_name} regarding '{product_category}' in {year}, following all instructions precisely to produce a single, themed HTML output."
    return prompt


def construct_direct_competitor_comparison_prompt(
        primary_company_name, competitor_name, product_category, aspect_to_compare, year,
        primary_company_reviews_text, competitor_reviews_text, aspect_styles_examples,
        primary_company_aspect_score, competitor_aspect_score, sentiment_difference):
    """Creates a prompt for direct competitor comparison, incorporating sentiment scores."""
    prompt = f"You are a strategic market analyst providing insights for the board of directors of {primary_company_name}.\n"
    prompt += f"Your task is to conduct a direct comparative analysis of {primary_company_name} against its competitor, {competitor_name}, "
    prompt += f"focusing specifically on the product category '{product_category}' and the business aspect '{aspect_to_compare}' for the year {year}.\n"
    prompt += f"The goal is to identify key differences, performance gaps, competitive advantages/disadvantages, and derive a key strategic takeaway for {primary_company_name} based on both the provided customer reviews AND the provided sentiment scores.\n\n"

    prompt += f"SENTIMENT SCORE CONTEXT FOR '{aspect_to_compare}' in {year} for '{product_category}':\n"
    if primary_company_aspect_score is not None:
        prompt += f"- {primary_company_name}'s sentiment score for '{aspect_to_compare}': {primary_company_aspect_score:.1f}\n"
    else:
        prompt += f"- {primary_company_name}'s sentiment score for '{aspect_to_compare}' is not available.\n"
    if competitor_aspect_score is not None:
        prompt += f"- {competitor_name}'s sentiment score for '{aspect_to_compare}': {competitor_aspect_score:.1f}\n"
    else:
        prompt += f"- {competitor_name}'s sentiment score for '{aspect_to_compare}' is not available.\n"

    if sentiment_difference is not None:
        prompt += f"- Sentiment Difference ({primary_company_name} vs {competitor_name}): {sentiment_difference:.1f}. "
        if sentiment_difference > 0:
            prompt += f"({primary_company_name} scores higher).\n"
        elif sentiment_difference < 0:
            prompt += f"({competitor_name} scores higher).\n"
        else:
            prompt += "(Scores are effectively the same).\n"
    else:
        prompt += "- Sentiment difference cannot be directly calculated (one or both scores are unavailable).\n"
    prompt += "Your analysis should integrate these quantitative sentiment differences with qualitative insights derived from the customer reviews provided below. Explicitly reference how the sentiment scores support or contrast with the review narratives.\n\n"

    prompt += "OUTPUT REQUIREMENTS (Single HTML Analysis Block):\n"
    # ... (rest of prompt same as before)
    prompt += "Your entire response MUST consist ONLY of a single HTML-like formatted analysis text block. Do not add any other explanations, introductions, or conclusions.\n"
    prompt += "The analysis block should:\n"
    prompt += f"1. Start with an introductory sentence summarizing the comparison regarding '{aspect_to_compare}' for '{product_category}' between the two companies in {year}.\n"
    prompt += f"2. Discuss key themes, differences in customer perception, and relative performance. Highlight specific strengths or weaknesses of {primary_company_name} *compared to* {competitor_name} concerning '{aspect_to_compare}', considering both the review text AND the provided sentiment scores.\n"
    prompt += "   - Draw evidence *directly* from the provided reviews for both companies.\n"
    prompt += "   - Use relevant emojis and '<b><u>Catchy Sub-theme Title</u></b>:' for distinct comparative points if applicable (e.g., '📞 <b><u>Communication Differences</u></b>:').\n"
    prompt += "   - Incorporate short, impactful quotes from reviews for both companies, clearly attributing them, formatted as '<i>\"{primary_company_name} Review: quote text\"</i>' or '<i>\"{competitor_name} Review: quote text\"</i>'.\n"
    prompt += "   - Use '<br><br>' for paragraph breaks between points.\n"
    prompt += "   - If a point relates to other service aspects (from the examples below), you can optionally use the span tags for emphasis, but the primary focus must remain '{aspect_to_compare}'.\n"
    for aspect, style_example in aspect_styles_examples.items():
        if aspect_to_compare in style_example or aspect_to_compare == aspect:
            prompt += f"       - Style for the current aspect '{aspect}': {style_example}\n"
        else:
            prompt += f"       - Example style for other aspect {aspect}: {style_example}\n"

    prompt += f"3. Conclude with a bolded '💡<b>Key Takeaway for {primary_company_name}:</b>' followed by 1-2 concise, commercially useful, and actionable recommendations or strategic considerations for {primary_company_name} based on the comprehensive comparison (reviews and sentiment scores).\n\n"
    prompt += f"If, after careful analysis, you find insufficient review data for a meaningful comparison on '{aspect_to_compare}' for '{product_category}' between {primary_company_name} and {competitor_name} for {year} (even considering sentiment scores), your entire response must be ONLY the text: '{NO_INSIGHT_PLACEHOLDER}'\n\n"

    prompt += f"CUSTOMER REVIEWS FOR {primary_company_name} (Year: {year}, Product: '{product_category}', Aspect Focus: '{aspect_to_compare}'):\n"
    prompt += "------------------------------------------------------------------------------------------------------\n"
    prompt += f"{primary_company_reviews_text if primary_company_reviews_text else 'No specific reviews provided for ' + primary_company_name}\n"
    prompt += "------------------------------------------------------------------------------------------------------\n\n"

    prompt += f"CUSTOMER REVIEWS FOR {competitor_name} (Year: {year}, Product: '{product_category}', Aspect Focus: '{aspect_to_compare}'):\n"
    prompt += "---------------------------------------------------------------------------------------------------\n"
    prompt += f"{competitor_reviews_text if competitor_reviews_text else 'No specific reviews provided for ' + competitor_name}\n"
    prompt += "---------------------------------------------------------------------------------------------------\n\n"

    prompt += f"Please generate the single HTML analysis block for {primary_company_name} vs {competitor_name} regarding '{product_category}' and '{aspect_to_compare}' for {year}, including the Key Takeaway, following all instructions precisely."
    return prompt


def call_llm_with_retry(client, model, system_prompt, user_content, max_retries=3, delay_seconds=REQUEST_DELAY_SECONDS):
    """Sends prompt to LLM with retry logic."""
    for attempt in range(max_retries):
        try:
            logging.info(f"Initiating LLM call (Attempt {attempt + 1}/{max_retries}) for model '{model}'...")
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.2,
            )
            analysis_text = completion.choices[0].message.content.strip()
            logging.info(f"LLM call successful (Attempt {attempt + 1}).")
            time.sleep(delay_seconds)
            return analysis_text
        except OpenAIError as e:
            logging.warning(f"OpenAI API error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                wait_time = delay_seconds * (2 ** attempt)
                logging.info(f"Retrying LLM call in {wait_time:.1f} seconds...")
                time.sleep(wait_time)
            else:
                logging.error(f"LLM call failed after {max_retries} attempts due to API errors.")
                return f"Error: LLM analysis failed after {max_retries} attempts. Last API error: {type(e).__name__} - {e}"
        except Exception as e:
            logging.error(f"An unexpected error occurred during LLM call on attempt {attempt + 1}: {e}", exc_info=True)
            if attempt < max_retries - 1:
                wait_time = delay_seconds * (2 ** attempt)
                logging.info(f"Retrying LLM call in {wait_time:.1f} seconds...")
                time.sleep(wait_time)
            else:
                return f"Error: An unexpected error occurred during LLM interaction after {max_retries} attempts: {e}"
    return f"Error: LLM call failed after {max_retries} attempts. Unknown reason."


# --- SECTION 6: MAIN EXECUTION LOGIC ---
def main():
    """Orchestrates the entire review synthesis process with incremental saving."""
    script_start_time = time.time()
    logging.info("Starting the review synthesis process...")

    validate_config()
    api_key_value = get_api_key_from_env_or_config()
    openai_client = initialize_openai_client(api_key_value)

    df_input_reviews = load_and_validate_input_df(INPUT_FILE, COL_DATE, COL_REVIEW, COL_PRODUCT, COL_REVIEW_COMPANY)
    primary_company_lower = PRIMARY_COMPANY_NAME.strip().lower()

    df_sentiment_pivot = None
    if GENERATE_DIRECT_COMPETITOR_INSIGHTS and DIRECT_COMPETITOR_ASPECTS and SENTIMENT_INPUT_FILE:
        df_sentiment_pivot = load_and_prepare_sentiment_data(
            SENTIMENT_INPUT_FILE,
            COL_SENTIMENT_YEAR_MONTH,
            COL_SENTIMENT_COMPANY,
            COL_SENTIMENT_PRODUCT,
            DIRECT_COMPETITOR_ASPECTS
        )

    base_name = os.path.splitext(os.path.basename(INPUT_FILE))[0]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_directory = os.path.dirname(INPUT_FILE) if os.path.dirname(INPUT_FILE) else '.'

    market_compare_output_file_path = os.path.join(output_directory,
                                                   f"{base_name}_{PRIMARY_COMPANY_NAME.replace(' ', '')}_MarketComparison_{timestamp}.csv")
    direct_competitor_output_file_path = None
    if GENERATE_DIRECT_COMPETITOR_INSIGHTS and DIRECT_COMPETITOR_ASPECTS:
        direct_competitor_output_file_path = os.path.join(output_directory,
                                                          f"{base_name}_{PRIMARY_COMPANY_NAME.replace(' ', '')}_DirectCompetitorComparison_{timestamp}.csv")

    # --- Market Comparison Analysis (Incremental Saving) ---
    if not MARKET_COMPARE_FOCUS_TYPES:
        logging.warning("MARKET_COMPARE_FOCUS_TYPES list is empty. Skipping market comparison analysis.")
    else:
        logging.info(f"Starting Market Comparison Analysis for {PRIMARY_COMPANY_NAME}...")
        logging.info(
            f"Market comparison analysis for {PRIMARY_COMPANY_NAME} will be saved incrementally to: {market_compare_output_file_path}")

        market_output_columns = ['Year', 'Product', 'Aspect', 'Analysis']
        market_file_initialized = False
        try:
            pd.DataFrame(columns=market_output_columns).to_csv(market_compare_output_file_path, index=False,
                                                               encoding='utf-8', mode='w')
            market_file_initialized = True
            logging.info(f"Initialized market comparison output file with headers: {market_compare_output_file_path}")
        except Exception as e:
            logging.error(
                f"File Error: Failed to initialize market comparison output file '{market_compare_output_file_path}': {e}")

        if market_file_initialized:
            grouped_by_year_product_reviews = df_input_reviews.groupby(['Year', 'ProductLower'])
            total_market_compare_iterations = len(grouped_by_year_product_reviews) * len(MARKET_COMPARE_FOCUS_TYPES)
            data_generated_for_market_compare = False

            if total_market_compare_iterations == 0:
                logging.info("No data groups found for market comparison.")
            else:
                with tqdm(total=total_market_compare_iterations, desc=f"{PRIMARY_COMPANY_NAME} Market Comparison",
                          unit="summary") as pbar_market:
                    for (year, product_lower_name), group_df_all_companies in grouped_by_year_product_reviews:
                        product_name_original_case = group_df_all_companies['Product'].iloc[0]
                        df_primary_market = group_df_all_companies[
                            group_df_all_companies['ReviewedCompanyLower'] == primary_company_lower].copy()
                        df_market_agg = group_df_all_companies[
                            group_df_all_companies['ReviewedCompanyLower'] != primary_company_lower].copy()

                        df_primary_market['Date'] = pd.to_datetime(df_primary_market['Date'])
                        primary_reviews_df = df_primary_market.sort_values(by='Date', ascending=False).head(
                            MAX_REVIEWS_FOR_SYNTHESIS // 2 + MAX_REVIEWS_FOR_SYNTHESIS % 2)
                        primary_reviews_list = primary_reviews_df['Review'].tolist()
                        primary_reviews_text = format_reviews_for_prompt(primary_reviews_list,
                                                                         company_label=PRIMARY_COMPANY_NAME)

                        df_market_agg['Date'] = pd.to_datetime(df_market_agg['Date'])
                        market_agg_reviews_df = df_market_agg.sort_values(by='Date', ascending=False).head(
                            MAX_REVIEWS_FOR_SYNTHESIS // 2)
                        market_agg_reviews_list = market_agg_reviews_df['Review'].tolist()
                        market_agg_reviews_text = format_reviews_for_prompt(market_agg_reviews_list,
                                                                            company_label="Market Aggregate")

                        if not primary_reviews_list and not market_agg_reviews_list:
                            logging.debug(
                                f"Skipping Market Comparison for Product: '{product_name_original_case}', Year: {year} - No reviews found for either group.")
                            pbar_market.update(len(MARKET_COMPARE_FOCUS_TYPES))
                            continue

                        for focus_type in MARKET_COMPARE_FOCUS_TYPES:
                            pbar_market.set_description(
                                f"Market Comp: {product_name_original_case[:15]} ({focus_type}) Y{year}")
                            system_prompt = construct_market_comparison_prompt(
                                PRIMARY_COMPANY_NAME, product_name_original_case, focus_type, year,
                                primary_reviews_text, market_agg_reviews_text, ASPECT_STYLES_EXAMPLES_FOR_PROMPT
                            )
                            user_content_for_llm = f"Generate the market comparison analysis for {PRIMARY_COMPANY_NAME} on product '{product_name_original_case}' ({focus_type}) in {year}."
                            generated_analysis = call_llm_with_retry(openai_client, MODEL, system_prompt,
                                                                     user_content_for_llm)

                            current_market_data = {
                                'Year': year, 'Product': product_name_original_case,
                                'Aspect': focus_type, 'Analysis': generated_analysis
                            }
                            try:
                                df_single_market_row = pd.DataFrame([current_market_data],
                                                                    columns=market_output_columns)
                                df_single_market_row.to_csv(market_compare_output_file_path, mode='a', header=False,
                                                            index=False, encoding='utf-8')
                                data_generated_for_market_compare = True
                            except Exception as e:
                                logging.error(
                                    f"File Error: Failed to append to market comparison output file '{market_compare_output_file_path}': {e}")
                            pbar_market.update(1)

                if data_generated_for_market_compare:
                    logging.info(
                        f"Finished incrementally saving market comparison analysis to '{market_compare_output_file_path}'")
                elif MARKET_COMPARE_FOCUS_TYPES:
                    logging.warning(
                        f"Market comparison analysis ran, but no data rows were generated to save to '{market_compare_output_file_path}'. The file may only contain headers if it was initialized.")
            if not data_generated_for_market_compare and market_file_initialized:
                logging.info(
                    f"Market comparison output file '{market_compare_output_file_path}' was initialized but no data was added.")

    # --- Direct Competitor Comparison Analysis (Incremental Saving) ---
    if GENERATE_DIRECT_COMPETITOR_INSIGHTS and DIRECT_COMPETITOR_ASPECTS:
        logging.info(f"Starting Direct Competitor Comparison Analysis for {PRIMARY_COMPANY_NAME}...")
        all_companies_original_case = df_input_reviews['ReviewedCompany'].str.strip().unique()
        competitors_original_case = [comp for comp in all_companies_original_case if
                                     comp.strip().lower() != primary_company_lower]

        if not direct_competitor_output_file_path:
            logging.error("Internal Error: direct_competitor_output_file_path not set for competitor analysis.")
        else:
            logging.info(
                f"Direct competitor comparison insights will be saved incrementally to: {direct_competitor_output_file_path}")
            competitor_output_columns = ['Company', 'Product', 'Aspect', 'Sentiment Score', 'Sentiment Difference',
                                         'Year', 'Analysis']
            competitor_file_initialized = False
            try:
                pd.DataFrame(columns=competitor_output_columns).to_csv(direct_competitor_output_file_path, index=False,
                                                                       encoding='utf-8', mode='w')
                competitor_file_initialized = True
                logging.info(
                    f"Initialized direct competitor output file with headers: {direct_competitor_output_file_path}")
            except Exception as e:
                logging.error(
                    f"File Error: Failed to initialize direct competitor output file '{direct_competitor_output_file_path}': {e}")

            if competitor_file_initialized:
                iterations_to_run = []
                for year_val in sorted(df_input_reviews['Year'].unique(), reverse=True):
                    df_year_reviews = df_input_reviews[df_input_reviews['Year'] == year_val]
                    df_primary_year_reviews = df_year_reviews[
                        df_year_reviews['ReviewedCompanyLower'] == primary_company_lower]
                    if df_primary_year_reviews.empty: continue
                    primary_products_lower_set = set(df_primary_year_reviews['ProductLower'].unique())
                    for competitor_name_original_case in competitors_original_case:
                        competitor_name_lower = competitor_name_original_case.strip().lower()
                        df_competitor_year_reviews = df_year_reviews[
                            df_year_reviews['ReviewedCompanyLower'] == competitor_name_lower]
                        if df_competitor_year_reviews.empty: continue
                        competitor_products_lower_set = set(df_competitor_year_reviews['ProductLower'].unique())
                        common_products_lower = list(
                            primary_products_lower_set.intersection(competitor_products_lower_set))
                        if common_products_lower:
                            for product_lower_name in common_products_lower:
                                product_name_original_case = \
                                df_primary_year_reviews[df_primary_year_reviews['ProductLower'] == product_lower_name][
                                    'Product'].iloc[0]
                                for aspect_name in DIRECT_COMPETITOR_ASPECTS:
                                    iterations_to_run.append((year_val, competitor_name_original_case,
                                                              product_name_original_case, product_lower_name,
                                                              aspect_name))

                data_generated_for_competitor_compare = False
                if not competitors_original_case:
                    logging.warning("No competitor company data found in review input file.")
                elif not iterations_to_run:
                    logging.warning(
                        "No common products found between primary company and any competitors for specified aspects for direct comparison.")
                else:
                    logging.info(
                        f"Preparing to generate {len(iterations_to_run)} direct competitor comparison summaries.")
                    with tqdm(total=len(iterations_to_run), desc="Direct Competitor Insights",
                              unit="comparison") as pbar_direct_comp:
                        for year_val, competitor_name_original_case, product_name_original_case, product_lower_name_for_lookup, aspect_name in iterations_to_run:
                            pbar_direct_comp.set_description(
                                f"Direct Comp: {competitor_name_original_case[:10]} ({product_name_original_case[:10]}-{aspect_name[:10]}) Y{year_val}")
                            current_reviews_df = df_input_reviews[
                                (df_input_reviews['Year'] == year_val) &
                                (df_input_reviews['ProductLower'] == product_lower_name_for_lookup)
                                ]
                            df_primary_iter_reviews = current_reviews_df[
                                current_reviews_df['ReviewedCompanyLower'] == primary_company_lower].copy()
                            df_primary_iter_reviews['Date'] = pd.to_datetime(df_primary_iter_reviews['Date'])
                            primary_reviews_df = df_primary_iter_reviews.sort_values(by='Date', ascending=False).head(
                                MAX_REVIEWS_FOR_SYNTHESIS // 2 + MAX_REVIEWS_FOR_SYNTHESIS % 2)
                            primary_reviews_list = primary_reviews_df['Review'].tolist()
                            primary_reviews_text = format_reviews_for_prompt(primary_reviews_list,
                                                                             company_label=PRIMARY_COMPANY_NAME)

                            df_competitor_iter_reviews = current_reviews_df[current_reviews_df[
                                                                                'ReviewedCompanyLower'] == competitor_name_original_case.strip().lower()].copy()
                            df_competitor_iter_reviews['Date'] = pd.to_datetime(df_competitor_iter_reviews['Date'])
                            competitor_reviews_df = df_competitor_iter_reviews.sort_values(by='Date',
                                                                                           ascending=False).head(
                                MAX_REVIEWS_FOR_SYNTHESIS // 2)
                            competitor_reviews_list = competitor_reviews_df['Review'].tolist()
                            competitor_reviews_text = format_reviews_for_prompt(competitor_reviews_list,
                                                                                company_label=competitor_name_original_case)

                            primary_score, competitor_score, sentiment_diff = None, None, None
                            if df_sentiment_pivot is not None:
                                primary_score = get_sentiment_score(df_sentiment_pivot, year_val, PRIMARY_COMPANY_NAME,
                                                                    product_lower_name_for_lookup, aspect_name)
                                competitor_score = get_sentiment_score(df_sentiment_pivot, year_val,
                                                                       competitor_name_original_case,
                                                                       product_lower_name_for_lookup, aspect_name)
                                if primary_score is not None and competitor_score is not None:
                                    sentiment_diff = primary_score - competitor_score

                            generated_analysis_for_competitor_file = NO_INSIGHT_PLACEHOLDER
                            if not primary_reviews_list and not competitor_reviews_list:
                                logging.debug(
                                    f"Skipping direct comparison: {PRIMARY_COMPANY_NAME} vs {competitor_name_original_case} for {product_name_original_case}/{aspect_name} in {year_val} - No reviews for EITHER.")
                            else:
                                system_prompt = construct_direct_competitor_comparison_prompt(
                                    PRIMARY_COMPANY_NAME, competitor_name_original_case, product_name_original_case,
                                    aspect_name, year_val,
                                    primary_reviews_text, competitor_reviews_text, ASPECT_STYLES_EXAMPLES_FOR_PROMPT,
                                    primary_score, competitor_score, sentiment_diff
                                )
                                user_content_for_llm = f"Generate direct comparison: {PRIMARY_COMPANY_NAME} vs {competitor_name_original_case} for {product_name_original_case}, aspect '{aspect_name}', year {year_val}."
                                generated_analysis_for_competitor_file = call_llm_with_retry(openai_client, MODEL,
                                                                                             system_prompt,
                                                                                             user_content_for_llm)

                            current_competitor_data = {
                                'Company': competitor_name_original_case, 'Product': product_name_original_case,
                                'Aspect': aspect_name,
                                'Sentiment Score': f"{primary_score:.1f}" if primary_score is not None else "N/A",
                                'Sentiment Difference': f"{sentiment_diff:.1f}" if sentiment_diff is not None else "N/A",
                                'Year': year_val, 'Analysis': generated_analysis_for_competitor_file
                            }
                            try:
                                df_single_competitor_row = pd.DataFrame([current_competitor_data],
                                                                        columns=competitor_output_columns)
                                df_single_competitor_row.to_csv(direct_competitor_output_file_path, mode='a',
                                                                header=False, index=False, encoding='utf-8')
                                data_generated_for_competitor_compare = True
                            except Exception as e:
                                logging.error(
                                    f"File Error: Failed to append to direct competitor output file '{direct_competitor_output_file_path}': {e}")
                            pbar_direct_comp.update(1)

                    if data_generated_for_competitor_compare:
                        logging.info(
                            f"Finished incrementally saving direct competitor comparison insights to '{direct_competitor_output_file_path}'")
                    elif GENERATE_DIRECT_COMPETITOR_INSIGHTS and DIRECT_COMPETITOR_ASPECTS:
                        logging.warning(
                            f"Direct competitor comparison analysis ran, but no data rows were generated to save to '{direct_competitor_output_file_path}'. The file may only contain headers if it was initialized.")
                if not data_generated_for_competitor_compare and competitor_file_initialized:
                    logging.info(
                        f"Direct competitor output file '{direct_competitor_output_file_path}' was initialized but no data was added.")

    elif GENERATE_DIRECT_COMPETITOR_INSIGHTS:
        logging.warning(
            "GENERATE_DIRECT_COMPETITOR_INSIGHTS is True, but DIRECT_COMPETITOR_ASPECTS list is empty. Skipping direct competitor comparison.")
    else:
        logging.info("GENERATE_DIRECT_COMPETITOR_INSIGHTS is False. Skipping direct competitor comparison.")

    # --- Final Cleanup & Completion ---
    # The old saving blocks are removed as saving is now incremental.
    # Warnings about files "not being created" are handled by new logic above,
    # as files with headers will now always be created if the process starts.

    script_end_time = time.time()
    total_script_duration = script_end_time - script_start_time
    logging.info(f"Review synthesis process completed in {total_script_duration:.2f} seconds.")


# --- SCRIPT EXECUTION ENTRY POINT ---
if __name__ == "__main__":
    main()