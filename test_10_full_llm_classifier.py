"""
review_analysis.py

A simple, easy-to-use script for non-expert users to:
  • Classify reviews into product categories per company definitions
  • Analyze aspect-based sentiment of reviews (services or energy)
  • Perform both tasks together, with test mode support

HOW TO USE:
  1. At the top of this file, edit the CONFIGURATION section:
     - COMPANY: choose your company, which determines product categories
     - MODE: 'classification', 'sentiment', or 'both'
     - ASPECT_MODE: 'services' or 'energy' (used if MODE includes sentiment)
     - TEST_MODE: True to run only SAMPLE_SIZE reviews, False to run all
     - INPUT_FILE: path to input CSV (must have 'Date' and 'Review')
     - API_KEY: your OpenAI key (or set OPENAI_API_KEY in env)
  2. Install dependencies:
       pip install openai pandas python-dotenv tqdm

FLOW:
  • Validate config and input (friendly error messages)
  • Read input CSV, parse dates, apply test mode limit
  • Auto-generate a timestamped output filename
  • Initialize the output CSV headers
  • For each review (with tqdm progress bar):
      – Classification (if enabled)
      – Sentiment (multi-aspect prompt) (if enabled)
      – Append row to output file
  • After processing, print a summary report:
      – Counts per product category
      – Average sentiment per aspect

"""

# ---------------- CONFIGURATION (EDIT ME) ----------------
LLM_PROVIDER = 'openai'  # Choose 'openai', 'google', or 'anthropic'
COMPANY = 'British Gas Services'  # choose from COMPANIES keys
MODE = 'classification'            # 'classification', 'sentiment', or 'both'
ASPECT_MODE = 'energy'  # 'services' or 'energy'
TEST_MODE = False        # True = run only SAMPLE_SIZE reviews
SAMPLE_SIZE = 50         # number of reviews when TEST_MODE=True
INPUT_FILE = 'BG All TrustPilot Data Pt2.csv' #'BG All TrustPilot Data.csv' #'BG 2024 TrustPilot Data.csv'
API_KEY = ''             # leave blank to use OPENAI_API_KEY in env
BATCH_SIZE = 5
MODEL = "gpt-4.1-mini"
USE_BATCHING = False
API_KEY_OPENAI = ''
API_KEY_GOOGLE = ''
API_KEY_ANTHROPIC = ''

# Model mapping for each provider
MODELS = {
    'openai': 'gpt-4.1-mini',
    'google': 'gemini-2.5-flash-preview-05-20', # Or 'gemini-2.5-pro-preview-06-05'
    'anthropic': 'claude-3-sonnet-20240229' # Or 'claude-3-haiku-20240307' for speed
}

# ---------------------------------------------------------

import os
import sys
import pandas as pd
import time
from datetime import datetime, timedelta
from openai import OpenAI, OpenAIError
import google.generativeai as genai
from anthropic import Anthropic, AnthropicError
from dotenv import load_dotenv
from tqdm import tqdm
import re

# Pre-define products and companies
PRODUCT_DEFINITIONS = {
    #'Electricity Supply': 'Covers the supply of electricity to homes or businesses, including fixed or variable tariffs, smart meter usage, renewable energy options, or issues related to power supply or billing for electricity.',
    #'Gas Supply': 'Covers the supply of natural gas for heating, cooking, or hot water, including fixed or variable tariffs, smart meter usage, renewable gas options, or issues related to gas supply or billing for gas.',
    'Gas Products': 'Covers boiler insurance, home care, annual service visits, or any instances where engineers visit customers homes to service, install, or fix boilers, central heating, or fix the hot water not running.',
    'Energy': 'Relates to British Gas as an energy / electricity supplier, or gas supply services, including tariffs, smart meters, and energy bills including charges and billing issues for unfixed tariffs. This category does not ever involve engineers visiting homes.',
    'Plumbing and Drains': 'Insurance for issues such as blocked drains, frozen pipes, or plumbing repairs, often handled by DynoRod or similar partners.',
    'Appliance Cover': 'Includes insurance for home appliances like ovens, washing machines, or any electrical appliances that we repair if they break down',
    'Home Electrical': 'Insurance for home electrics, including cover for wiring, fusebox breakdowns and broken sockets.',
    "Building": "General building services.",
    "Pest Control": "Removal of a pest infestation in the home, eg. Wasps and hornets nests, mice or rat infestation.",
    'Unknown': 'Please use this only if there is absolutely no information to categorize the review, after making every effort to find relevant clues. If you have to infer anything from non-conclusive evidence such as the company name or the sentiment of the review, it should be classificed as unknown.',
}
COMPANIES = {
    'British Gas Services': ['Gas Products', 'Energy', 'Plumbing and Drains', 'Appliance Cover', 'Home Electrical'],
    'British Gas Energy': ['Electricity Supply', 'Gas Supply'],
    "HomeServe": ["Gas Products", "Plumbing and Drains", "Appliance Cover", "Home Electrical"],
    "CheckATrade": ["Gas Products", "Plumbing and Drains", "Home Electrical", "Building"],
    "Corgi HomePlan": ["Gas Products", "Plumbing and Drains", "Home Electrical", "Building"],
    "Domestic & General": ["Gas Products", "Plumbing and Drains", "Appliance Cover", "Home Electrical"],
    "247 Home Rescue": ["Gas Products", "Plumbing and Drains", "Appliance Cover", "Home Electrical", "Pest Control"],
    'Octopus': ['Electricity Supply', 'Gas Supply'],
}
for cats in COMPANIES.values():
    cats.append("Unknown")


# Aspect sets
ASPECT_SETS = {
    'services': ["Overall Sentiment", "Appointment Scheduling", "Customer Service", "Response Speed", "Engineer Experience", "Solution Quality", "Value for Money"],
    'energy': ["Overall Sentiment", "Appointment Scheduling", "Customer Service", "Response Speed", "Meter Readings", "Value for Money"]
}

# Validate configuration
def validate_config():
    if COMPANY not in COMPANIES:
        print(f"Error: COMPANY must be one of {list(COMPANIES.keys())}")
        sys.exit(1)
    if MODE not in ('classification', 'sentiment', 'both'):
        print("Error: MODE must be 'classification', 'sentiment', or 'both'.")
        sys.exit(1)
    if MODE in ('sentiment', 'both') and ASPECT_MODE not in ASPECT_SETS:
        print(f"Error: ASPECT_MODE must be one of {list(ASPECT_SETS.keys())}")
        sys.exit(1)
    if not os.path.isfile(INPUT_FILE):
        print(f"Error: INPUT_FILE '{INPUT_FILE}' not found.")
        sys.exit(1)
    if LLM_PROVIDER not in ('openai', 'google', 'anthropic'):
            print(f"Error: LLM_PROVIDER must be 'openai', 'google', or 'anthropic'.")
            sys.exit(1)
    valid_models = ('gpt-4o-mini', 'gpt-3.5-turbo', 'gpt-4o', 'gpt-4.1-mini', 'gpt-4.1-nano', 'gemini-2.5-flash-preview-05-20', 'gemini-2.5-pro-preview-06-05')
    if MODEL not in valid_models:
        print(f"Error: MODEL must be one of {valid_models}")
        sys.exit(1)
    if not isinstance(BATCH_SIZE, int) or BATCH_SIZE < 1:
        print("Error: BATCH_SIZE must be a positive integer")
        sys.exit(1)
    if not isinstance(USE_BATCHING, bool):
        print("Error: USE_BATCHING must be True or False")
        sys.exit(1)

def get_api_key(provider):
    load_dotenv()
    key_map = {
        'openai': API_KEY_OPENAI or os.getenv('OPENAI_API_KEY'),
        'google': API_KEY_GOOGLE or os.getenv('GOOGLE_API_KEY'),
        'anthropic': API_KEY_ANTHROPIC or os.getenv('ANTHROPIC_API_KEY'),
    }
    key = key_map.get(provider)
    if not key:
        env_var = f"{provider.upper()}_API_KEY"
        print(f"Error: {provider.capitalize()} API key not set. Please edit the config or set {env_var}.")
        sys.exit(1)
    return key


def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))


def validate_input_df(df):
    if 'Review' not in df.columns or 'Date' not in df.columns:
        print("Error: Input CSV must contain 'Date' and 'Review' columns.")
        sys.exit(1)
    if df['Review'].isna().any() or df['Review'].str.strip().eq('').any():
        print("Warning: Skipping rows with empty or null reviews.")
        df = df[df['Review'].notna() & df['Review'].str.strip().ne('')]
    try:
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        if df['Date'].isna().any():
            print("Warning: Some dates are invalid and will be skipped.")
            df = df[df['Date'].notna()]
    except Exception as e:
        print(f"Error: Invalid date format in CSV: {e}")
        sys.exit(1)
    return df


def initialize_output_file(path, aspects, do_classify):
    cols = ['Date', 'Year-Month', 'Review']
    if do_classify:
        cols += ['prod_score', 'prod_category', 'prod_reason']
    if aspects:
        for asp in aspects:
            cols += [f"{asp}_score", f"{asp}_reason"]
    pd.DataFrame(columns=cols).to_csv(path, index=False)


def append_row(path, row):
    pd.DataFrame([row]).to_csv(path, mode='a', header=False, index=False)


def validate_classification_result(score, category, reason, valid_categories):
    try:
        score = int(score)
        if not (0 <= score <= 100):
            raise ValueError("Score out of range")
        if category not in valid_categories:
            raise ValueError(f"Invalid category: {category}")
        if not reason.strip():
            raise ValueError("Empty reason")
        return score, category, reason
    except (ValueError, TypeError) as e:
        print(f"Warning: Invalid classification result: {e}")
        return 0, 'Unknown', 'Invalid response'


def validate_sentiment_result(score, reason, aspect):
    try:
        # Handle 'N/A' or non-numeric scores
        if isinstance(score, str) and score.strip().lower() in ('n/a', 'na', 'none'):
            print(f"Warning: Non-numeric score 'N/A' for aspect {aspect}, defaulting to 0")
            score = 0
        else:
            score = int(score)
        if not (-100 <= score <= 100):
            raise ValueError("Score out of range")
        if not reason.strip():
            raise ValueError("Empty reason")
        return score, reason
    except (ValueError, TypeError) as e:
        print(f"Warning: Invalid sentiment result for {aspect}: {e}")
        return 0, 'Invalid response'

def generate_completion(client, provider, model, system_msg, user_msg, max_tokens=150, temperature=0.1):
    """
    Calls the appropriate LLM provider and returns the response text.
    Handles different API formats and exceptions.
    """
    try:
        if provider == 'openai':
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {'role': 'system', 'content': system_msg},
                    {'role': 'user', 'content': user_msg}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()

        elif provider == 'google':
            # Gemini combines system and user prompts
            full_prompt = f"{system_msg}\n\n{user_msg}"
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature
            )
            response = client.generate_content(full_prompt, generation_config=generation_config)
            return response.text.strip()

        elif provider == 'anthropic':
            response = client.messages.create(
                model=model,
                system=system_msg,
                messages=[
                    {'role': 'user', 'content': user_msg}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.content[0].text.strip()

    except (OpenAIError, AnthropicError, Exception) as e:
        # Catch exceptions from all providers
        print(f"Warning: API call failed for {provider} with error: {e}")
        return "" # Return empty string on failure

def classify_product(review, client, categories):

    if COMPANY == "British Gas Services":
        # Create a detailed list of category definitions for the prompt
        category_definitions_text = ""
        for cat in categories:
            if cat in PRODUCT_DEFINITIONS:
                category_definitions_text += f"- **{cat}**: {PRODUCT_DEFINITIONS[cat]}\n"

        system_msg = (
            "You are a product analysis expert for a company that provides home services and energy. "
            "Your task is to categorize a customer review based on the detailed definitions provided below. Pay close attention to the specific details that distinguish one category from another, such as the mention of an engineer visit for a repair versus a billing issue.\n\n"
            "**Available Categories and Their Definitions:**\n"
            f"{category_definitions_text}\n"
            "Carefully analyze the review and select the single best category. The 'Energy' category is for supply and billing and **does not** involve an engineer visiting to fix something.\n\n"
            "Reply exactly in the format:\nScore: <0-100 confidence score>\nProduct: <category name>\nReason: <brief explanation>"
        )
    else:
        system_msg = (
                "You are a product analysis expert. Categorize this review into one of the following categories:\n" +
                '\n'.join(f"{i + 1}. {cat}" for i, cat in enumerate(categories)) +
                "\nReply exactly in format:\nScore: <0-100>\nProduct: <category>\nReason: <brief explanation>"
        )
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {'role': 'system', 'content': system_msg},
                {'role': 'user', 'content': f"Review: {review}"}
            ], temperature=0.0, max_tokens=150
        )
        raw = response.choices[0].message.content.strip()
        score_match = re.search(r'Score:\s*(\d+)', raw)
        product_match = re.search(r'Product:\s*([^\n]+)', raw)
        reason_match = re.search(r'Reason:\s*([^\n]+)', raw)
        if not (score_match and product_match and reason_match):
            raise ValueError("Incomplete response")
        score, category, reason = validate_classification_result(
            score_match.group(1),
            product_match.group(1).strip(),
            reason_match.group(1).strip(),
            categories
        )
        return score, category, reason
    except (OpenAIError, ValueError, AttributeError) as e:
        print(f"Warning: Error classifying review: {e}")
        return 0, 'Unknown', 'Processing error'


def classify_product_batch(reviews, client, categories):
    if not reviews:
        return [None] * len(reviews)

    system_msg = (
            "You are a product analysis expert. Categorize each review below into one of the following categories:\n" +
            '\n'.join(f"{i + 1}. {cat}" for i, cat in enumerate(categories)) +
            "\nFor each review, reply in the format:\n[Review <index>]\nScore: <0-100>\nProduct: <category>\nReason: <brief explanation>\n"
    )
    batch_prompt = '\n'.join(f"Review {i + 1}: {review}" for i, review in enumerate(reviews))

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {'role': 'system', 'content': system_msg},
                {'role': 'user', 'content': batch_prompt}
            ], temperature=0.2, max_tokens=150 * len(reviews)
        )
        raw = response.choices[0].message.content.strip()

        # Parse response
        results = [None] * len(reviews)
        blocks = re.split(r'\[Review (\d+)\]', raw)[1:]  # Split by [Review X]
        for i in range(0, len(blocks), 2):
            index = int(blocks[i]) - 1  # Review index (1-based to 0-based)
            block = blocks[i + 1].strip()
            try:
                score_match = re.search(r'Score:\s*(\d+)', block)
                product_match = re.search(r'Product:\s*([^\n]+)', block)
                reason_match = re.search(r'Reason:\s*([^\n]+)', block)
                if not (score_match and product_match and reason_match):
                    raise ValueError("Incomplete response")
                score, category, reason = validate_classification_result(
                    score_match.group(1),
                    product_match.group(1).strip(),
                    reason_match.group(1).strip(),
                    categories
                )
                results[index] = (score, category, reason)
            except (ValueError, AttributeError) as e:
                print(f"Warning: Failed to parse classification for Review {index + 1}: {e}")
                results[index] = (0, 'Unknown', 'Parsing error')

        # Fill missing results
        for i in range(len(reviews)):
            if results[i] is None:
                print(f"Warning: No classification result for Review {i + 1}")
                results[i] = (0, 'Unknown', 'Missing response')

        return results

    except OpenAIError as e:
        print(f"Warning: API error during classification: {e}")
        return [(0, 'Unknown', 'API failure')] * len(reviews)


def analyze_sentiments(review, aspects, client):
    prompt = (
            "You are a sentiment analysis expert. For each aspect below, provide a score (-100 to 100) and a brief reason.\n" +
            "Aspects:\n" + '\n'.join(f"- {asp}" for asp in aspects) +
            "\nIf an aspect cannot be evaluated due to missing information, use a score of 0.\n" +
            "Reply exactly in the format for each aspect:\nAspect: <name>\nScore: <value>\nReason: <text>\n"
    )
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {'role': 'system', 'content': prompt},
                {'role': 'user', 'content': f"Review: {review}"}
            ], temperature=0.2, max_tokens=400
        )
        raw = response.choices[0].message.content.strip()
        results = {}
        blocks = raw.split('Aspect: ')[1:]  # Skip leading
        for block in blocks:
            lines = block.splitlines()
            asp = lines[0].strip()
            score_match = re.search(r'Score:\s*([-]?\d+|N/A)', lines[1], re.IGNORECASE)
            reason_match = re.search(r'Reason:\s*([^\n]+)', lines[2])
            if not (score_match and reason_match):
                print(f"Warning: Incomplete sentiment response for aspect {asp}")
                results[asp] = (0, 'Incomplete response')
                continue
            score = score_match.group(1)
            reason = reason_match.group(1).strip()
            score, reason = validate_sentiment_result(score, reason, asp)
            results[asp] = (score, reason)
        # Ensure only requested aspects are included
        valid_results = {asp: results.get(asp, (0, 'Missing response')) for asp in aspects}
        return valid_results
    except (OpenAIError, ValueError, AttributeError, IndexError) as e:
        print(f"Warning: Error analyzing sentiments for review: {e}")
        return {asp: (0, 'Processing error') for asp in aspects}


def analyze_sentiments_batch(reviews, aspects, client):
    if not reviews:
        return [None] * len(reviews)

    prompt = (
            "You are a sentiment analysis expert. For each review below, analyze the specified aspects.\n" +
            "Aspects:\n" + '\n'.join(f"- {asp}" for asp in aspects) +
            "\nIf an aspect cannot be evaluated due to missing information, use a score of 0.\n" +
            "For each review, reply in the format:\n[Review <index>]\n" +
            '\n'.join(f"Aspect: {asp}\nScore: <-100 to 100>\nReason: <text>" for asp in aspects) + "\n"
    )
    batch_prompt = '\n'.join(f"Review {i + 1}: {review}" for i, review in enumerate(reviews))

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {'role': 'system', 'content': prompt},
                {'role': 'user', 'content': batch_prompt}
            ], temperature=0.2, max_tokens=400 * len(reviews)
        )
        raw = response.choices[0].message.content.strip()

        # Parse response
        results = [None] * len(reviews)
        blocks = re.split(r'\[Review (\d+)\]', raw)[1:]  # Split by [Review X]
        for i in range(0, len(blocks), 2):
            index = int(blocks[i]) - 1  # Review index (1-based to 0-based)
            block = blocks[i + 1].strip()
            result = {}
            try:
                aspect_blocks = block.split('Aspect: ')[1:]  # Split by Aspect
                for asp_block in aspect_blocks:
                    lines = asp_block.splitlines()
                    asp = lines[0].strip()
                    score_match = re.search(r'Score:\s*([-]?\d+|N/A)', lines[1], re.IGNORECASE)
                    reason_match = re.search(r'Reason:\s*([^\n]+)', lines[2])
                    if not (score_match and reason_match):
                        raise ValueError("Incomplete response")
                    score = score_match.group(1)
                    reason = reason_match.group(1).strip()
                    score, reason = validate_sentiment_result(score, reason, asp)
                    result[asp] = (score, reason)
                # Ensure only requested aspects are included
                valid_result = {asp: result.get(asp, (0, 'Missing response')) for asp in aspects}
                results[index] = valid_result
            except (ValueError, AttributeError, IndexError) as e:
                print(f"Warning: Failed to parse sentiments for Review {index + 1}: {e}")
                results[index] = {asp: (0, 'Parsing error') for asp in aspects}

        # Fill missing results
        for i in range(len(reviews)):
            if results[i] is None:
                print(f"Warning: No sentiment result for Review {i + 1}")
                results[i] = {asp: (0, 'Missing response') for asp in aspects}

        return results

    except OpenAIError as e:
        print(f"Warning: API error during sentiment analysis: {e}")
        return [{asp: (0, 'API failure') for asp in aspects}] * len(reviews)


def main():
    validate_config()
    aspects = ASPECT_SETS.get(ASPECT_MODE, []) if MODE in ('sentiment', 'both') else []
    categories = COMPANIES[COMPANY] if MODE in ('classification', 'both') else []

    df = pd.read_csv(INPUT_FILE)
    df = validate_input_df(df)
    df['Year-Month'] = df['Date'].values.astype('datetime64[M]')

    if TEST_MODE:
        df = df.head(SAMPLE_SIZE)

    base = os.path.splitext(INPUT_FILE)[0]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"{base}_{timestamp}_output.csv"
    print(f"Saving results to {output_file}")

    initialize_output_file(output_file, aspects, bool(categories))

    client = OpenAI(api_key=get_api_key(LLM_PROVIDER))

    start = time.time()

    if USE_BATCHING:
        # Process reviews in batches
        reviews = df['Review'].tolist()
        for i in tqdm(range(0, len(reviews), BATCH_SIZE), desc="Processing review batches"):
            batch_reviews = reviews[i:i + BATCH_SIZE]
            batch_rows = df.iloc[i:i + BATCH_SIZE][['Date', 'Year-Month', 'Review']].to_dict('records')

            # Classification
            if categories:
                batch_results = classify_product_batch(batch_reviews, client, categories)
                for row, result in zip(batch_rows, batch_results):
                    if result:
                        row.update({
                            'prod_score': result[0],
                            'prod_category': result[1],
                            'prod_reason': result[2]
                        })
                    else:
                        row.update({
                            'prod_score': 0,
                            'prod_category': 'Unknown',
                            'prod_reason': 'Processing error'
                        })

            # Sentiment analysis
            if aspects:
                batch_results = analyze_sentiments_batch(batch_reviews, aspects, client)
                for row, result in zip(batch_rows, batch_results):
                    if result:
                        for asp, (score, reason) in result.items():
                            row[f"{asp}_score"] = score
                            row[f"{asp}_reason"] = reason
                    else:
                        for asp in aspects:
                            row[f"{asp}_score"] = 0
                            row[f"{asp}_reason"] = 'Processing error'

            # Append batch to output
            for row in batch_rows:
                append_row(output_file, row)
    else:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing individual reviews"):
            review = row['Review']  # No need for `or ''` since validate_input_df ensures valid reviews
            result = {'Date': row['Date'], 'Year-Month': row['Year-Month'], 'Review': review}

            if categories:
                prod_score, prod_cat, prod_reason = classify_product(review, client, categories)
                result.update({'prod_score': prod_score, 'prod_category': prod_cat, 'prod_reason': prod_reason})

            if aspects:
                sentiment_results = analyze_sentiments(review, aspects, client)
                for asp, (score, reason) in sentiment_results.items():
                    result[f"{asp}_score"] = score
                    result[f"{asp}_reason"] = reason

            append_row(output_file, result)

    duration = time.time() - start
    print(f"Completed {len(df)} reviews in {format_time(duration)}.")

    # Summary
    out_df = pd.read_csv(output_file)
    if categories:
        print("\nProduct category counts:")
        print(out_df['prod_category'].value_counts().to_string())
    if aspects:
        print("\nAverage sentiment scores:")
        for asp in aspects:
            col = f"{asp}_score"
            avg = out_df[col].mean()
            print(f"  {asp}: {avg:.1f}")

    print(f"Summary saved. Details in {output_file}")

main()
