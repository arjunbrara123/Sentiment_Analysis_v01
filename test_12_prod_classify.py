"""
################################################################################
#                                                                              #
#        AI-POWERED CUSTOMER REVIEW ANALYSIS & CATEGORIZATION PLATFORM         #
#                                                                              #
################################################################################

PURPOSE:
This script leverages advanced Artificial Intelligence (Generative AI) to automatically
read, understand, and analyze large volumes of customer reviews. Its primary goals are:
1. To classify each review into a predefined business category (e.g., 'Gas Products', 'Energy').
2. To score the sentiment of each review across key business aspects (e.g., 'Customer Service').
This provides rapid, data-driven insights into customer feedback, identifying key trends,
strengths, and areas for operational improvement.

OUTPUTS:
The script generates a single, timestamped CSV file (e.g., 'SourceFileName_20250612_132520_output.csv').
This file contains the original review data enriched with the AI-generated analysis, including:
- Product Category & Confidence Score
- Sentiment Scores for each business aspect (-100 to +100)
- A brief, AI-generated reason for each classification and score.

SCRIPT STRUCTURE (Table of Contents):
--------------------------------------
--- USER INPUTS ---
    User-editable settings to control the analysis.

--- BUSINESS LOGIC & DEFINITIONS ---
    Internal definitions for product categories and sentiment aspects.

--- SECTION 1: SCRIPT INITIALIZATION & SETUP ---
    Imports necessary libraries and defines core helper functions.

--- SECTION 2: AI INTERFACE & API WRAPPER ---
    Contains the function responsible for communicating with the selected AI model.

--- SECTION 3: CORE ANALYTICAL FUNCTIONS ---
    Houses the primary AI logic for classification and sentiment scoring,
    including both single-review and high-performance batch processing.

--- SECTION 4: MAIN WORKFLOW ORCHESTRATOR ---
    The main execution block that orchestrates the entire process from data
    loading to final report generation.
"""

# ==============================================================================
# --- USER INPUTS ---
# Users can adjust these settings without needing to
# understand the underlying code.
# ==============================================================================

# --- Primary Analysis Settings ---
COMPANY = 'British Gas Services'    # BUSINESS PURPOSE: Specifies the company being analyzed. This determines the
                                    # list of relevant product categories for classification.
                                    # Options: 'British Gas Services', 'British Gas Energy', 'HomeServe', etc.

MODE = 'both'                       # BUSINESS PURPOSE: Defines the type of analysis to perform.
                                    # 'classification': Assigns a product category to each review.
                                    # 'sentiment': Scores each review on key service aspects.
                                    # 'both': Performs both classification and sentiment analysis.

ASPECT_MODE = 'services'            # BUSINESS PURPOSE: Selects the set of criteria for sentiment analysis.
                                    # 'services': For service-based reviews (e.g., engineer visits).
                                    # 'energy': For energy supply reviews (e.g., billing, tariffs).

# --- Data & Execution Settings ---
INPUT_FILE = 'BG All TrustPilot Data.csv' # BUSINESS PURPOSE: The path to the source CSV file containing customer reviews.
                                          # This file MUST contain columns named 'Date' and 'Review'.

USE_BATCHING = True                 # BUSINESS PURPOSE: Determines how reviews are sent to the AI.
                                    # True: Processes reviews in larger, faster batches (highly recommended for >100 reviews).
                                    # False: Processes reviews one-by-one (slower, but useful for debugging).

BATCH_SIZE = 10                     # BUSINESS PURPOSE: The number of reviews to include in each batch if USE_BATCHING is True.
                                    # A larger size can increase speed but may hit API limits. Good values are 5-20.

TEST_MODE = True                    # BUSINESS PURPOSE: A switch for conducting a quick test run.
                                    # True: Analyzes only a small sample of reviews (defined by SAMPLE_SIZE).
                                    # False: Analyzes the entire input file.

SAMPLE_SIZE = 20                    # BUSINESS PURPOSE: The number of reviews to process when TEST_MODE is enabled.

# --- AI Model & API Key Settings ---
LLM_PROVIDER = 'anthropic'          # BUSINESS PURPOSE: Selects the Artificial Intelligence provider.
                                    # Options: 'openai', 'google', 'anthropic'.

# API Keys: The secret keys required to use the AI services.
# Leave these blank if you have set them up as system environment variables.
API_KEY_OPENAI = ''
API_KEY_GOOGLE = ''
API_KEY_ANTHROPIC = ''


# ==============================================================================
# --- BUSINESS LOGIC & DEFINITIONS ---
# These are the core business rules and categories used by the AI. They are
# separated from the configuration to protect them from accidental changes.
# ==============================================================================

# Defines the AI model to be used for each provider. We select models optimized
# for a balance of cost, speed, and analytical accuracy.
MODELS = {
    'openai': 'gpt-4o-mini',
    'google': 'gemini-1.5-flash-latest',
    'anthropic': 'claude-3-haiku-20240307'
}

# Detailed definitions that instruct the AI on how to categorize reviews.
# The clarity of these definitions is critical for accurate classification.
PRODUCT_DEFINITIONS = {
    'Gas Products': 'Covers boiler insurance, home care, annual service visits, or any instances where engineers visit customers homes to service, install, or fix boilers, central heating, or fix the hot water not running.',
    'Energy': 'Relates to British Gas as an energy / electricity supplier, or gas supply services, including tariffs, smart meters, and energy bills including charges and billing issues for unfixed tariffs. This category does not ever involve engineers visiting homes.',
    'Plumbing and Drains': 'Insurance for issues such as blocked drains, frozen pipes, or plumbing repairs, often handled by DynoRod or similar partners.',
    'Appliance Cover': 'Includes insurance for home appliances like ovens, washing machines, or any electrical appliances that we repair if they break down',
    'Home Electrical': 'Insurance for home electrics, including cover for wiring, fusebox breakdowns and broken sockets.',
    "Building": "General building services.",
    "Pest Control": "Removal of a pest infestation in the home, eg. Wasps and hornets nests, mice or rat infestation.",
    'Unknown': 'Please use this only if there is absolutely no information to categorize the review, after making every effort to find relevant clues. If you have to infer anything from non-conclusive evidence such as the company name or the sentiment of the review, it should be classificed as unknown.',
}

# Maps each company to its relevant set of product categories.
COMPANIES = {
    'British Gas Services': ['Gas Products', 'Energy', 'Plumbing and Drains', 'Appliance Cover', 'Home Electrical'],
    'British Gas Energy': ['Energy'],
    "HomeServe": ["Gas Products", "Plumbing and Drains", "Appliance Cover", "Home Electrical"],
    "CheckATrade": ["Gas Products", "Plumbing and Drains", "Home Electrical", "Building"],
    "Corgi HomePlan": ["Gas Products", "Plumbing and Drains", "Home Electrical", "Building"],
    "Domestic & General": ["Gas Products", "Plumbing and Drains", "Appliance Cover", "Home Electrical"],
    "247 Home Rescue": ["Gas Products", "Plumbing and Drains", "Appliance Cover", "Home Electrical", "Pest Control"],
    'Octopus': ['Energy'],
}
# Automatically adds the 'Unknown' category to every company's list.
for cats in COMPANIES.values():
    if "Unknown" not in cats:
        cats.append("Unknown")


# Defines the specific sets of criteria (aspects) for sentiment analysis.
ASPECT_SETS = {
    'services': ["Overall Sentiment", "Appointment Scheduling", "Customer Service", "Response Speed", "Engineer Experience", "Solution Quality", "Value for Money"],
    'energy': ["Overall Sentiment", "Appointment Scheduling", "Customer Service", "Energy Readings", "Meter Readings", "Value for Money"]
}

# ==============================================================================
# --- SECTION 1: SCRIPT INITIALIZATION & SETUP ---
# This section loads all necessary code packages and defines foundational
# utility functions for validation, file handling, and formatting.
# ==============================================================================
import os
import sys
import pandas as pd
import time
import re
from datetime import datetime, timedelta
from dotenv import load_dotenv
from tqdm import tqdm

# Import AI provider libraries
from openai import OpenAI, OpenAIError
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from anthropic import Anthropic, AnthropicError


def validate_configuration_settings():
    """
    Performs a rigorous check of all user-defined settings from the CONFIGURATION
    section to prevent errors during runtime.
    """
    print("Validating configuration settings...")
    if COMPANY not in COMPANIES:
        sys.exit(f"❌ CONFIGURATION ERROR: The 'COMPANY' named '{COMPANY}' is not a valid choice. Please select from: {list(COMPANIES.keys())}")
    if MODE not in ('classification', 'sentiment', 'both'):
        sys.exit(f"❌ CONFIGURATION ERROR: 'MODE' must be 'classification', 'sentiment', or 'both'.")
    if MODE in ('sentiment', 'both') and ASPECT_MODE not in ASPECT_SETS:
        sys.exit(f"❌ CONFIGURATION ERROR: 'ASPECT_MODE' for sentiment analysis must be one of: {list(ASPECT_SETS.keys())}")
    if not os.path.isfile(INPUT_FILE):
        sys.exit(f"❌ CONFIGURATION ERROR: The input file '{INPUT_FILE}' was not found.")
    if LLM_PROVIDER not in MODELS:
        sys.exit(f"❌ CONFIGURATION ERROR: 'LLM_PROVIDER' must be one of: {list(MODELS.keys())}")
    print("✅ Configuration validated successfully.")


def retrieve_llm_api_key(provider: str) -> str:
    """
    Securely retrieves the necessary API key for the selected AI provider.
    It first checks the script's CONFIGURATION section, then checks for
    system environment variables (a more secure practice).
    """
    load_dotenv()  # Load environment variables from a .env file if it exists.
    key_map = {
        'openai': API_KEY_OPENAI or os.getenv('OPENAI_API_KEY'),
        'google': API_KEY_GOOGLE or os.getenv('GOOGLE_API_KEY'),
        'anthropic': API_KEY_ANTHROPIC or os.getenv('ANTHROPIC_API_KEY'),
    }
    api_key = key_map.get(provider)
    if not api_key:
        env_var_name = f"{provider.upper()}_API_KEY"
        sys.exit(f"❌ API KEY ERROR: The API key for '{provider}' is missing. Please add it to the CONFIGURATION section or set it as the '{env_var_name}' environment variable.")
    return api_key


def format_elapsed_time_for_reporting(seconds: float) -> str:
    """Converts a duration in seconds into a human-readable HH:MM:SS format."""
    return str(timedelta(seconds=int(seconds)))


def create_and_initialize_output_file(output_path: str, active_aspects: list, is_classification_active: bool):
    """
    Generates a new, empty CSV file with the correct headers for the analysis results.
    This ensures a clean output file for every run.
    """
    headers = ['Date', 'Year-Month', 'Review']
    if is_classification_active:
        headers += ['Product_Category_Confidence', 'Product_Category', 'Product_Category_Reasoning']
    if active_aspects:
        for aspect in active_aspects:
            # Sanitize aspect names for use as column headers
            clean_aspect = aspect.replace(' ', '_')
            headers += [f"{clean_aspect}_Score", f"{clean_aspect}_Reasoning"]

    pd.DataFrame(columns=headers).to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ Output file created at: {output_path}")


def append_analysis_results_to_output_file(output_path: str, result_data: dict):
    """Appends a single row of analysis results to the specified output CSV file."""
    pd.DataFrame([result_data]).to_csv(output_path, mode='a', header=False, index=False, encoding='utf-8-sig')


def validate_and_sanitize_classification_output(score_str: str, category: str, reason: str, valid_categories: list) -> tuple:
    """
    Validates the AI's classification output to ensure it matches the expected format and values.
    If validation fails, it returns safe, default values to avoid corrupting the data.
    """
    try:
        score = int(score_str)
        if not (0 <= score <= 100):
            raise ValueError("Confidence score is outside the 0-100 range.")
        if category not in valid_categories:
            # A simple fix for when the AI adds extra text.
            matched_cat = next((cat for cat in valid_categories if cat in category), None)
            if matched_cat:
                category = matched_cat
            else:
                raise ValueError(f"Category '{category}' is not a valid option.")
        if not reason or not reason.strip():
            reason = "No reasoning provided by AI."
        return score, category, reason
    except (ValueError, TypeError) as e:
        print(f"⚠️ Warning: Invalid classification data from AI ({e}). Defaulting to 'Unknown'.")
        return 0, 'Unknown', f'Invalid AI Response: {e}'


def validate_and_sanitize_sentiment_output(score_str: str, reason: str, aspect_name: str) -> tuple:
    """
    Validates the AI's sentiment score to ensure it's a valid number in the expected range.
    If validation fails, it returns a neutral score of 0.
    """
    try:
        # Handle cases where the AI correctly identifies an aspect is not mentioned.
        if isinstance(score_str, str) and score_str.strip().lower() in ('n/a', 'na', 'none'):
            score = 0
            reason = f"Aspect '{aspect_name}' was not mentioned in the review."
        else:
            score = int(score_str)
            if not (-100 <= score <= 100):
                raise ValueError("Sentiment score is outside the -100 to 100 range.")
        if not reason or not reason.strip():
            reason = "No reasoning provided by AI."
        return score, reason
    except (ValueError, TypeError) as e:
        print(f"⚠️ Warning: Invalid sentiment data for aspect '{aspect_name}' ({e}). Defaulting to 0.")
        return 0, f'Invalid AI Response: {e}'


# ==============================================================================
# --- SECTION 2: AI INTERFACE & API WRAPPER ---
# This section contains the technical function for communicating with the
# selected AI provider's API. It acts as a universal translator, handling the
# unique requirements of each AI service (OpenAI, Google, Anthropic).
# ==============================================================================

def execute_llm_api_call(client, provider: str, model: str, system_prompt: str, user_prompt: str, max_tokens=500, temperature=0.1) -> str:
    """
    PURPOSE: To send a request to the designated AI model and retrieve its response.
    PROCESS: This function formats the request according to the specific provider's
             API standards (e.g., how OpenAI, Google, and Anthropic expect to receive
             instructions) and handles potential network errors or API failures gracefully.
    BUSINESS VALUE: It provides a reliable, centralized point of communication with
                    the AI, making the system resilient to API issues and easy to
                    update if a new AI provider is added.
    """
    try:
        if provider == 'openai':
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()

        elif provider == 'google':
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature
            )
            # Safety settings are relaxed to prevent the AI from refusing to analyze
            # reviews that it might mistakenly deem "harmful" (e.g., very angry customer feedback).
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
            response = client.generate_content(
                full_prompt,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            return response.text.strip()

        elif provider == 'anthropic':
            response = client.messages.create(
                model=model,
                system=system_prompt,
                messages=[{'role': 'user', 'content': user_prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.content[0].text.strip()

    except (OpenAIError, AnthropicError, Exception) as e:
        print(f"API CALL FAILED: An error occurred with the '{provider}' service: {e}")
        return "" # Return an empty string to signal a failure.


# ==============================================================================
# --- SECTION 3: CORE ANALYTICAL FUNCTIONS ---
# These functions contain the "brains" of the operation. They construct the
# detailed instructions (prompts) for the AI and parse its responses into
# structured data. They include methods for single reviews and high-performance batches.
# ==============================================================================

def categorize_review_into_product_area(review_text: str, client, provider: str, model: str, valid_categories: list) -> tuple:
    """Instructs the AI to classify a single customer review into one of the predefined product categories."""
    category_definitions_text = ""
    for cat in valid_categories:
        if cat in PRODUCT_DEFINITIONS:
            category_definitions_text += f"- **{cat}**: {PRODUCT_DEFINITIONS[cat]}\n"

    system_prompt = (
        "You are a meticulous product analysis expert. Your task is to categorize a customer review based on the detailed definitions provided. "
        "Reply **only** in the following format, with no extra text:\n"
        "Score: <0-100 confidence score>\nProduct: <category name>\nReason: <brief explanation>\n\n"
        "**Available Categories and Official Definitions:**\n"
        f"{category_definitions_text}"
    )
    user_prompt = f"Review to analyze: \"{review_text}\""
    raw_response = execute_llm_api_call(client, provider, model, system_prompt, user_prompt, max_tokens=150, temperature=0.0)
    if not raw_response:
        return 0, 'Unknown', 'AI API call failed'
    try:
        score_match = re.search(r'Score:\s*(\d+)', raw_response)
        product_match = re.search(r'Product:\s*([^\n]+)', raw_response)
        reason_match = re.search(r'Reason:\s*([^\n]+)', raw_response)
        if not (score_match and product_match and reason_match):
            raise ValueError("Response from AI did not match expected format.")
        return validate_and_sanitize_classification_output(
            score_match.group(1), product_match.group(1).strip(), reason_match.group(1).strip(), valid_categories
        )
    except (ValueError, AttributeError) as e:
        return 0, 'Unknown', f'AI response parsing error: {e}'


def score_sentiment_of_review_across_aspects(review_text: str, aspects: list, client, provider: str, model: str) -> dict:
    """Instructs the AI to perform a multi-faceted sentiment analysis on a single review."""
    system_prompt = (
        "You are a precision sentiment analysis expert. For the customer review provided, evaluate it against each of the following business aspects. "
        "Provide a sentiment score from -100 to +100 and a brief justification. If an aspect is not mentioned, score it 'N/A'.\n\n"
        "**Aspects to Evaluate:**\n" + '\n'.join(f"- {asp}" for asp in aspects) + "\n\n"
        "Reply **only** in the following format for each aspect, repeating the block for every aspect listed:\n"
        "Aspect: <aspect name>\nScore: <score or N/A>\nReason: <brief justification>"
    )
    user_prompt = f"Review to analyze: \"{review_text}\""
    raw_response = execute_llm_api_call(client, provider, model, system_prompt, user_prompt, max_tokens=700, temperature=0.0)
    if not raw_response:
        return {asp: (0, 'AI API call failed') for asp in aspects}
    results = {}
    try:
        aspect_blocks = raw_response.strip().split('Aspect: ')
        for block in aspect_blocks:
            if not block.strip(): continue
            lines = block.strip().split('\n')
            if len(lines) < 3: continue
            aspect_name, score_match, reason_match = lines[0].strip(), re.search(r'Score:\s*([-]?\d+|N/A|NA)', lines[1], re.IGNORECASE), re.search(r'Reason:\s*([^\n]+)', lines[2])
            if aspect_name in aspects and score_match and reason_match:
                score, reason = validate_and_sanitize_sentiment_output(score_match.group(1), reason_match.group(1).strip(), aspect_name)
                results[aspect_name] = (score, reason)
        return {asp: results.get(asp, (0, 'AI failed to provide a response.')) for asp in aspects}
    except (ValueError, AttributeError, IndexError) as e:
        return {asp: (0, f'AI response parsing error: {e}') for asp in aspects}


def categorize_batch_of_reviews_into_product_areas(reviews: list, client, provider: str, model: str, valid_categories: list) -> list:
    """
    PURPOSE: To classify a batch of reviews in a single, efficient API call.
    PROCESS: This function bundles multiple reviews into one large request. It instructs the AI to analyze each
             review and provide a structured response for each one, identified by an index tag (e.g., "[Review 1]").
             The function then parses this single large response back into individual results.
    BUSINESS VALUE: Dramatically increases processing speed and reduces API costs for large datasets,
                    making large-scale analysis feasible and affordable.
    """
    category_definitions_text = "\n".join(f"- **{cat}**: {PRODUCT_DEFINITIONS.get(cat, '')}" for cat in valid_categories if cat in PRODUCT_DEFINITIONS)
    system_prompt = (
        "You are a meticulous product analysis expert. You will be given a numbered list of customer reviews. "
        "Analyze each review individually and categorize it based on the definitions provided. Your response must contain a block for each review.\n\n"
        "**Available Categories and Official Definitions:**\n"
        f"{category_definitions_text}\n\n"
        "For each review, reply **only** in the following format:\n"
        "[Review <index>]\nScore: <0-100>\nProduct: <category>\nReason: <brief explanation>"
    )
    user_prompt = "\n".join(f"[Review {i+1}]: \"{review}\"" for i, review in enumerate(reviews))
    # Estimate required tokens: ~100 tokens per review classification. Add a buffer.
    max_tokens_needed = len(reviews) * 150
    raw_response = execute_llm_api_call(client, provider, model, system_prompt, user_prompt, max_tokens=max_tokens_needed, temperature=0.0)

    if not raw_response:
        return [(0, 'Unknown', 'AI API call failed')] * len(reviews)

    results = [None] * len(reviews)
    # Split the response by "[Review X]" tags
    review_blocks = re.split(r'\[Review (\d+)\]', raw_response)
    # The split results in a list like ['', '1', '...response 1...', '2', '...response 2...']
    for i in range(1, len(review_blocks), 2):
        try:
            index = int(review_blocks[i]) - 1
            if 0 <= index < len(reviews):
                block_content = review_blocks[i+1].strip()
                score_match = re.search(r'Score:\s*(\d+)', block_content)
                product_match = re.search(r'Product:\s*([^\n]+)', block_content)
                reason_match = re.search(r'Reason:\s*([^\n]+)', block_content)
                if not (score_match and product_match and reason_match):
                    raise ValueError("Incomplete response block.")
                results[index] = validate_and_sanitize_classification_output(
                    score_match.group(1), product_match.group(1).strip(), reason_match.group(1).strip(), valid_categories
                )
        except (ValueError, IndexError, AttributeError) as e:
            print(f"⚠️ Warning: Could not parse classification for a review in the batch. Error: {e}")

    # Fill in any results that were missing from the AI's response
    return [res if res is not None else (0, 'Unknown', 'Missing AI response in batch') for res in results]


def score_sentiment_for_batch_of_reviews(reviews: list, aspects: list, client, provider: str, model: str) -> list:
    """
    PURPOSE: To perform multi-faceted sentiment analysis on a batch of reviews in a single API call.
    PROCESS: Similar to batch classification, this bundles reviews and sends them in one request. It asks the
             AI to return a structured block for each review, which in turn contains sub-blocks for each
             sentiment aspect. The function then parses this nested structure.
    BUSINESS VALUE: Provides the same deep, nuanced sentiment insights as the single-review method but at
                    a much higher throughput and lower cost, enabling rapid analysis of customer sentiment at scale.
    """
    system_prompt = (
        "You are a precision sentiment analysis expert. For each numbered review provided, evaluate it against all of the business aspects listed. "
        "If an aspect is not mentioned, score it 'N/A'.\n\n"
        "**Aspects to Evaluate:**\n" + '\n'.join(f"- {asp}" for asp in aspects) + "\n\n"
        "For each review, reply using this exact nested format:\n"
        "[Review <index>]\n"
        "Aspect: <aspect name>\nScore: <score or N/A>\nReason: <brief justification>\n"
        "(...repeat for all aspects...)"
    )
    user_prompt = "\n".join(f"[Review {i+1}]: \"{review}\"" for i, review in enumerate(reviews))
    # Estimate required tokens: ~50 tokens per aspect * number of aspects. Add a buffer.
    max_tokens_needed = len(reviews) * len(aspects) * 75
    raw_response = execute_llm_api_call(client, provider, model, system_prompt, user_prompt, max_tokens=max_tokens_needed, temperature=0.0)

    if not raw_response:
        return [{asp: (0, 'AI API call failed') for asp in aspects}] * len(reviews)

    results = [None] * len(reviews)
    review_blocks = re.split(r'\[Review (\d+)\]', raw_response)
    for i in range(1, len(review_blocks), 2):
        try:
            index = int(review_blocks[i]) - 1
            if 0 <= index < len(reviews):
                block_content = review_blocks[i+1].strip()
                aspect_results = {}
                aspect_sub_blocks = block_content.strip().split('Aspect: ')
                for sub_block in aspect_sub_blocks:
                    if not sub_block.strip(): continue
                    lines = sub_block.strip().split('\n')
                    if len(lines) < 3: continue
                    aspect_name = lines[0].strip()
                    score_match = re.search(r'Score:\s*([-]?\d+|N/A|NA)', lines[1], re.IGNORECASE)
                    reason_match = re.search(r'Reason:\s*([^\n]+)', lines[2])
                    if aspect_name in aspects and score_match and reason_match:
                        score, reason = validate_and_sanitize_sentiment_output(score_match.group(1), reason_match.group(1).strip(), aspect_name)
                        aspect_results[aspect_name] = (score, reason)
                # Ensure all aspects are present in the final dict for this review
                results[index] = {asp: aspect_results.get(asp, (0, 'Missing AI response in batch')) for asp in aspects}
        except (ValueError, IndexError, AttributeError) as e:
            print(f"⚠️ Warning: Could not parse sentiment for a review in the batch. Error: {e}")

    # Fill in any whole reviews that were missing
    default_result = {asp: (0, 'Missing AI response in batch') for asp in aspects}
    return [res if res is not None else default_result for res in results]


def main_workflow():
    """
    This is the main orchestrator of the entire script. It executes the analysis
    process in a clear, sequential order.
    """
    # --------------------------------------------------------------------------
    # STEP 1: SYSTEM AND CONFIGURATION VALIDATION
    # --------------------------------------------------------------------------
    validate_configuration_settings()
    api_key = retrieve_llm_api_key(LLM_PROVIDER)
    model_name = MODELS[LLM_PROVIDER]
    active_categories = COMPANIES[COMPANY] if MODE in ('classification', 'both') else []
    active_aspects = ASPECT_SETS.get(ASPECT_MODE, []) if MODE in ('sentiment', 'both') else []

    # --------------------------------------------------------------------------
    # STEP 2: SOURCE DATA LOADING AND HYGIENE
    # --------------------------------------------------------------------------
    print(f"\nLoading source data from '{INPUT_FILE}'...")
    try:
        df = pd.read_csv(INPUT_FILE)
        if 'Review' not in df.columns or 'Date' not in df.columns: sys.exit("❌ DATA ERROR: Input CSV must contain 'Date' and 'Review' columns.")
        df.dropna(subset=['Review'], inplace=True)
        df = df[df['Review'].str.strip() != '']
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df.dropna(subset=['Date'], inplace=True)
        df['Year-Month'] = df['Date'].dt.to_period('M')
    except Exception as e:
        sys.exit(f"❌ DATA ERROR: Could not read or process the input file. Error: {e}")

    if TEST_MODE:
        df = df.head(SAMPLE_SIZE)
        print(f"⚠️ TEST MODE ENABLED: Processing only the first {SAMPLE_SIZE} reviews.")
    print(f"✅ Data loaded successfully. Found {len(df)} valid reviews to analyze.")

    # --------------------------------------------------------------------------
    # STEP 3: AI MODEL AND OUTPUT FILE INITIALIZATION
    # --------------------------------------------------------------------------
    print(f"\nInitializing AI Model: {LLM_PROVIDER.capitalize()} ({model_name})...")
    client = None
    if LLM_PROVIDER == 'openai': client = OpenAI(api_key=api_key)
    elif LLM_PROVIDER == 'google': genai.configure(api_key=api_key); client = genai.GenerativeModel(model_name)
    elif LLM_PROVIDER == 'anthropic': client = Anthropic(api_key=api_key)
    print("✅ AI model initialized.")

    base_filename = os.path.splitext(os.path.basename(INPUT_FILE))[0]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f"{base_filename}_{timestamp}_output.csv"
    create_and_initialize_output_file(output_filename, active_aspects, bool(active_categories))

    # --------------------------------------------------------------------------
    # STEP 4: DEPLOYING AI FOR REVIEW ANALYSIS (CORE ENGINE)
    # --------------------------------------------------------------------------
    start_time = time.time()
    print(f"\n🚀 Starting analysis of {len(df)} reviews...")
    if USE_BATCHING:
        print(f"Using high-performance BATCH processing (Batch Size: {BATCH_SIZE})...")
        # Process reviews in batches
        for i in tqdm(range(0, len(df), BATCH_SIZE), desc="Analyzing Batches"):
            batch_df = df.iloc[i:i + BATCH_SIZE]
            batch_reviews = batch_df['Review'].tolist()

            class_results, sentiment_results = None, None
            # --- 4.1: Batch Product Classification ---
            if MODE in ('classification', 'both'):
                class_results = categorize_batch_of_reviews_into_product_areas(batch_reviews, client, LLM_PROVIDER, model_name, active_categories)
            # --- 4.2: Batch Sentiment Scoring ---
            if MODE in ('sentiment', 'both'):
                sentiment_results = score_sentiment_for_batch_of_reviews(batch_reviews, active_aspects, client, LLM_PROVIDER, model_name)

            # --- Combine and write results for the batch ---
            for j, (idx, row) in enumerate(batch_df.iterrows()):
                result_data = {'Date': row['Date'].strftime('%Y-%m-%d'), 'Year-Month': str(row['Year-Month']), 'Review': row['Review']}
                if class_results:
                    score, cat, reason = class_results[j]
                    result_data.update({'Product_Category_Confidence': score, 'Product_Category': cat, 'Product_Category_Reasoning': reason})
                if sentiment_results:
                    for aspect, (score, reason) in sentiment_results[j].items():
                        result_data[f"{aspect.replace(' ', '_')}_Score"] = score
                        result_data[f"{aspect.replace(' ', '_')}_Reasoning"] = reason
                append_analysis_results_to_output_file(output_filename, result_data)

    else:
        print("Using standard one-by-one processing...")
        # Process reviews individually
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing Reviews"):
            result_data = {'Date': row['Date'].strftime('%Y-%m-%d'), 'Year-Month': str(row['Year-Month']), 'Review': row['Review']}
            if MODE in ('classification', 'both'):
                score, cat, reason = categorize_review_into_product_area(row['Review'], client, LLM_PROVIDER, model_name, active_categories)
                result_data.update({'Product_Category_Confidence': score, 'Product_Category': cat, 'Product_Category_Reasoning': reason})
            if MODE in ('sentiment', 'both'):
                sentiment_results = score_sentiment_of_review_across_aspects(row['Review'], active_aspects, client, LLM_PROVIDER, model_name)
                for aspect, (score, reason) in sentiment_results.items():
                    result_data[f"{aspect.replace(' ', '_')}_Score"] = score
                    result_data[f"{aspect.replace(' ', '_')}_Reasoning"] = reason
            append_analysis_results_to_output_file(output_filename, result_data)

    elapsed_time = time.time() - start_time
    print(f"\n✅ Analysis complete. Processed {len(df)} reviews in {format_elapsed_time_for_reporting(elapsed_time)}.")

    # --------------------------------------------------------------------------
    # STEP 5: SYNTHESIZING KEY PERFORMANCE INDICATORS (KPIs)
    # --------------------------------------------------------------------------
    print("\n" + "="*50)
    print("        EXECUTIVE SUMMARY OF ANALYSIS RESULTS")
    print("="*50)
    summary_df = pd.read_csv(output_filename)
    if MODE in ('classification', 'both'):
        print("\n📊 Product Category Distribution:")
        print(summary_df['Product_Category'].value_counts().to_string())
    if MODE in ('sentiment', 'both'):
        print("\n📈 Average Sentiment Scores (-100 to +100):")
        for aspect in active_aspects:
            col_name = f"{aspect.replace(' ', '_')}_Score"
            if col_name in summary_df.columns:
                average_score = summary_df[col_name].mean()
                print(f"  - {aspect:<25}: {average_score:6.1f}")
    print("\n" + "="*50)
    print(f"Full, detailed results have been saved to: {output_filename}")

main_workflow()