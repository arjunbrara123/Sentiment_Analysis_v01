#!/usr/bin/env python3
"""
review_synthesis_generator.py

A script to take customer reviews and generate structured, HTML-formatted analysis
summaries (Strengths, Weaknesses, Improvements, Growth, Specific Aspects) using an LLM.

Designed for non-technical users to configure and run.

HOW TO USE:
  1.  At the top of this file, edit the CONFIGURATION section:
      - INPUT_FILE: Path to your input CSV. It *must* contain 'Date', 'Review', and 'Product' columns.
      - OUTPUT_FILE: Name for the CSV file where results will be saved.
      - API_KEY: Your OpenAI API key (or set OPENAI_API_KEY in your environment).
      - MODEL: The OpenAI model to use (e.g., "gpt-4o-mini", "gpt-4o").
               "gpt-4o" might produce higher quality but is more expensive.
      - COMPANY_NAME: Your company's name, used in prompts for context.
      - GENERAL_ANALYSIS_TYPES: List of overall summary types you want (e.g., Strength).
      - DETAILED_SERVICE_ASPECTS: List of specific service aspects to analyze.
      - MAX_REVIEWS_FOR_SYNTHESIS: How many (most recent) reviews to use for each analysis point.
                                   Too many might exceed LLM context limits or increase cost.
      - REQUEST_DELAY_SECONDS: Time to wait between OpenAI API calls to avoid rate limits.
      - NO_INSIGHT_PLACEHOLDER: Text to use if the LLM finds no specific insights for a category.

  2.  Install dependencies (if you haven't already for the previous script):
        pip install openai pandas python-dotenv tqdm

  3.  Run from your terminal:
        python review_synthesis_generator.py

FLOW:
  - Validate configuration.
  - Load and validate the input CSV.
  - Group reviews by Product and Year.
  - For each Product:
    - For each configured Analysis Type (e.g., Strength, Appointment Scheduling):
      - Select relevant recent reviews.
      - Construct a detailed prompt instructing the LLM on the desired HTML output format.
      - Call the OpenAI API.
      - Store the LLM's generated analysis.
  - Save all generated analyses to the output CSV.
"""

# ---------------- CONFIGURATION (EDIT ME) ----------------
INPUT_FILE = "test_reviews_01.csv"  # CSV with 'Date', 'Review', 'Product'
API_KEY = ""  # Leave blank to use OPENAI_API_KEY from environment variables
MODEL = "gpt-4o-mini"  # "gpt-4o" for higher quality, "gpt-4o-mini" for speed/lower cost
COMPANY_NAME = "British Gas" # This will be used in the LLM prompts

# Define the types of analysis to perform for each product.
# The script will generate one row in the output CSV for each product combined with each of these.
GENERAL_ANALYSIS_TYPES = ["Strength", "Weakness", "Improvement"]
DETAILED_SERVICE_ASPECTS = [
    "Appointment Scheduling", "Customer Service", "Response Speed",
    "Engineer Experience", "Solution Quality", "Value for Money"
]
# If you only want strengths and weaknesses, you could change the lists like so:
# GENERAL_ANALYSIS_TYPES = ["Strength", "Weakness"]
# DETAILED_SERVICE_ASPECTS = []


# How many of the most recent reviews to provide to the LLM for each Product/Analysis Type.
# The LLM will synthesize these reviews to create its analysis.
# Adjust based on context window of the model and desired level of detail vs. cost.
MAX_REVIEWS_FOR_SYNTHESIS = 15 # Using 15-25 reviews often provides a good balance.

# Delay between API calls (in seconds) to help avoid hitting OpenAI rate limits.
# Increase if you encounter rate limit errors.
REQUEST_DELAY_SECONDS = 5

# Placeholder text to use in the output 'Analysis' column if the LLM determines
# it cannot generate a meaningful summary for a given product/aspect combination.
NO_INSIGHT_PLACEHOLDER = "No specific insights found in the provided reviews for this category."
# ---------------------------------------------------------

import pandas as pd
import time
from datetime import datetime
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv
import os
import sys
from tqdm import tqdm
import logging

# Setup basic logging for warnings and errors
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
OUTPUT_FILE = INPUT_FILE[:-4] + "_" + timestamp + "_output.csv"

# --- PREDEFINED STYLES FOR PROMPT (based on the 2024 example format) ---
# These are examples of the HTML spans the LLM should use when highlighting specific aspects.
# The prompt will guide the LLM to use these.
ASPECT_STYLES_EXAMPLES_FOR_PROMPT = {
    "Appointment Scheduling": '<span style="background: lightcyan">⌚ Appointment Scheduling</span>',
    "Customer Service": '<span style="background: mistyrose">📞 Customer Service</span>',
    "Response Speed": '<span style="background: PapayaWhip">🥇 Response Speed</span>',
    "Engineer Experience": '<span style="background: lightcyan">🧑‍🔧 Engineer Experience</span>',
    "Solution Quality": '<span style="background: lavenderblush">🧠 Solution Quality</span>',
    "Value For Money": '<span style="background: Honeydew">💵 Value For Money</span>',
}

def validate_config():
    """Checks if the essential configurations are set."""
    if not INPUT_FILE or not os.path.isfile(INPUT_FILE):
        logging.error(f"Error: INPUT_FILE '{INPUT_FILE}' not found or not specified.")
        sys.exit(1)
    if not OUTPUT_FILE:
        logging.error("Error: OUTPUT_FILE path is not specified.")
        sys.exit(1)
    if not MODEL:
        logging.error("Error: MODEL is not specified.")
        sys.exit(1)
    if not COMPANY_NAME:
        logging.warning("Warning: COMPANY_NAME is not specified. Prompts may be less specific.")
    if not isinstance(MAX_REVIEWS_FOR_SYNTHESIS, int) or MAX_REVIEWS_FOR_SYNTHESIS <= 0:
        logging.error("Error: MAX_REVIEWS_FOR_SYNTHESIS must be a positive integer.")
        sys.exit(1)
    if not isinstance(REQUEST_DELAY_SECONDS, (int, float)) or REQUEST_DELAY_SECONDS < 0:
        logging.error("Error: REQUEST_DELAY_SECONDS must be a non-negative number.")
        sys.exit(1)
    logging.info("Configuration validated successfully.")

def get_api_key_from_env_or_config():
    """Gets the OpenAI API key from the config variable or environment."""
    if API_KEY:
        logging.info("Using API key from script configuration.")
        return API_KEY
    load_dotenv()  # Load .env file if it exists
    env_key = os.getenv('OPENAI_API_KEY')
    if not env_key:
        logging.error("Error: OpenAI API key not found. "
                        "Please set API_KEY in the script or OPENAI_API_KEY in your environment.")
        sys.exit(1)
    logging.info("Using API key from environment variable.")
    return env_key

def initialize_openai_client(api_key_value):
    """Initializes and returns the OpenAI client."""
    try:
        client = OpenAI(api_key=api_key_value)
        # Perform a simple test call, e.g., listing models (optional, but good for validation)
        # client.models.list() # This would confirm authentication
        logging.info("OpenAI client initialized successfully.")
        return client
    except OpenAIError as e:
        logging.error(f"Error initializing OpenAI client: {e}")
        sys.exit(1)

def load_and_validate_input_df(filepath):
    """Loads the input CSV and validates required columns."""
    try:
        df = pd.read_csv(filepath)
        logging.info(f"Successfully loaded input CSV: {filepath}")
    except FileNotFoundError:
        logging.error(f"Error: Input file '{filepath}' not found.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error reading CSV file '{filepath}': {e}")
        sys.exit(1)

    required_cols = ['Date', 'Review', 'Product']
    for col in required_cols:
        if col not in df.columns:
            logging.error(f"Error: Input CSV must contain the column '{col}'.")
            sys.exit(1)

    # Convert 'Date' to datetime and handle errors
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    if df['Date'].isna().any():
        logging.warning("Warning: Some rows have invalid dates and will be skipped for year extraction or sorting.")
        df.dropna(subset=['Date'], inplace=True)

    # Ensure 'Review' and 'Product' are not empty
    df.dropna(subset=['Review', 'Product'], inplace=True)
    df = df[df['Review'].astype(str).str.strip() != '']
    df = df[df['Product'].astype(str).str.strip() != '']

    if df.empty:
        logging.error("Error: No valid data found in the input CSV after validation and cleaning.")
        sys.exit(1)

    df['Year'] = df['Date'].dt.year
    logging.info("Input CSV loaded and validated.")
    return df

def format_reviews_for_prompt(review_list):
    """Formats a list of review texts for inclusion in the LLM prompt."""
    if not review_list:
        return "No reviews provided."
    # Simple formatting: enumerate reviews
    return "\n".join([f"Review {i+1}: {review}" for i, review in enumerate(review_list)])

def construct_llm_prompt(company_name, product_name, analysis_type, reviews_text, aspect_styles_examples):
    """
    Constructs the detailed prompt for the LLM. This is the most crucial part.
    It instructs the LLM on its role, the task, and the exact output format.
    """

    # General instructions
    prompt = f"You are an expert market review analyst for {company_name}.\n"
    prompt += f"Your task is to synthesize insights from a list of customer reviews for the product: '{product_name}'.\n"
    prompt += f"You need to perform an analysis focusing on: '{analysis_type}'.\n\n"
    prompt += "INSTRUCTIONS FOR OUTPUT FORMATTING:\n"
    prompt += "- Your entire response MUST be only the HTML-like formatted analysis. Do not include any introductory or concluding text outside of this formatted block.\n"
    prompt += "- Use emojis at the start of each major point/title.\n"
    prompt += "- Use '<b><u>Catchy Title Text</u></b>:' for main points.\n"
    prompt += "- Incorporate direct quotes from reviews where relevant, formatted as '<i>\"quote text\"</i>'.\n"
    prompt += "- Use '<br><br>' for paragraph breaks between distinct points.\n"
    prompt += "- If a point clearly relates to a specific service aspect, highlight that aspect at the end of the point using the predefined span styles. Examples of aspects and their styles:\n"
    for aspect, style_example in aspect_styles_examples.items():
        prompt += f"  - {aspect}: {style_example}\n"
    prompt += "- If no specific insights or relevant reviews are found for this product/analysis type combination, return ONLY the text: "
    prompt += f"'{NO_INSIGHT_PLACEHOLDER}'\n\n"

    # Type-specific instructions (based on 2024 example formats)
    if analysis_type == "Strength" or analysis_type == "Weakness":
        prompt += f"For '{analysis_type}' analysis:\n"
        prompt += "Identify 2-4 key themes. For each theme:\n"
        prompt += "  1. Start with an appropriate emoji and a bolded, underlined title (e.g., '💡 <b><u>Key Strength Area</u></b>:').\n"
        prompt += "  2. Explain the theme, using insights and quotes from the reviews.\n"
        prompt += "  3. If applicable, add the relevant aspect span tag at the end of the description.\n"
        prompt += "  4. Separate themes with '<br><br>'.\n\n"
    elif analysis_type == "Improvement":
        prompt += f"For '{analysis_type}' analysis:\n"
        prompt += "Identify 2-4 key areas for improvement. For each area:\n"
        prompt += "  1. Start with an appropriate emoji and a bolded, underlined title for the improvement area (e.g., '🛠️ <b><u>Rescheduling Process</u></b>:').\n"
        prompt += "  2. Describe the issue based on reviews, including quotes.\n"
        prompt += "  3. Clearly state an 'Actionable Item:' (e.g., 'Actionable Item: Implement real-time online rescheduling and SMS updates.').\n"
        prompt += "  4. If applicable, add the relevant aspect span tag at the end.\n"
        prompt += "  5. Separate areas with '<br><br>'.\n\n"
    elif analysis_type == "Growth":
        prompt += f"For '{analysis_type}' analysis:\n"
        prompt += "Identify 1-2 key growth opportunities. For each opportunity:\n"
        prompt += "  1. Start with an appropriate emoji.\n"
        prompt += "  2. Provide market insight or context, possibly referencing competitor data or social media trends if mentioned in reviews or if you can infer general knowledge (be generic if specific data is not in reviews).\n"
        prompt += "  3. Conclude with a bolded '💡 <b>Key Takeaway</b>:' followed by an actionable growth strategy for {company_name} regarding {product_name}.\n"
        prompt += "  4. Separate opportunities with '<br><br>'.\n\n"
    elif analysis_type in aspect_styles_examples: # For detailed service aspects
        prompt += f"For the specific aspect '{analysis_type}':\n"
        prompt += "  1. Provide a concise summary of findings related ONLY to this aspect from the reviews.\n"
        prompt += "  2. Mention common praises, complaints, or suggestions.\n"
        prompt += "  3. If improvements are suggested or implied, you can include an 'Actionable:' point.\n"
        prompt += "  4. The overall tone should be a factual summary for this specific aspect.\n"
        prompt += f"  5. Start with an emoji relevant to '{analysis_type}'.\n\n"
    else: # Fallback for any other analysis type
        prompt += f"For '{analysis_type}' analysis:\n"
        prompt += "  - Summarize the key findings from the reviews related to this topic.\n"
        prompt += "  - Structure your response with clear points, using emojis and bold titles as appropriate.\n\n"

    prompt += "CUSTOMER REVIEWS TO ANALYZE:\n"
    prompt += f"{reviews_text}\n\n"
    prompt += "Generate ONLY the HTML-like formatted analysis based on these instructions and reviews."
    return prompt

def call_llm_with_retry(client, model, system_prompt, user_content, max_retries=3, delay_seconds=REQUEST_DELAY_SECONDS):
    """Calls the LLM with a retry mechanism for transient errors."""
    for attempt in range(max_retries):
        try:
            logging.info(f"Attempting LLM call (attempt {attempt + 1}/{max_retries}) for model {model}...")
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content} # User content is minimal here as reviews are in system prompt
                ],
                temperature=0.3, # Lower temperature for more factual, less creative output
                # max_tokens can be set if you have an idea of output length, e.g., 1000-1500
            )
            analysis_text = completion.choices[0].message.content.strip()
            logging.info("LLM call successful.")
            # A small delay after each successful call to respect rate limits
            time.sleep(delay_seconds)
            return analysis_text
        except OpenAIError as e:
            logging.warning(f"OpenAI API error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                logging.info(f"Retrying in {delay_seconds * (attempt + 1)} seconds...")
                time.sleep(delay_seconds * (attempt + 1)) # Exponential backoff could be used too
            else:
                logging.error("Max retries reached. LLM call failed.")
                return f"Error: LLM call failed after {max_retries} attempts. Last error: {e}"
        except Exception as e: # Catch any other unexpected errors
            logging.error(f"An unexpected error occurred during LLM call: {e}")
            return f"Error: An unexpected error occurred: {e}"
    return f"Error: LLM call failed after {max_retries} attempts." # Should be unreachable if loop completes


def main():
    """Main function to drive the script."""
    validate_config()
    api_key_value = get_api_key_from_env_or_config()
    openai_client = initialize_openai_client(api_key_value)

    df_input = load_and_validate_input_df(INPUT_FILE)

    # Combine general and detailed aspects for iteration
    all_analysis_types_to_perform = GENERAL_ANALYSIS_TYPES + DETAILED_SERVICE_ASPECTS
    if not all_analysis_types_to_perform:
        logging.info("No analysis types specified (GENERAL_ANALYSIS_TYPES and DETAILED_SERVICE_ASPECTS are empty). Exiting.")
        return

    output_data = []
    processed_combinations = 0 # For tqdm progress

    # Calculate total iterations for tqdm
    # Group by Year and Product first
    grouped_by_year_product = df_input.groupby(['Year', 'Product'])
    total_iterations = len(grouped_by_year_product) * len(all_analysis_types_to_perform)

    logging.info(f"Starting analysis generation for {len(grouped_by_year_product)} Product/Year combinations "
                f"and {len(all_analysis_types_to_perform)} analysis types each.")

    with tqdm(total=total_iterations, desc="Generating Analyses") as pbar:
        # Iterate through each unique (Year, Product)
        for (year, product_name), group_df in grouped_by_year_product:
            # For each product, get the most recent reviews up to MAX_REVIEWS_FOR_SYNTHESIS
            # Sort by date within the group to get the most recent ones
            # Ensuring 'Date' is datetime before sorting
            group_df['Date'] = pd.to_datetime(group_df['Date'])
            recent_reviews_df = group_df.sort_values(by='Date', ascending=False).head(MAX_REVIEWS_FOR_SYNTHESIS)
            reviews_list = recent_reviews_df['Review'].tolist()

            if not reviews_list:
                logging.warning(f"No reviews found for Product: '{product_name}', Year: {year}. Skipping analysis for this group.")
                # We still need to advance pbar for the analysis types that would have run
                for _ in all_analysis_types_to_perform:
                    pbar.update(1)
                continue

            reviews_text_for_prompt = format_reviews_for_prompt(reviews_list)

            # For this product, iterate through each type of analysis we want to perform
            for analysis_type in all_analysis_types_to_perform:
                pbar.set_description(f"Product: {product_name[:20]}..., Analysis: {analysis_type}")

                # Construct the system prompt
                system_prompt = construct_llm_prompt(
                    COMPANY_NAME,
                    product_name,
                    analysis_type,
                    reviews_text_for_prompt, # Pass the actual review text here
                    ASPECT_STYLES_EXAMPLES_FOR_PROMPT
                )
                # User content can be minimal if most info is in system prompt
                user_content_for_llm = "Please generate the analysis based on the reviews and instructions provided in the system message."

                generated_analysis = call_llm_with_retry(
                    openai_client,
                    MODEL,
                    system_prompt,
                    user_content_for_llm
                )

                output_data.append({
                    'Year': year,
                    'Product': product_name,
                    'Aspect': analysis_type, # This 'Aspect' column now means the type of analysis
                    'Analysis': generated_analysis
                })
                pbar.update(1)

    if not output_data:
        logging.info("No analysis data was generated. Output file will not be created.")
        return

    # Create DataFrame from the collected data and save to CSV
    df_output = pd.DataFrame(output_data)
    try:
        df_output.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
        logging.info(f"Successfully saved analysis to '{OUTPUT_FILE}'")
    except Exception as e:
        logging.error(f"Error saving output CSV to '{OUTPUT_FILE}': {e}")

main()