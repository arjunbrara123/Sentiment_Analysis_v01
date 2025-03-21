"""
Chatbot Module (chatbot.py)
================================
This module handles the interaction with the OpenAI API to generate data-driven chatbot responses based on review data.
It processes review data to create context and then uses that context along with a user query to obtain insights from the AI.

Function Overview:
------------------
1. sample_reviews: Samples reviews proportionally from the provided review data for a specific company, product and year.
2. prepare_context: Concatenates the selected company's reviews with British Gas reviews (if available) into a single context string for the chatbot.
3. generate_response: Sends the prepared context and user query to the OpenAI API to return a chatbot response

How It Works:
-------------
- The module first filters and samples reviews to ensure a representative dataset.
- It then builds a comprehensive context by merging different review sources.
- Finally, it constructs a system prompt and communicates with the OpenAI API to provide concise, data-driven insights.
"""

# Import Required Modules
import pandas as pd
from openai import OpenAI

# ------------------------------------------------------------------------------
# OpenAI API Configuration Parameters:
# ------------------------------------------------------------------------------
MODEL = "gpt-4o-mini"       #   - The model to use for generating responses.
TEMPERATURE = 0.3           #   - Controls the randomness in the output. Lower values yield more deterministic responses.
MAX_TOKENS = 1000           #   - Maximum number of tokens for the generated response.
FREQUENCY_PENALTY = 1.5     #   - Reduces repetition by penalising frequent words in the output.
PRESENCE_PENALTY = 1.5      #   - Encourages the introduction of new topics by penalising words already present in the prompt.

# Base system prompt used for generating chatbot responses.
BASE_SYSTEM_PROMPT = (
    "You are a commercial strategy expert at British Gas Insurance. "
    "Your task is to analyse the provided social media data and reviews to provide well-reasoned, data-driven insights. "
    "Please provide your analysis in a single paragraph in a concise manner."
)

# ------------------------------------------------------------------------------
# AI Chatbot Utility Functions:
# ------------------------------------------------------------------------------

def sample_reviews(reviews_data: pd.DataFrame, product_name: str, filter_year: str, review_limit: int = 50,
                   random_state: int = 42) -> pd.DataFrame:
    # Samples reviews proportionally from each month for the given product and year.

    filtered_reviews = reviews_data[reviews_data["Final Product Category"] == product_name]
    if filter_year != "All":
        filtered_reviews = filtered_reviews[filtered_reviews["Year-Month"].str[-4:] == str(filter_year)]

    monthly_counts = filtered_reviews["Year-Month"].value_counts()
    sample_sizes = (monthly_counts / monthly_counts.sum() * review_limit).astype(int)

    sampled_reviews = pd.concat([
        filtered_reviews[filtered_reviews["Year-Month"] == month].sample(
            n=min(sample_sizes[month], len(filtered_reviews[filtered_reviews["Year-Month"] == month])),
            random_state=random_state
        )
        for month in monthly_counts.index if month in sample_sizes.index
    ])

    return sampled_reviews


def prepare_context(selected_reviews: pd.DataFrame, bg_reviews: pd.DataFrame, selected_company: str) -> str:
    # Prepares context for the chatbot by concatenating the selected company's reviews and BG reviews (if provided).

    context = f"Selected Company ({selected_company}) Reviews:\n{selected_reviews.to_string(index=False)}\n\n"
    if bg_reviews is not None:
        context += f"British Gas Reviews:\n{bg_reviews.to_string(index=False)}\n\n"
    return context


def generate_response(context: str, query: str, product: str, selected_company: str, bg_reviews) -> str:
    # Calls the OpenAI API to generate a chatbot response based on the provided context and query.

    client = OpenAI()  # Assumes API key is set in the environment
    system_prompt = BASE_SYSTEM_PROMPT
    if bg_reviews is not None:
        system_prompt += f"\n\nYou are comparing British Gas with {selected_company} for the {product} product line. Provide insights based on the comparison of their reviews."
    else:
        system_prompt += f"\n\nYou are analyzing reviews for British Gas's {product} product line."

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Social Media Data: {context}\n\nQuestion: {query}"}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            frequency_penalty=FREQUENCY_PENALTY,
            presence_penalty=PRESENCE_PENALTY
        )
        answer = response.choices[0].message.content.strip()
        return answer
    except Exception as e:
        return f"Oops, something went wrong! Error: {str(e)}"
