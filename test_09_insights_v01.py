import os
from datetime import timedelta
import pandas as pd
import tiktoken
import time
import asyncio
import aiohttp
from openai import OpenAI
from dotenv import load_dotenv

# ------------------- CONFIGURATION -------------------
load_dotenv()
client = OpenAI()

SENTIMENT_DATA_FILE = "LLM_SA_Monthly_Data.csv"
REVIEWS_DATA_FILE = "reviews_all.csv"
OUTPUT_FILE = "review_analysis_output.csv"
DEBUG_FILE = "debug_product_company_combinations.csv"
MODEL = "gpt-4o-mini"
MAX_TOKENS_PER_CHUNK = 16000
WAIT_BETWEEN_CALLS = 1
YEAR = 2024
QUALITY_SCORE_THRESHOLD = 0.89

ASPECTS = ["Appointment Scheduling", "Customer Service", "Response Speed",
           "Engineer Experience", "Solution Quality", "Value For Money"]

# ------------------- HELPER FUNCTIONS -------------------
def count_tokens(text, model=MODEL):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

async def analyze_comparison(session, competitor, category, aspect, year, sentiment_diff, reviews):
    """Prompt 1: Compare British Gas to competitor."""
    messages = [
        {"role": "system", "content": (
            f"You are an expert competitor analyst writing for senior board members. "
            f"Compare British Gas to {competitor} in the product category '{category}' for the aspect '{aspect}' in {year}. "
            f"The sentiment difference is {sentiment_diff:.2f} (positive means British Gas is better). "
            f"Your response should be formatted with clear, separate paragraphs. "
            f"Begin with one or two concise paragraphs that analyze the reviews in a data-driven way—explain which company performs better and describe the key trend(s) driving this difference. "
            f"Support your analysis with 1-2 brief, specific review examples (either paraphrased or quoted) that best illustrate this trend. "
            f"Finally, on a new line and after an empty line, add a concluding paragraph that starts with a bold markdown header '**Key Takeaway:**' followed by one or two very specific, actionable insights for British Gas. "
            f"Keep the language focused and avoid vagueness. "
            f"Example:\n\n"
            f"CheckATrade customers report faster response times than British Gas customers. Since independent contractors manage their own schedules, many can offer same-day or next-day service—especially for urgent boiler repairs. Reviews indicate that some CheckATrade engineers prioritize emergency cases more efficiently than a large organization can.\n\n"
            f"British Gas customers, however, often mention delays in dispatching engineers, particularly for non-urgent issues. The centralized dispatch system, while effective for structured planning, can struggle with last-minute requests and high seasonal demand.\n\n"
            f"**Key Takeaway:** British Gas should explore faster emergency response options, including same-day service slots for urgent cases and real-time tracking updates for customers waiting for an engineer."
        )},
        {"role": "user", "content": f"Analyze these reviews:\n\n{reviews}"}
    ]
    async with session.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}", "Content-Type": "application/json"},
            json={"model": MODEL, "messages": messages, "max_tokens": 300}  # Tight limit for conciseness
    ) as response:
        result = await response.json()
        return result["choices"][0]["message"]["content"]

async def analyze_insights(session, competitor, category, aspect, year, sentiment_diff, reviews):
    """Prompt 2: Extract actionable insights for British Gas."""
    messages = [
        {"role": "system", "content": (
            f"You are an expert competitor analyst writing for senior board members. "
            f"Based on the review analysis provided, generate a list of the biggest and most important actionable business insights in a concise numbered list which British Gas can do better based on the differences you can learn from the difference in reviews. "
            f"Only look for the big important ones and do not list any small ones which are not very data driven from the reviews or would not have much of an overall business impact. "
            f"Support your analysis with 1-2 brief, specific review examples (either paraphrased or quoted) that best illustrate this trend. Anything which cannot be backed up by data should not be listed. "
            f"Each insight should be brief and formatted as follows:\n\n"
            f"1. **[Insight Title]:** [Short actionable recommendation].\n\n"
            f"For example:\n\n"
            f"1. **Faster Responses:** British Gas should explore faster emergency response options, including same-day service slots for urgent cases and real-time tracking updates for customers waiting for an engineer.\n\n"
            f"Keep your language direct and focused, avoiding unnecessary details since these insights are for busy senior board members."
        )},
        {"role": "user",
         "content": f"Based on the following review analysis, list actionable business insights:\n\n{reviews}"}
    ]
    async with session.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}", "Content-Type": "application/json"},
            json={"model": MODEL, "messages": messages, "max_tokens": 500}  # Tight limit for conciseness
    ) as response:
        result = await response.json()
        return result["choices"][0]["message"]["content"]

# ------------------- MAIN PROCESS -------------------
async def main():
    total_start_time = time.time()

    # Load and filter data
    sentiment_df = pd.read_csv(SENTIMENT_DATA_FILE)
    sentiment_df['Year-Month'] = pd.to_datetime(sentiment_df['Year-Month'], format='%d/%m/%Y', errors='raise')
    sentiment_df['Year'] = sentiment_df['Year-Month'].dt.year
    sentiment_df = sentiment_df[sentiment_df["Year"] == YEAR]

    reviews_df = pd.read_csv(REVIEWS_DATA_FILE)
    reviews_df = reviews_df[reviews_df["Year"] == YEAR]

    # Step 1: Unique combinations
    competitors = [c for c in sentiment_df["Company"].unique() if c != "British Gas"]
    categories = sentiment_df["Final Product Category"].unique()
    combinations = [(comp, cat, asp) for comp in competitors for cat in categories for asp in ASPECTS]
    print(f"Total unique combinations: {len(combinations)}")
    print(f"Sample combinations (Competitor | Product Category | Aspect):")
    for i, (comp, cat, asp) in enumerate(combinations[:5], 1):
        print(f"{i}. {comp} | {cat} | {asp}")
    if len(combinations) > 5:
        print("... (and more)")

    # Debugging: Product-Company combinations
    debug_data = []
    for category in categories:
        for company in ["British Gas"] + competitors:
            total_reviews = len(reviews_df[(reviews_df["Company"] == company) &
                                          (reviews_df["Final Product Category"] == category)])
            quality_reviews = len(reviews_df[(reviews_df["Company"] == company) &
                                            (reviews_df["Final Product Category"] == category) &
                                            (reviews_df["review_weight"] >= QUALITY_SCORE_THRESHOLD)])
            tokens_per_review = 100
            max_reviews_in_limit = (MAX_TOKENS_PER_CHUNK // 2) // tokens_per_review
            debug_data.append({
                "Company": company,
                "Product Category": category,
                "Total Reviews": total_reviews,
                "Reviews Above Threshold": quality_reviews,
                "Max Reviews in Token Limit": min(quality_reviews, max_reviews_in_limit)
            })
    debug_df = pd.DataFrame(debug_data)
    debug_df.to_csv(DEBUG_FILE, index=False)
    print(f"\nDebug data saved to {DEBUG_FILE}. Sample:")
    print(debug_df.head().to_string(index=False))

    # Step 2: Process reviews
    output = []
    summary_table = []

    async with aiohttp.ClientSession() as session:
        tasks = []
        for competitor, category, aspect in combinations:
            # Sentiment data
            try:
                bg_sentiment = sentiment_df[(sentiment_df["Company"] == "British Gas") &
                                            (sentiment_df["Final Product Category"] == category)].iloc[-1]
                comp_sentiment = sentiment_df[(sentiment_df["Company"] == competitor) &
                                             (sentiment_df["Final Product Category"] == category)].iloc[-1]
                sentiment_diff = round(bg_sentiment[f"{aspect}_sentiment_score"] -
                                      comp_sentiment[f"{aspect}_sentiment_score"], 2)
            except (IndexError, KeyError) as e:
                print(f"Skipping {competitor} | {category} | {aspect} - missing sentiment data: {e}")
                continue

            # Filter reviews
            bg_reviews = reviews_df[(reviews_df["Company"] == "British Gas") &
                                   (reviews_df["Final Product Category"] == category) &
                                   (reviews_df["review_weight"] >= QUALITY_SCORE_THRESHOLD)]
            comp_reviews = reviews_df[(reviews_df["Company"] == competitor) &
                                     (reviews_df["Final Product Category"] == category) &
                                     (reviews_df["review_weight"] >= QUALITY_SCORE_THRESHOLD)]

            if bg_reviews.empty or comp_reviews.empty:
                print(f"Skipping {competitor} | {category} | {aspect} - insufficient reviews.")
                continue

            # Balance and fit within token limit
            max_reviews_per_company = min(len(bg_reviews), len(comp_reviews))
            tokens_per_review = 100
            max_possible_reviews = (MAX_TOKENS_PER_CHUNK // 2) // tokens_per_review
            num_reviews = min(max_reviews_per_company, max_possible_reviews)

            bg_sample = bg_reviews.nlargest(num_reviews, "review_weight")["Review"].tolist()
            comp_sample = comp_reviews.nlargest(num_reviews, "review_weight")["Review"].tolist()

            bg_text = " ".join(bg_sample)
            comp_text = " ".join(comp_sample)
            combined_text = (f"British Gas Reviews ({num_reviews}):\n{bg_text}\n\n"
                            f"{competitor} Reviews ({num_reviews}):\n{comp_text}")

            total_tokens = count_tokens(combined_text)
            if total_tokens > MAX_TOKENS_PER_CHUNK:
                print(f"Warning: {competitor} | {category} | {aspect} exceeds token limit ({total_tokens} > {MAX_TOKENS_PER_CHUNK}). Truncating.")
                combined_text = combined_text[:MAX_TOKENS_PER_CHUNK * 4]

            summary_table.append({
                "Competitor": competitor,
                "Product Category": category,
                "Aspect": aspect,
                "British Gas Reviews": num_reviews,
                "Competitor Reviews": num_reviews,
                "Total Tokens": min(total_tokens, MAX_TOKENS_PER_CHUNK)
            })

            # Two separate tasks for each prompt
            metadata = {"Company": competitor, "Product": category, "Aspect": aspect,
                        "Sentiment Difference": sentiment_diff, "Year": YEAR}
            task_comparison = analyze_comparison(session, competitor, category, aspect, YEAR, sentiment_diff, combined_text)
            task_insights = analyze_insights(session, competitor, category, aspect, YEAR, sentiment_diff, combined_text)
            tasks.append((metadata, task_comparison, task_insights))
            await asyncio.sleep(WAIT_BETWEEN_CALLS)

        print("\nReview Summary Table:")
        print(pd.DataFrame(summary_table).to_string(index=False))

        # Run analyses
        responses = await asyncio.gather(*(task for _, task_comp, task_ins in tasks for task in [task_comp, task_ins]))
        for i, (metadata, _, _) in enumerate(tasks):
            analysis = responses[i * 2]  # Even indices: comparison
            insights = responses[i * 2 + 1]  # Odd indices: insights
            output.append({**metadata, "Analysis": analysis.strip(), "Breakdown": insights.strip()})

    # Save results
    output_df = pd.DataFrame(output)
    output_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nAnalysis completed. Results saved to {OUTPUT_FILE}")
    print(f"Total time taken: {format_time(time.time() - total_start_time)}")

def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))

# ------------------- RUN SCRIPT -------------------
asyncio.run(main())