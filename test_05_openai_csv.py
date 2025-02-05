from openai import OpenAI
import os
import pandas as pd
import time
from datetime import timedelta
from dotenv import load_dotenv

# Load environment variables from a .env file
#load_dotenv()

# Fetch the OpenAI API key from environment variables
#API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI()

# Define the aspects to analyze (update this list for new aspects)
#ASPECTS_TO_ANALYSE = ["Engineer Professionalism", "Response Speed", "Customer Support", "Solution Quality", "Communication Clarity", "Value For Money"]
#ASPECTS_TO_ANALYSE = ["Product Categorisation"]
#ASPECTS_TO_ANALYSE = ["Overall Sentiment", "Service Quality", "Response Speed", "Solution Quality", "Company Reliability", "Ease of Booking", "Service Delivery Issues", "Product Coverage", "Communication Failures", "Financial Concerns"]
ASPECTS_TO_ANALYSE = ["Overall Sentiment", "Appointment Scheduling", "Customer Service", "Response Speed", "Engineer Experience", "Solution Quality", "Value for Money"]

# Specify input and output file paths
#input_file = "HomeServe Raw Data Feb24 Onwards.csv" #"BG_20pct_Reviews_March_Onwards.csv"  # Replace with your input file path
input_file = "CheckATrade Reviews of Relevant Prods 15k.csv"
output_file = "CheckATrade 15k Output v01.csv"  # Replace with your output file path

def analyze_aspect_sentiment(review, aspect):
    """
    Analyze sentiment for a specific aspect of a review and return a score between -100 and 100 with a reason.

    :param review: The customer's review.
    :param aspect: The aspect to analyze (e.g., "value for money").
    :return: A tuple (sentiment score, explanation).
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            # messages=[
            #     {"role": "system", "content": (
            #         "You are a product analysis expert. Your task is to assess customer reviews for HomeServe "
            #         "and determine the most appropriate product category from the following five options: \n"
            #         "1. Gas Products: Covers boiler insurance, hot water issues, gas supply issues, central heating, or any instances where engineers visit customers' homes to service, install, or fix boilers or fix the hot water not running.\n"
            #         "2. Plumbing and Drains: Insurance cover for pipes, drains, toilets and tanks, including things like blocked drains, frozen pipes, or other plumbing repairs.\n"
            #         "3. Home Electrical: Insurance for home electrics, including cover for wiring, fusebox breakdowns and broken sockets.\n"
            #         "4. Unknown: Please use this only if there is absolutely no information to categorize the review, after making every effort to find relevant clues.\n\n"
            #         "### Additional Guidance:\n"
            #         "- Pleease look for specific words, phrases, or scenarios that suggest the correct category (e.g., 'boiler' → Gas Products).\n"
            #         "- If multiple product types are mentioned, select the one most emphasized or central to the review.\n"
            #         "- Please include a confidence score between 0 and 100 as to how certain you are that you have classified the product type correctly.\n"
            #         "- Avoid marking as 'Unknown' unless all other attempts to categorize fail.\n\n"
            #         "Provide the output in this exact format:\n"
            #         "Score: <numeric_value>\n"
            #         "Product: <product_type>\n"
            #         "Reason: <brief_explanation>\n\n"
            #     )},
            #     {"role": "user",
            #      "content": f"Review: {review}\nQuestion: What is the product type, confidence score, and reason for your categorization?"}
            # ],
            # messages=[
            #     {"role": "system", "content": (
            #         "You are a product analysis expert. Your task is to assess customer reviews for British Gas "
            #         "and determine the most appropriate product category from the following five options: \n"
            #         "1. Gas Products: Covers boiler insurance, home care, annual service visits, or any instances where engineers visit customers' homes to service, install, or fix boilers or fix the hot water not running.\n"
            #         "2. Energy: Relates to British Gas as an energy / electricity supplier, or gas supply services, including tariffs, smart meters, and energy bills including charges and billing issues for unfixed tariffs. This category does not involve engineers visiting homes.\n"
            #         "3. Appliance Cover: Includes insurance for home appliances like ovens, washing machines, or any electrical appliances that we repair if they break down.\n"
            #         "4. Plumbing and Drains: Insurance for issues such as blocked drains, frozen pipes, or plumbing repairs, often handled by DynoRod or similar partners.\n"
            #         "5. Unknown: Please use this only if there is absolutely no information to categorize the review, after making every effort to find relevant clues.\n\n"
            #         "### Additional Guidance:\n"
            #         "- Pleease look for specific words, phrases, or scenarios that suggest the correct category (e.g., 'boiler' → Gas Products, 'billing' → Energy).\n"
            #         "- If multiple product types are mentioned, select the one most emphasized or central to the review.\n"
            #         "- Please include a confidence score between 0 and 100 as to how certain you are that you have classified the product type correctly.\n"
            #         "- Avoid marking as 'Unknown' unless all other attempts to categorize fail.\n\n"
            #         "Provide the output in this exact format:\n"
            #         "Score: <numeric_value>\n"
            #         "Product: <product_type>\n"
            #         "Reason: <brief_explanation>\n\n"
            #     )},
            #     {"role": "user",
            #      "content": f"Review: {review}\nQuestion: What is the product type, confidence score, and reason for your categorization?"}
            # ],
            messages=[
                {"role": "system", "content": (
                    "You are a sentiment analysis expert. Your task is to assess customer reviews for these CheckATrade local contractors "
                    "based on specific aspects. For each review, provide a sentiment score "
                    "between -100 and 100, where:\n"
                    "-100 = Extremely negative sentiment.\n"
                    "0 = Neutral sentiment.\n"
                    "100 = Extremely positive sentiment.\n"
                    "Focus only on the specified aspect and provide a brief explanation for your score.\n"
                    "Provide the output in the following format:\n"
                    "Score: < numeric_value >\n"
                    "Reason: < brief_explanation >\n"
                )},
                {"role": "user",
                 "content": f"Review: {review}\nQuestion: What is the sentiment score and reason for '{aspect}'?"}
            ],
            temperature=0.2,
            max_tokens=150
        )

        # Parse the response content
        content = response.choices[0].message.content.strip()

        # Ensure the response format matches expectations
        if "Score:" not in content or "Reason:" not in content:
            return None, f"Unexpected response format: {content}"

        # Extract score and reason
        parts = content.split("Reason:", maxsplit=1)
        #prod_split = parts[0].split("Product:", maxsplit=1)
        #score_str = prod_split[0].strip().replace("Score:", "").strip()
        score_str = parts[0].strip().replace("Score:", "").strip()
        reason = parts[1].strip() if len(parts) > 1 else "No explanation provided"
        #prod_type = prod_split[1].strip() if len(prod_split) > 1 else "No product type provided"
        score = int(score_str)
        #return score, prod_type, reason
        return score, reason

    except Exception as e:
        return None, f"Error: {str(e)}"


def format_time(seconds):
    """
    Convert seconds into a formatted hh:mm:ss string.

    :param seconds: Time in seconds.
    :return: Time formatted as hh:mm:ss.
    """
    return str(timedelta(seconds=int(seconds)))


def initialize_output_file(output_csv, aspects):
    """
    Initialize the output CSV file with the appropriate headers.

    :param output_csv: Path to the output CSV file.
    :param aspects: A list of aspects to analyze.
    """
    #columns = ["Date", "Year-Month", "Final Product Category", "Review"]  # Start with Date and Review columns
    columns = ["Date", "Year-Month", "Review"]  # Start with Date and Review columns
    for aspect in aspects:
        columns.append(f"{aspect}_sentiment_score")
        #columns.append(f"{aspect}_prod_type")
        columns.append(f"{aspect}_reason")

    # Create an empty DataFrame with the desired columns and save it to the output file
    pd.DataFrame(columns=columns).to_csv(output_csv, index=False)


def append_to_output_file(output_csv, row_data):
    """
    Append a row of data to the output CSV file.

    :param output_csv: Path to the output CSV file.
    :param row_data: Dictionary containing the row data to append.
    """
    pd.DataFrame([row_data]).to_csv(output_csv, mode='a', header=False, index=False)


def process_reviews(input_csv, output_csv, aspects):
    """
    Process a CSV of reviews to analyze sentiment for specified aspects and save results to a new CSV.

    :param input_csv: Path to the input CSV file with a 'reviews' column.
    :param output_csv: Path to save the output CSV file with sentiment analysis results.
    :param aspects: A list of aspects to analyze (e.g., ["value for money", "customer service"]).
    """

    # Load the input CSV file
    df = pd.read_csv(input_csv)

    # Ensure the 'Date' column is in datetime format
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

    if 'Year-Month' not in df.columns:
        df['Year-Month'] = df['Date'].apply(lambda x: x.replace(day=1))

    # Ensure the 'reviews' and 'Date' columns exist
    if 'Review' not in df.columns or 'Date' not in df.columns:
        raise ValueError("Input CSV must contain 'Review' and 'Date' columns.")

    # Add columns for new aspects if they don't exist
    for aspect in aspects:
        if f"{aspect}_sentiment_score" not in df.columns:
            df[f"{aspect}_sentiment_score"] = None
        # if f"{aspect}_type" not in df.columns:
        #     df[f"{aspect}_type"] = None
        if f"{aspect}_reason" not in df.columns:
            df[f"{aspect}_reason"] = None

    # Initialize the output CSV file
    initialize_output_file(output_csv, aspects)

    print("Starting sentiment analysis...")

    # Track the total start time
    total_start_time = time.time()

    unknown_count = 0

    # Check how many rows have already been processed
    try:
        processed_df = pd.read_csv(output_csv)
        processed_count = len(processed_df)
    except FileNotFoundError:
        processed_count = 0

    print(f"Resuming from review {processed_count + 1}...")

    # Process each review
    for index, row in df.iloc[processed_count:].iterrows():
        review = row['Review']
        date = row['Date']
        yearMonth = row['Year-Month']
        #prod = row['Final Product Category']

        if pd.isna(review) or not isinstance(review, str):
            # Skip invalid or missing reviews
            #row_data = {"Date": date, "Year-Month": yearMonth, "Final Product Category": prod, "Review": review}
            row_data = {"Date": date, "Year-Month": yearMonth, "Review": review}
            for aspect in aspects:
                row_data[f"{aspect}_sentiment_score"] = None
                #row_data[f"{aspect}_type"] = "No product provided"
                row_data[f"{aspect}_reason"] = "No review provided"
                df.at[index, f"{aspect}_sentiment_score"] = None
                #df.at[index, f"{aspect}_type"] = "No product provided"
                df.at[index, f"{aspect}_reason"] = "No review provided"
            append_to_output_file(output_csv, row_data)
            continue

        # Prepare row data
        #row_data = {"Date": date, "Year-Month": yearMonth, "Final Product Category": prod, "Review": review}
        row_data = {"Date": date, "Year-Month": yearMonth, "Review": review}

        # Analyze sentiment for each aspect
        for aspect in aspects:
            try:
                score, reason = analyze_aspect_sentiment(review, aspect)
            except ValueError as ve:
                # Handle unpacking or formatting errors
                score, reason = 0, "Error", f"Error ValueError: {str(ve)}"
            except Exception as e:
                # Catch any other errors
                score, reason = 0, "Error", f"Error Exception: {str(e)}"
            row_data[f"{aspect}_sentiment_score"] = score
            #row_data[f"{aspect}_type"] = prod
            row_data[f"{aspect}_reason"] = reason
            df.at[index, f"{aspect}_sentiment_score"] = score  # Update DataFrame
            #df.at[index, f"{aspect}_type"] = prod  # Update DataFrame
            df.at[index, f"{aspect}_reason"] = reason  # Update DataFrame
            #print(f"Prod: {prod}, Score: {score}, Reason: {reason}")
            print(f"Score: {score}, Reason: {reason}")

        # Append the result to the output file
        append_to_output_file(output_csv, row_data)

        # Calculate elapsed time and estimate remaining time
        total_elapsed_time = time.time() - total_start_time
        reviews_processed = index + 1
        reviews_remaining = len(df) - reviews_processed
        avg_time_per_review = total_elapsed_time / reviews_processed
        estimated_time_remaining = avg_time_per_review * reviews_remaining
        # if prod == "Unknown":
        #     unknown_count = unknown_count + 1

        # Print progress
        # print(f"Review {reviews_processed}/{len(df)} - Elapsed: {format_time(total_elapsed_time)} - Remaining: {format_time(estimated_time_remaining)}, Unknown {unknown_count}, {unknown_count / reviews_processed:.1%}")
        print(f"Review {reviews_processed}/{len(df)} - Elapsed: {format_time(total_elapsed_time)} - Remaining: {format_time(estimated_time_remaining)}")
        print(f"Review: {review}")
        print("--------------------------------------")
    print(f"Sentiment analysis completed. Results saved to {output_csv}.")
    print(f"Total time taken: {format_time(time.time() - total_start_time)}")

    # Group by 'Year-Month' and 'Final Product Category' and calculate average sentiment scores
    print("Columns in the DataFrame:", df.columns.tolist())
    #aggregation_columns = {f"{aspect}_sentiment_score": 'mean' for aspect in aspects}
    #aggregated_df = df.groupby(['Year-Month', 'Final Product Category'], as_index=False).agg(aggregation_columns)

    # Save the aggregated data to a new CSV
    #aggregated_output_csv = output_file.replace(".csv","_agg.csv")
    #aggregated_df.to_csv(aggregated_output_csv, index=False)
    #print(f"Aggregated average scores saved to {aggregated_output_csv}")

# Run the analysis
process_reviews(input_file, output_file, ASPECTS_TO_ANALYSE)

# # Example review and aspect
# example_review = "The repair was quick and affordable."
# example_aspect = "value for money"
#
# # Test the function with the first review
# score, reason = analyse_aspect_sentiment(example_review, example_aspect)
#
# # Print the result to the console
# print(f"Review: {example_review}")
# print(f"Aspect: {example_aspect}")
# print(f"Sentiment Score: {score}")
# print(f"Reason: {reason}")

