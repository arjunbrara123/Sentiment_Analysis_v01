import streamlit as st
import pandas as pd
import os
import time
import datetime
import re
import asyncio
import aiohttp
import tiktoken
from openai import OpenAI
from transformers import pipeline
#from dotenv import load_dotenv

# ---------------------------------------------------
# Page Configuration & Environment Setup
# ---------------------------------------------------
#st.set_page_config(layout="wide", page_title="Review Analysis Dashboard")
#load_dotenv()
#API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI()

# ---------------------------------------------------
# Load and inject CSS
# ---------------------------------------------------
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---------------------------------------------------
# Prompts and Default Data
# ---------------------------------------------------
DEFAULT_SENTIMENT_PROMPT = (
    "You are a sentiment analysis expert. Your task is to assess customer reviews based on a specified aspect. "
    "For each review, provide a sentiment score between -100 and 100, where -100 means extremely negative and 100 means extremely positive. "
    "Focus solely on the given aspect and provide a brief explanation. "
    "Provide the output in the following format:\n"
    "Score: <numeric_value>\n"
    "Reason: <brief_explanation>"
)

PRODUCT_DESCRIPTIONS = {
    "Gas Products": "Covers boiler insurance, home care, annual service visits, or any instances where engineers visit to service or repair boilers or hot water systems.",
    "Energy": "Relates to energy/electricity supply including tariffs, smart meters, and billing issues.",
    "Appliance Cover": "Insurance for home appliances like ovens, washing machines, televisions, laptops, computers, etc.",
    "Home Electrical": "Covers issues with home electrics such as, wiring, fusebox breakdowns, and broken sockets.",
    "Plumbing and Drains": "Covers plumbing repairs such as blocked drains or frozen pipes.",
    "Building": "General building services.",
    "Pest Control": "Removal of a pest infestation in the home, eg. Wasps and hornets nests, mice or rat infestation."
}

COMPANY_PRODUCTS = {
    "British Gas": ["Gas Products", "Energy", "Appliance Cover", "Home Electrical", "Plumbing and Drains"],
    "HomeServe": ["Gas Products", "Plumbing and Drains", "Appliance Cover", "Home Electrical"],
    "CheckATrade": ["Gas Products", "Plumbing and Drains", "Home Electrical", "Building"],
    "Corgi HomePlan": ["Gas Products", "Plumbing and Drains", "Home Electrical", "Building"],
    "Domestic & General": ["Gas Products", "Plumbing and Drains", "Appliance Cover", "Home Electrical"],
    "247 Home Rescue": ["Gas Products", "Plumbing and Drains", "Appliance Cover", "Home Electrical", "Pest Control"]
}

DEFAULT_COMPARISON_PROMPT = (
    "You are an expert competitor analyst writing for senior board members. "
    "Compare British Gas to {competitor} in the product category '{category}' for the aspect '{aspect}' in {year}. "
    "The sentiment difference is {sentiment_diff:.2f} (positive means British Gas is better). "
    "In one concise paragraph, state which company performs better, explain the key trend driving this based on the reviews trying to be as specific as possible, "
    "and include 1-2 brief, specific review examples (paraphrased or quoted) that best show this trend. "
    "Keep it focused, data-driven, and avoid waffle or vagueness. "
    "Example: 'British Gas outperforms HomeServe by 0.34 in Engineer Experience. "
    "Reviews highlight BG’s better skilled engineers as the primary driver for this big sentiment difference, e.g., ‘Fixed my boiler in one visit,’ while HomeServe’s are inconsistent, "
    "e.g., ‘Engineer left it worse.’'"
)

DEFAULT_INSIGHTS_PROMPT = (
    "You are an expert competitor analyst writing for senior board members. "
    "Based on reviews for British Gas and {competitor} in the product category '{category}' for the aspect '{aspect}' in {year}, "
    "where the sentiment difference is {sentiment_diff:.2f} (positive means British Gas is better), "
    "provide one concise paragraph with 1-2 actionable insights British Gas can use to improve. "
    "Focus on the biggest trend or difference from the data, make it clear every insight is grounded specifically in the reviews, "
    "and include 1 brief, specific review example (paraphrased or quoted) per insight to build confidence. "
    "Keep it focused, actionable, and avoid minor details or any vagueness. "
    "Example: 'British Gas should ensure first-visit fixes, as HomeServe’s ‘3 visits to fix’ shows a gap we can avoid and seems to be a big theme driving the majority of the sentiment difference here. "
    "Also, the data suggests maintain training—BG’s ‘knowledgeable staff’ is a strength driving a lot of the positive sentiment in the data, therefore something should double down on to further extend and improve this positive sentiment driver.'"
)

def generate_prod_prompt(company, selected_products_dict):
    prompt = f"You are a product analysis expert. Your task is to assess customer reviews for {company} and determine the most appropriate product category from the following options:\n"
    i = 1
    for prod, desc in selected_products_dict.items():
        prompt += f"{i}. {prod}: {desc}\n"
        i += 1
    prompt += "\n### Guidance:\n"
    prompt += "- Look for keywords or scenarios that hint at the correct category.\n"
    prompt += "- If you are unsure, mark the product type as 'Unknown', but make every effort to try and allocate one of the given product categories and only mark as Unknown if you have to.\n"
    prompt += "- Include a confidence score between 0 and 100.\n"
    prompt += "Provide the output in this exact format:\n"
    prompt += "Score: <numeric_value>\nProduct: <product_type>\nReason: <brief_explanation>"
    return prompt


# ---------------------------------------------------
# API Call Functions
# ---------------------------------------------------
def analyze_product_categorisation(review, prompt_template, model, temperature, max_tokens):
    try:
        messages = [
            {"role": "system", "content": prompt_template},
            {"role": "user",
             "content": f"Review: {review}\nQuestion: What is the product type, confidence score, and reason for your categorization?"}
        ]
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        content = response.choices[0].message.content.strip()
        if "Score:" not in content or "Reason:" not in content:
            return None, None, f"Unexpected response format: {content}"
        parts = content.split("Reason:", maxsplit=1)
        if "Product:" in parts[0]:
            score_part, product_part = parts[0].split("Product:", maxsplit=1)
            score_str = score_part.replace("Score:", "").strip()
            product = product_part.strip()
        else:
            score_str = parts[0].replace("Score:", "").strip()
            product = "Not provided"
        reason = parts[1].strip()
        score = int(score_str)
        return score, product, reason
    except Exception as e:
        return None, None, f"Error: {str(e)}"


def analyze_sentiment(review, aspect, prompt_template, model, temperature, max_tokens):
    try:
        messages = [
            {"role": "system", "content": prompt_template},
            {"role": "user",
             "content": f"Review: {review}\nQuestion: What is the sentiment score and reason for '{aspect}'?"}
        ]
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        content = response.choices[0].message.content.strip()
        if "Score:" not in content or "Reason:" not in content:
            return None, f"Unexpected response format: {content}"
        parts = content.split("Reason:", maxsplit=1)
        score_str = parts[0].replace("Score:", "").strip()
        reason = parts[1].strip() if len(parts) > 1 else "No explanation provided"
        score = int(score_str)
        return score, reason
    except Exception as e:
        return None, f"Error: {str(e)}"

# ---------------------------------------------------
# New Insights Functions
# ---------------------------------------------------
def count_tokens(text, model="gpt-4o-mini"):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

async def analyze_comparison(session, competitor, category, aspect, year, sentiment_diff, reviews, prompt_template, model, temperature, max_tokens):
    messages = [
        {"role": "system", "content": prompt_template.format(competitor=competitor, category=category, aspect=aspect, year=year, sentiment_diff=sentiment_diff)},
        {"role": "user", "content": f"Analyze these reviews:\n\n{reviews}"}
    ]
    async with session.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}", "Content-Type": "application/json"},
            json={"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    ) as response:
        result = await response.json()
        return result["choices"][0]["message"]["content"]

async def analyze_insights(session, competitor, category, aspect, year, sentiment_diff, reviews, prompt_template, model, temperature, max_tokens):
    messages = [
        {"role": "system", "content": prompt_template.format(competitor=competitor, category=category, aspect=aspect, year=year, sentiment_diff=sentiment_diff)},
        {"role": "user", "content": f"Analyze these reviews:\n\n{reviews}"}
    ]
    async with session.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}", "Content-Type": "application/json"},
            json={"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    ) as response:
        result = await response.json()
        return result["choices"][0]["message"]["content"]

async def process_insights(sentiment_df, reviews_df, output_filename, comparison_prompt, insights_prompt, model, temperature, max_tokens, active_year):
    ASPECTS = ["Appointment Scheduling", "Customer Service", "Response Speed", "Engineer Experience", "Solution Quality", "Value For Money"]
    YEAR = active_year
    QUALITY_SCORE_THRESHOLD = 0.96
    MAX_TOKENS_PER_CHUNK = 16000
    WAIT_BETWEEN_CALLS = 1

    sentiment_df['Year-Month'] = pd.to_datetime(sentiment_df['Year-Month'], format='%d/%m/%Y', errors='raise')
    sentiment_df['Year'] = sentiment_df['Year-Month'].dt.year
    sentiment_df = sentiment_df[sentiment_df["Year"] == YEAR]
    if YEAR != "All":
        reviews_df = reviews_df[reviews_df["Year"] == YEAR]

    competitors = [c for c in sentiment_df["Company"].unique() if c != "British Gas"]
    categories = sentiment_df["Final Product Category"].unique()
    combinations = [(comp, cat, asp) for comp in competitors for cat in categories for asp in ASPECTS]

    total_combinations = len(combinations)
    output_cols = ["Competitor", "Product Line", "Aspect", "Sentiment Difference", "Year", "Analysis", "Key Insights"]
    pd.DataFrame(columns=output_cols).to_csv(output_filename, index=False)

    progress_bar = st.progress(0)
    progress_text = st.empty()
    preview_placeholder = st.empty()
    start_time = time.time()
    processed = 0

    async with aiohttp.ClientSession() as session:
        for i, (competitor, category, aspect) in enumerate(combinations):
            try:
                bg_sentiment = sentiment_df[(sentiment_df["Company"] == "British Gas") &
                                            (sentiment_df["Final Product Category"] == category)].iloc[-1]
                comp_sentiment = sentiment_df[(sentiment_df["Company"] == competitor) &
                                              (sentiment_df["Final Product Category"] == category)].iloc[-1]
                sentiment_diff = round(bg_sentiment[f"{aspect}_sentiment_score"] - comp_sentiment[f"{aspect}_sentiment_score"], 2)
            except (IndexError, KeyError):
                continue

            bg_reviews = reviews_df[(reviews_df["Company"] == "British Gas") &
                                    (reviews_df["Final Product Category"] == category) &
                                    (reviews_df["review_weight"] >= QUALITY_SCORE_THRESHOLD)]
            comp_reviews = reviews_df[(reviews_df["Company"] == competitor) &
                                      (reviews_df["Final Product Category"] == category) &
                                      (reviews_df["review_weight"] >= QUALITY_SCORE_THRESHOLD)]

            if bg_reviews.empty or comp_reviews.empty:
                continue

            max_reviews_per_company = min(len(bg_reviews), len(comp_reviews))
            tokens_per_review = 100
            max_possible_reviews = (MAX_TOKENS_PER_CHUNK // 2) // tokens_per_review
            num_reviews = min(max_reviews_per_company, max_possible_reviews)

            bg_sample = bg_reviews.nlargest(num_reviews, "review_weight")["Review"].tolist()
            comp_sample = comp_reviews.nlargest(num_reviews, "review_weight")["Review"].tolist()

            combined_text = f"British Gas Reviews ({num_reviews}):\n{' '.join(bg_sample)}\n\n{competitor} Reviews ({num_reviews}):\n{' '.join(comp_sample)}"
            total_tokens = count_tokens(combined_text)
            if total_tokens > MAX_TOKENS_PER_CHUNK:
                combined_text = combined_text[:MAX_TOKENS_PER_CHUNK * 4]

            tasks = [
                analyze_comparison(session, competitor, category, aspect, YEAR, sentiment_diff, combined_text, comparison_prompt, model, temperature, max_tokens),
                analyze_insights(session, competitor, category, aspect, YEAR, sentiment_diff, combined_text, insights_prompt, model, temperature, max_tokens)
            ]
            analysis, insights = await asyncio.gather(*tasks)

            result = {
                "Competitor": competitor,
                "Product Line": category,
                "Aspect": aspect,
                "Sentiment Difference": sentiment_diff,
                "Year": YEAR,
                "Analysis": analysis.strip(),
                "Key Insights": insights.strip()
            }
            pd.DataFrame([result]).to_csv(output_filename, mode='a', header=False, index=False)

            processed += 1
            elapsed = time.time() - start_time
            avg_time = elapsed / processed
            remaining = avg_time * (total_combinations - processed)
            progress_text.text(f"Processed {processed}/{total_combinations} combinations. Elapsed: {str(datetime.timedelta(seconds=int(elapsed)))} | Remaining: {str(datetime.timedelta(seconds=int(remaining)))}")
            progress_bar.progress(min(1.0, processed / total_combinations))

            with preview_placeholder.container():
                st.markdown("**Latest Output Preview:**")
                st.table(pd.DataFrame([result]))

            await asyncio.sleep(WAIT_BETWEEN_CALLS)

    return total_combinations, time.time() - start_time

# ---------------------------------------------------
# Utility Functions
# ---------------------------------------------------
def create_output_filename(input_filename, analyses, output_folder):
    base = os.path.splitext(os.path.basename(input_filename))[0] if input_filename else "uploaded_file"
    analyses_str = "_".join(analyses)
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{base}_{analyses_str}_{timestamp}.csv"
    return os.path.join(output_folder if output_folder.strip() != "" else ".", filename)


def check_csv_validity(file_df, required_cols):
    missing = [col for col in required_cols if col not in file_df.columns]
    if missing:
        return False, f"CSV missing required columns: {', '.join(missing)}."
    return True, ""


def update_run_log(run_info, log_file="run_log.csv"):
    df_log = pd.DataFrame([run_info])
    if os.path.exists(log_file):
        df_log.to_csv(log_file, mode='a', header=False, index=False)
    else:
        df_log.to_csv(log_file, index=False)


def process_reviews(file_df, output_filename, prod_prompt, sentiment_prompt,
                    enable_prod, enable_sentiment, sentiment_aspects, model, temperature, max_tokens):
    total_reviews = len(file_df)
    processed_rows = []
    start_time = time.time()
    progress_text = st.empty()
    progress_bar = st.progress(0)
    out_cols = ["Date", "Year-Month", "Review"]
    if enable_prod:
        out_cols += ["Prod_Cat_score", "Prod_Cat_product", "Prod_Cat_reason"]
    if enable_sentiment:
        for aspect in sentiment_aspects:
            out_cols += [f"{aspect}_sentiment_score", f"{aspect}_reason"]
    pd.DataFrame(columns=out_cols).to_csv(output_filename, index=False)
    for idx, row in file_df.iterrows():
        review = row.get("Review")
        date_val = row.get("Date", "")
        ym_val = row.get("Year-Month", "")
        result_row = {"Date": date_val, "Year-Month": ym_val, "Review": review}
        if not isinstance(review, str) or review.strip() == "":
            if enable_prod:
                result_row.update(
                    {"Prod_Cat_score": None, "Prod_Cat_product": None, "Prod_Cat_reason": "No review provided"})
            if enable_sentiment:
                for aspect in sentiment_aspects:
                    result_row.update({f"{aspect}_sentiment_score": None, f"{aspect}_reason": "No review provided"})
            pd.DataFrame([result_row]).to_csv(output_filename, mode='a', header=False, index=False)
            processed_rows.append(result_row)
        else:
            if enable_prod:
                score, product, reason = analyze_product_categorisation(review, prod_prompt, model, temperature,
                                                                        max_tokens)
                result_row.update({"Prod_Cat_score": score, "Prod_Cat_product": product, "Prod_Cat_reason": reason})
            if enable_sentiment:
                for aspect in sentiment_aspects:
                    s_score, s_reason = analyze_sentiment(review, aspect, sentiment_prompt, model, temperature,
                                                          max_tokens)
                    result_row.update({f"{aspect}_sentiment_score": s_score, f"{aspect}_reason": s_reason})
            pd.DataFrame([result_row]).to_csv(output_filename, mode='a', header=False, index=False)
            processed_rows.append(result_row)
        elapsed = time.time() - start_time
        avg_time = elapsed / (idx)
        remaining = avg_time * (total_reviews - idx)
        progress_text.text(
            f"Processed {idx} of {total_reviews} reviews. Elapsed: {str(datetime.timedelta(seconds=int(elapsed)))} | Estimated remaining: {str(datetime.timedelta(seconds=int(remaining)))}")
        progress_value = min(1.0, (idx) / total_reviews)
        progress_bar.progress(progress_value)
    with st.expander("✅ Successfully Processed Results..."):
        st.table(processed_rows)
    total_time = time.time() - start_time
    return processed_rows, total_time

def run(active_year):

    # ---------------------------------------------------
    # Dashboard Tabs with Emojis in Titles
    # ---------------------------------------------------
    st.markdown(f"# 🧮 Social Media Underlying Database")
    tabs = st.tabs(["🧪 Data Cleaning", "🏗️ Run Setup", "🚗 Live Running View", "🪵 Run Log", "🧠 AI Config", "🔮 Insights"])
    cleaning_tab, setup_tab, output_tab, runlog_tab, config_tab, insights_tab = tabs

    # ===================================================
    # CONFIG TAB
    # ===================================================
    with config_tab:
        st.markdown("<div style='text-align:center;'><h1>⚙️ AI API Configuration</h1></div>", unsafe_allow_html=True)
        with st.expander("⚙️ API Config",expanded=True):
            st.markdown("⚠️ <b>WARNING</b>: This section is for advanced API configuration. Adjust these settings only if necessary.", unsafe_allow_html=True)
            st.markdown("---------------------------------------------------------------------------", unsafe_allow_html=True)
            model_options = ["gpt-4o-mini", "gpt-3.5-turbo"]
            selected_model = st.selectbox("🧠 Select API Model:", options=model_options, index=0)
            temperature = st.slider("🔥 Temperature (default is 0.2):", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
            max_tokens = st.number_input("🗣️ Max Tokens (default is 150):", min_value=50, max_value=1000, value=150, step=10)
            if st.button("Save API Config"):
                st.session_state["api_config"] = {
                    "selected_model": selected_model,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                st.success("API configuration saved.")

    # ===================================================
    # NEW TAB: Data Cleaning & Analysis
    # ===================================================
    with cleaning_tab:
        st.markdown("<div style='text-align:center;'><h1>🧪 Data Cleaning & Analysis</h1></div>", unsafe_allow_html=True)
        st.markdown(
            "This section processes the reviews to detect spam, determine language, check for justification, and calculate a final weight for each review.")
        company_options = list(COMPANY_PRODUCTS.keys())
        selected_company_clean = st.selectbox("Select Company:", options=company_options, index=0)
        cleaning_file = st.file_uploader("Upload Reviews CSV for Cleaning", type=["csv"], key="cleaning_file")
        if cleaning_file is not None:
            try:
                df_clean = pd.read_csv(cleaning_file)
                valid, err_msg = check_csv_validity(df_clean, ["Review"])
                if not valid:
                    st.error(err_msg)
                else:
                    st.success("File loaded successfully!")
                    st.markdown("**Data Preview (first 5 rows):**")
                    st.dataframe(df_clean.head())

                    st.markdown("---")
                    st.markdown("### Step 1: Spam Detection")
                    spam_classifier = pipeline(
                        "text-classification",
                        model="mrm8488/bert-tiny-finetuned-sms-spam-detection",
                        truncation=True,
                        max_length=512,
                        padding=True
                    )
                    reviews_list = df_clean["Review"].tolist()
                    batch_size = 256
                    spam_labels = []
                    spam_scores = []
                    for i in range(0, len(reviews_list), batch_size):
                        batch = reviews_list[i: i + batch_size]
                        results = spam_classifier(batch)
                        for r in results:
                            spam_labels.append(r["label"])
                            spam_scores.append(round(r["score"], 3))
                    df_clean["spam_label"] = spam_labels
                    df_clean["spam_confidence"] = spam_scores
                    SPAM_CONFIDENCE_THRESHOLD = 0.9
                    df_clean["is_spam"] = df_clean.apply(lambda row: True if (row["spam_label"] == "spam" and row[
                        "spam_confidence"] > SPAM_CONFIDENCE_THRESHOLD) else False, axis=1)
                    st.success("Spam detection complete!")

                    st.markdown("---")

                    st.markdown("---")
                    st.markdown("### Step 2: Justification Check")
                    justification_regex = re.compile(r"\b(because|since|as|due to|reason|why|so that)\b", re.IGNORECASE)

                    def has_refined_justification(text):
                        return bool(justification_regex.search(str(text)))

                    df_clean["has_justification"] = df_clean["Review"].apply(has_refined_justification)
                    st.success("Justification check complete!")

                    st.markdown("---")
                    st.markdown("### Step 3: Calculate Final Review Weight")

                    def calculate_review_weight(row):
                        if row["is_spam"]:
                            return 0.0
                        weight = 1.0
                        if row["has_justification"]:
                            weight *= 1.0
                        else:
                            weight *= 0.9
                        return weight

                    df_clean["review_weight"] = df_clean.apply(calculate_review_weight, axis=1)
                    st.success("Review weight calculation complete!")

                    st.markdown("---")
                    st.markdown("### Final Cleaned Data Preview")
                    st.dataframe(df_clean.head(20))

                    output_clean_filename = f"Cleaned Reviews {selected_company_clean} {datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
                    df_clean.to_csv(output_clean_filename, index=False)
                    try:
                        with open(output_clean_filename, "rb") as file:
                            st.download_button(
                                label="⬇️ Download Cleaned CSV",
                                data=file,
                                file_name=output_clean_filename,
                                mime="text/csv"
                            )
                    except Exception as e:
                        st.error(f"Error providing download: {e}")
            except Exception as e:
                st.error(f"Error processing file: {e}")
        else:
            st.info("Please upload a CSV file for cleaning.")

    # ===================================================
    # SETUP TAB
    # ===================================================
    with setup_tab:
        st.markdown("<div style='text-align:center;'><h1>📝 Setup</h1></div>", unsafe_allow_html=True)
        st.markdown("#### Follow the steps below to set up your analysis:")

        # --- Step 1: File Input and Validation ---
        st.markdown("<hr style='border: 1px solid #0490d7;'>", unsafe_allow_html=True)
        st.markdown("<h3>Step 1: File Input and Validation</h3>", unsafe_allow_html=True)
        st.markdown("**Upload your CSV file** (must contain a **Review** column) or enter its full file path:")
        col1, col2 = st.columns([3,2])
        with col1:
            with st.container(border=True):
                st.markdown("**Input: Social media data file**")
                input_method = st.radio("Select input method:", options=["Upload File", "Enter File Path"], key="input_method")
                uploaded_file = None
                input_file_path = ""
                if input_method == "Upload File":
                    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="uploaded_file")
                else:
                    input_file_path = st.text_input("Enter full file path:", value="", key="input_file_path")
                if st.button("Validate File"):
                    try:
                        if input_method == "Upload File":
                            if uploaded_file is None:
                                st.error("Please upload a file.")
                            else:
                                file_df = pd.read_csv(uploaded_file)
                                valid, err_msg = check_csv_validity(file_df, ["Review"])
                                if valid:
                                    st.session_state["file_df"] = file_df
                                    st.success("File validated successfully!")
                                else:
                                    st.error(err_msg)
                        else:
                            if not os.path.exists(input_file_path):
                                st.error("File path does not exist.")
                            else:
                                file_df = pd.read_csv(input_file_path)
                                valid, err_msg = check_csv_validity(file_df, ["Review"])
                                if valid:
                                    st.session_state["file_df"] = file_df
                                    st.success("File validated successfully!")
                                else:
                                    st.error(err_msg)
                    except Exception as e:
                        st.error(f"Error validating file: {e}")
        with col2:
            with st.container(border=True):
                st.markdown("**Select Specific AI Analysis Options**")
                enable_prod = st.checkbox("🧩 Enable Product Categorisation", value=True)
                enable_sentiment = st.checkbox("🤬 Enable Sentiment Analysis", value=True)
                if not enable_prod and not enable_sentiment:
                    st.error("At least one analysis option must be enabled.")

        # --- Step 2: Analysis Options & Detailed Configuration ---
        if "file_df" in st.session_state:
            st.markdown("<hr style='border: 1px solid #0490d7;'>", unsafe_allow_html=True)
            st.markdown("<h3>Step 2: Analysis Options & Detailed Configuration</h3>",
                        unsafe_allow_html=True)
            file_df = st.session_state["file_df"]
            st.markdown("**File Preview (first 5 rows):**")
            st.dataframe(file_df.head())
            col1, col2 = st.columns([4, 2])
            with col1:
                if enable_prod:
                    with st.expander("🧩 Product Analysis Config:", expanded=True):
                        st.markdown("**Product Categorisation Configuration:**")
                        st.markdown("Select a company and choose which products to include.")
                        company_options = list(COMPANY_PRODUCTS.keys())
                        selected_company = st.selectbox("Select Company:", options=company_options, index=0, key="company_cleaning_select")
                        products_for_company = COMPANY_PRODUCTS[selected_company]
                        selected_products = {}
                        st.markdown("**Product Details:**")
                        for prod in products_for_company:
                            default_desc = PRODUCT_DESCRIPTIONS.get(prod, "No description available")
                            col_a, col_b = st.columns([3, 1])
                            with col_a:
                                new_desc = st.text_area(f"Description for **{prod}**:", value=default_desc, key=f"desc_{prod}")
                            with col_b:
                                include = st.checkbox(f"Include **{prod}**", value=True, key=f"chk_{prod}")
                            if include:
                                selected_products[prod] = new_desc
                        if not selected_products:
                            st.warning("No products selected. At least one must be selected.")
                        prod_prompt = generate_prod_prompt(selected_company, selected_products)
                else:
                    with st.expander("🧩 Product Analysis Config (disabled):", expanded=False):
                        st.write("This analysis has been disabled by the user")
            with col2:
                if enable_sentiment:
                    with st.expander("🤬 Sentiment Analysis Config:", expanded=True):
                        st.markdown("**Sentiment Analysis Configuration:**")
                        st.markdown("Select which sentiment aspects to analyze:")
                        default_aspects = ["Overall Sentiment", "Appointment Scheduling", "Customer Service", "Response Speed", "Engineer Experience", "Solution Quality", "Value for Money"]
                        sentiment_aspects = st.multiselect("Select Sentiment Aspects:", options=default_aspects,
                                                           default=default_aspects)
                else:
                    sentiment_aspects = []
                    with st.expander("🤬 Sentiment Analysis Config (disabled):", expanded=False):
                        st.write("This analysis has been disabled by the user")

            st.markdown("**Output File Configuration:**")
            st.markdown("Specify the folder where the output file should be saved (leave blank for current folder):")
            output_folder = st.text_input("Output Folder:", value="")

            # Button to confirm analysis options and reveal preview step
            if (enable_prod or enable_sentiment) and st.button("Confirm Analysis Options"):
                st.session_state["analysis_confirmed"] = True

        # --- Step 3: First Row Processing Preview (Only after confirming analysis options) ---
        if "analysis_confirmed" in st.session_state and st.session_state["analysis_confirmed"]:
            st.markdown("<hr style='border: 1px solid #0490d7;'>", unsafe_allow_html=True)
            st.markdown("<h3>Step 3: First Row Processing Preview</h3>",
                        unsafe_allow_html=True)
            first_row = file_df.iloc[[0]].copy()
            row = first_row.iloc[0]
            review = row.get("Review", "")
            preview_result = {"Date": row.get("Date", ""), "Year-Month": row.get("Year-Month", ""), "Review": review}
            api_defaults = st.session_state.get("api_config",
                                                {"selected_model": "gpt-4o-mini", "temperature": 0.2, "max_tokens": 150})
            if enable_prod:
                score, product, reason = analyze_product_categorisation(
                    review, prod_prompt,
                    api_defaults["selected_model"], api_defaults["temperature"], api_defaults["max_tokens"]
                )
                preview_result.update({"Prod_Cat_score": score, "Prod_Cat_product": product, "Prod_Cat_reason": reason})
            if enable_sentiment:
                for aspect in sentiment_aspects:
                    s_score, s_reason = analyze_sentiment(
                        review, aspect, DEFAULT_SENTIMENT_PROMPT,
                        api_defaults["selected_model"], api_defaults["temperature"], api_defaults["max_tokens"]
                    )
                    preview_result.update({f"{aspect}_sentiment_score": s_score, f"{aspect}_reason": s_reason})
            st.markdown("**First Row Processed Preview:**")
            st.table(pd.DataFrame([preview_result]))

            if st.button("Approve and Save Run Configuration"):
                st.session_state["run_config"] = {
                    "input_method": input_method,
                    "uploaded_file": uploaded_file,
                    "input_file_path": input_file_path,
                    "output_folder": output_folder,
                    "enable_prod": enable_prod,
                    "enable_sentiment": enable_sentiment,
                    "selected_company": selected_company if enable_prod else "",
                    "selected_products": selected_products if enable_prod else {},
                    "prod_prompt": prod_prompt if enable_prod else "",
                    "sentiment_aspects": sentiment_aspects,
                    "sentiment_prompt": DEFAULT_SENTIMENT_PROMPT
                }
                st.session_state["run_requested"] = True
                st.success("Run configuration saved! Progress updates consolidated in Output tab")

    # ===================================================
    # OUTPUT TAB
    # ===================================================
    with output_tab:
        st.markdown("<div style='text-align:center;'><h1>📤 Output</h1></div>", unsafe_allow_html=True)
        st.markdown("This section processes the entire file and provides a download link for the processed CSV.")
        if "run_requested" not in st.session_state or not st.session_state.get("run_requested"):
            st.markdown("Please complete the setup in the Setup tab and save the run configuration.")
        else:
            config = st.session_state["run_config"]
            api_config = st.session_state.get("api_config",
                                              {"selected_model": "gpt-4o-mini", "temperature": 0.2, "max_tokens": 150})
            file_df = st.session_state["file_df"]
            if config["input_method"] == "Upload File" and config.get("uploaded_file") is not None:
                input_filename = config["uploaded_file"].name
            else:
                input_filename = os.path.basename(config["input_file_path"])
            analyses_selected = []
            if config["enable_prod"]:
                analyses_selected.append("prod")
            if config["enable_sentiment"]:
                analyses_selected.append("sentiment")
            output_filename = create_output_filename(input_filename, analyses_selected, config["output_folder"])
            st.markdown(f"**Output will be saved to:** `{output_filename}`")
            if "processing_done" not in st.session_state:
                with st.spinner("Processing reviews..."):
                    processed_rows, total_time = process_reviews(
                        file_df.iloc[1:],
                        output_filename,
                        config["prod_prompt"],
                        config["sentiment_prompt"],
                        config["enable_prod"],
                        config["enable_sentiment"],
                        config["sentiment_aspects"],
                        api_config["selected_model"],
                        api_config["temperature"],
                        api_config["max_tokens"]
                    )
                    st.session_state["processing_done"] = True
                    st.session_state["processing_time"] = total_time
                    st.success(f"Processing complete in {str(datetime.timedelta(seconds=int(total_time)))}")
                    run_info = {
                        "Input File": input_filename,
                        "Output File": os.path.basename(output_filename),
                        "Run Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Total Rows": len(file_df),
                        "Rows Processed": len(file_df),
                        "Analyses": ", ".join(analyses_selected),
                        "Model": api_config["selected_model"],
                        "Temperature": api_config["temperature"],
                        "Max Tokens": api_config["max_tokens"],
                        "Processing Time (s)": int(total_time)
                    }
                    update_run_log(run_info)
                try:
                    with open(output_filename, "rb") as file:
                        st.download_button(
                            label="⬇️ Download Processed CSV",
                            data=file,
                            file_name=os.path.basename(output_filename),
                            mime="text/csv"
                        )
                except Exception as e:
                    st.error(f"Error providing download: {e}")
                try:
                    processed_df = pd.read_csv(output_filename)
                    st.markdown("**Processed Data Preview (first 20 rows):**")
                    st.dataframe(processed_df.head(20))
                except Exception as e:
                    st.error(f"Error reading processed output: {e}")
            else:
                st.markdown("Processing already completed. See output below.")
                try:
                    processed_df = pd.read_csv(output_filename)
                    st.dataframe(processed_df.head(20))
                except Exception as e:
                    st.error(f"Error reading processed output: {e}")

    # ===================================================
    # RUN LOG TAB
    # ===================================================
    with runlog_tab:
        st.markdown("<div style='text-align:center;'><h1>📋 Run Log</h1></div>", unsafe_allow_html=True)
        st.markdown(
            "Below is a log of previous runs with details such as input/output files, processing time, and API settings.")
        log_file = "run_log.csv"
        if os.path.exists(log_file):
            try:
                log_df = pd.read_csv(log_file)
                st.dataframe(log_df)
            except Exception as e:
                st.error(f"Error reading run log: {e}")
        else:
            st.markdown("No run log available yet.")

    # New Insights Tab
    with insights_tab:
        st.markdown("<div style='text-align:center;'><h1>🔮 Insights</h1></div>", unsafe_allow_html=True)
        st.markdown("Generate competitive insights comparing British Gas to competitors based on sentiment and reviews.")

        # Step 1: File Input
        st.markdown("<hr style='border: 1px solid #0490d7;'>", unsafe_allow_html=True)
        st.markdown("<h3>Step 1: File Input</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Sentiment Data**")
            sentiment_input_method = st.radio("Select input method:", ["Upload File", "Enter File Path"], key="sentiment_input")
            sentiment_file = None
            sentiment_path = ""
            if sentiment_input_method == "Upload File":
                sentiment_file = st.file_uploader("Upload Sentiment CSV", type=["csv"], key="sentiment_file")
            else:
                sentiment_path = st.text_input("Enter Sentiment File Path:", key="sentiment_path", value="LLM SA Monthly Data.csv")
        with col2:
            st.markdown("**Reviews Data**")
            reviews_input_method = st.radio("Select input method:", ["Upload File", "Enter File Path"], key="reviews_input")
            reviews_file = None
            reviews_path = ""
            if reviews_input_method == "Upload File":
                reviews_file = st.file_uploader("Upload Reviews CSV", type=["csv"], key="reviews_file")
            else:
                reviews_path = st.text_input("Enter Reviews File Path:", key="reviews_path", value="reviews_all.csv")

        if st.button("Validate Files"):
            try:
                sentiment_df = pd.read_csv(sentiment_file) if sentiment_file else pd.read_csv(sentiment_path)
                reviews_df = pd.read_csv(reviews_file) if reviews_file else pd.read_csv(reviews_path)
                sentiment_valid, sentiment_err = check_csv_validity(sentiment_df, ["Company", "Final Product Category", "Year-Month"])
                reviews_valid, reviews_err = check_csv_validity(reviews_df, ["Company", "Final Product Category", "Review", "review_weight", "Year"])
                if not sentiment_valid:
                    st.error(sentiment_err)
                elif not reviews_valid:
                    st.error(reviews_err)
                else:
                    st.session_state["insights_sentiment_df"] = sentiment_df
                    st.session_state["insights_reviews_df"] = reviews_df
                    st.success("Files validated successfully!")
                    st.markdown("**Sentiment Preview:**")
                    st.dataframe(sentiment_df.head())
                    st.markdown("**Reviews Preview:**")
                    st.dataframe(reviews_df.head())
            except Exception as e:
                st.error(f"Error validating files: {e}")

        # Step 2: Config Prompts
        if "insights_sentiment_df" in st.session_state and "insights_reviews_df" in st.session_state:
            st.markdown("<hr style='border: 1px solid #0490d7;'>", unsafe_allow_html=True)
            st.markdown("<h3>Step 2: Configure Prompts</h3>", unsafe_allow_html=True)
            comparison_prompt = st.text_area("Comparison Prompt:", value=DEFAULT_COMPARISON_PROMPT, height=200)
            insights_prompt = st.text_area("Insights Prompt:", value=DEFAULT_INSIGHTS_PROMPT, height=200)
            output_folder = st.text_input("Output Folder (leave blank for current folder):", value="")
            api_config = st.session_state.get("api_config", {"selected_model": "gpt-4o-mini", "temperature": 0.2, "max_tokens": 200})

            if st.button("Preview First Combination"):
                sentiment_df = st.session_state["insights_sentiment_df"]
                reviews_df = st.session_state["insights_reviews_df"]
                first_combination = [(comp, cat, asp) for comp in [c for c in sentiment_df["Company"].unique() if c != "British Gas"]
                                    for cat in sentiment_df["Final Product Category"].unique()
                                    for asp in ["Appointment Scheduling"]][0]  # Preview one aspect
                competitor, category, aspect = first_combination

                try:
                    bg_sentiment = sentiment_df[(sentiment_df["Company"] == "British Gas") &
                                                (sentiment_df["Final Product Category"] == category)].iloc[-1]
                    comp_sentiment = sentiment_df[(sentiment_df["Company"] == competitor) &
                                                  (sentiment_df["Final Product Category"] == category)].iloc[-1]
                    sentiment_diff = round(bg_sentiment[f"{aspect}_sentiment_score"] - comp_sentiment[f"{aspect}_sentiment_score"], 2)
                except (IndexError, KeyError):
                    st.error("No sentiment data for first combination.")
                    return

                bg_reviews = reviews_df[(reviews_df["Company"] == "British Gas") &
                                        (reviews_df["Final Product Category"] == category) &
                                        (reviews_df["review_weight"] >= 0.96)]
                comp_reviews = reviews_df[(reviews_df["Company"] == competitor) &
                                          (reviews_df["Final Product Category"] == category) &
                                          (reviews_df["review_weight"] >= 0.96)]

                if not bg_reviews.empty and not comp_reviews.empty:
                    num_reviews = min(min(len(bg_reviews), len(comp_reviews)), (16000 // 2) // 100)
                    bg_sample = bg_reviews.nlargest(num_reviews, "review_weight")["Review"].tolist()
                    comp_sample = comp_reviews.nlargest(num_reviews, "review_weight")["Review"].tolist()
                    combined_text = f"British Gas Reviews ({num_reviews}):\n{' '.join(bg_sample)}\n\n{competitor} Reviews ({num_reviews}):\n{' '.join(comp_sample)}"

                    async def preview():
                        async with aiohttp.ClientSession() as session:
                            tasks = [
                                analyze_comparison(session, competitor, category, aspect, active_year, sentiment_diff, combined_text,
                                                  comparison_prompt, api_config["selected_model"], api_config["temperature"], api_config["max_tokens"]),
                                analyze_insights(session, competitor, category, aspect, active_year, sentiment_diff, combined_text,
                                                 insights_prompt, api_config["selected_model"], api_config["temperature"], api_config["max_tokens"])
                            ]
                            return await asyncio.gather(*tasks)

                    analysis, insights = asyncio.run(preview())
                    preview_result = {
                        "Competitor": competitor,
                        "Product Line": category,
                        "Aspect": aspect,
                        "Sentiment Difference": sentiment_diff,
                        "Year": active_year,
                        "Analysis": analysis.strip(),
                        "Key Insights": insights.strip()
                    }
                    st.markdown("**Preview of First Combination:**")
                    st.table(pd.DataFrame([preview_result]))
                    st.session_state["insights_preview"] = preview_result
                    st.session_state["insights_config"] = {
                        "sentiment_df": sentiment_df,
                        "reviews_df": reviews_df,
                        "comparison_prompt": comparison_prompt,
                        "insights_prompt": insights_prompt,
                        "output_folder": output_folder,
                        "model": api_config["selected_model"],
                        "temperature": api_config["temperature"],
                        "max_tokens": api_config["max_tokens"]
                    }

        # Step 3: Run Full Process
        if "insights_config" in st.session_state and st.button("Run Full Analysis"):
            config = st.session_state["insights_config"]
            sentiment_input = sentiment_file.name if sentiment_file else sentiment_path
            reviews_input = reviews_file.name if reviews_file else reviews_path
            output_filename = create_output_filename(sentiment_input, config["output_folder"])
            with st.spinner("Processing insights..."):
                total_combinations, total_time = asyncio.run(process_insights(
                    config["sentiment_df"], config["reviews_df"], output_filename,
                    config["comparison_prompt"], config["insights_prompt"],
                    config["model"], config["temperature"], config["max_tokens"], active_year
                ))
                st.success(f"Completed in {str(datetime.timedelta(seconds=int(total_time)))}")
                run_info = {
                    "Input File": f"{sentiment_input}, {reviews_input}",
                    "Output File": os.path.basename(output_filename),
                    "Run Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Total Rows": total_combinations,
                    "Rows Processed": total_combinations,
                    "Analyses": "insights",
                    "Model": config["model"],
                    "Temperature": config["temperature"],
                    "Max Tokens": config["max_tokens"],
                    "Processing Time (s)": int(total_time)
                }
                update_run_log(run_info)
            with open(output_filename, "rb") as file:
                st.download_button(label="⬇️ Download Insights CSV", data=file, file_name=os.path.basename(output_filename), mime="text/csv")

#run()
