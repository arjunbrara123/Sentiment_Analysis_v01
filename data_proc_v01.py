import streamlit as st
import pandas as pd
import os
import time
import datetime
from openai import OpenAI
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
# Fixed Prompts and Default Data
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
    "Appliance Cover": "Insurance for home appliances like ovens, washing machines, etc.",
    "Home Electrical": "Covers issues with home electrics, wiring, fusebox breakdowns, and broken sockets.",
    "Plumbing and Drains": "Covers plumbing repairs such as blocked drains or frozen pipes.",
    "Building": "General building services."
}

COMPANY_PRODUCTS = {
    "British Gas": ["Gas Products", "Energy", "Appliance Cover", "Home Electrical", "Plumbing and Drains"],
    "HomeServe": ["Gas Products", "Plumbing and Drains", "Home Electrical"],
    "CheckATrade": ["Plumbing and Drains", "Home Electrical", "Building"]
}


def generate_prod_prompt(company, selected_products_dict):
    prompt = f"You are a product analysis expert. Your task is to assess customer reviews for {company} and determine the most appropriate product category from the following options:\n"
    i = 1
    for prod, desc in selected_products_dict.items():
        prompt += f"{i}. {prod}: {desc}\n"
        i += 1
    prompt += "\n### Guidance:\n"
    prompt += "- Look for keywords or scenarios that hint at the correct category.\n"
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
# Utility Functions
# ---------------------------------------------------
def create_output_filename(input_filename, analyses, output_folder):
    base = os.path.splitext(os.path.basename(input_filename))[0] if input_filename else "uploaded_file"
    analyses_str = "_".join(analyses)
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{base}_{analyses_str}_{timestamp}.csv"
    return os.path.join(output_folder if output_folder.strip() != "" else ".", filename)


def check_csv_validity(file_df):
    if "Review" not in file_df.columns:
        return False, "CSV file must contain a column named 'Review'."
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
        avg_time = elapsed / (idx + 1)
        remaining = avg_time * (total_reviews - idx - 1)
        progress_text.text(
            f"Processed {idx + 1} of {total_reviews} reviews. Elapsed: {str(datetime.timedelta(seconds=int(elapsed)))} | Estimated remaining: {str(datetime.timedelta(seconds=int(remaining)))}")
        progress_value = min(1.0, (idx + 1) / total_reviews)
        progress_bar.progress(progress_value)
        if (idx + 1) % 100 == 0:
            st.session_state["latest_preview"] = pd.DataFrame(processed_rows[-100:])
            with st.expander("😶‍🌫️ Results Preview"):
                st.table(pd.DataFrame(processed_rows))
    total_time = time.time() - start_time
    return processed_rows, total_time

def run():

    # ---------------------------------------------------
    # Dashboard Tabs with Emojis in Titles
    # ---------------------------------------------------
    st.markdown(f"# 🧮 Social Media Underlying Database")
    tabs = st.tabs(["🏗️ Setup", "🚗 Output", "🪵 Run Log", "🧠 AI Config"])
    setup_tab, output_tab, runlog_tab, config_tab = tabs

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
                                valid, err_msg = check_csv_validity(file_df)
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
                                valid, err_msg = check_csv_validity(file_df)
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
                enable_prod = st.checkbox("Enable Product Categorisation", value=True)
                enable_sentiment = st.checkbox("Enable Sentiment Analysis", value=True)
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
                        selected_company = st.selectbox("Select Company:", options=company_options, index=0)
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

#run()
