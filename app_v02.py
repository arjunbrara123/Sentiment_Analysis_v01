"""
Competitor Intelligence Dashboard
============================================

Overview:
---------
This dashboard provides interactive insights into competitor sentiment and market trends using Streamlit.
It leverages AI-driven analyses and interactive Plotly charts to help users explore two primary modes:
    • Company Mode: Detailed sentiment analysis and comparisons for a selected competitor, including an AI chatbot ("Alice") for insights.
    • Market Mode: Aggregated market sentiment trends and insights across products/markets.

Required File Inputs:
-----------------------
1) company_summary_data: AI Generated insight for Company Mode, by Competitor/Product/Aspect, file 'LLM Prod Level Summary'
2) market_summary_data: AI Generated summaries for Market Mode, by Product breakdown only, file 'LLM Market Summary'.
3) sa_monthly_data: Monthly sentiment data - aggregated up by company/market for company/market mode, file 'LLM SA Monthly Data'.
4) reviews: Reviews for a specific company (One file per company) - Used for 'Ask Alice' mode only, file 'Cleaned Reviews <CompanyName>'

Supporting modules:
    - auth.py: Handles secure user authentication.
    - chatbot.py: Manages the AI chatbot interactions.
    - data_proc_v01: Provides additional data processing functions (for Dev Mode).
    - charts.py: Contains all charting functions.
    - main.py: Orchestrates page setup, data loading, UI configuration, and view rendering.

Code Structure Overview:
    1. Setup Functions: Import libraries, configure the HTML Streamlit page, load custom CSS, and initialize environment variables.
    2. Data Loading and Preprocessing Functions: Read and clean CSV data for sentiment, market summaries, and company details.
    3. Sidebar Configuration: Render sidebar widgets and capture user inputs for mode selection and filters.
    4. Rendering Functions: Display various dashboard views (company overview, chatbot, market trends, dev mode).
    5. Main Application Workflow: Orchestrate the overall process by integrating setup, data loading, sidebar, and rendering.

"""

# =============================================================================
# Dashboard Setup
# =============================================================================

#Import required python packages
import streamlit as st
import pandas as pd
#from dotenv import load_dotenv

# Input Filepath #1 - Company Mode - AI Gen Summary Text Data
company_summary_data_filepath = "LLM Prod Level Summary v3.csv"

# Input File #2 - Market Mode - AI Gen Summary Text Data
market_summary_data_filepath = "LLM Market Summary v3.csv"

# Input File #3 - Monthly sentiment data - agg by company / market for company / market mode respectively
sa_monthly_data_filepath = "LLM SA Monthly Data.csv"


#Configures the Streamlit page settings including title, favicon, and layout.
st.set_page_config(
    page_title="Competitor Intelligence App 🔬",
    page_icon="images/favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import custom modules
import auth
import chatbot
import data_proc_v01
from charts import *  # Contains all Plotly chart functions

# Loads and injects custom CSS into the Streamlit app.
try:
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except Exception as e:
    st.error(f"Error loading CSS: {e}")


# ==========================================================================================
# Data Loading and Preprocessing Functions: Functions for loading CSV files and preparing the data
# ------------------------------------------------------------------------------------------
# 1. load_data: Aggregates loading of all 3 CSVs files, and return them into a dictionary with keys for easy access.
# ==========================================================================================

@st.cache_data(show_spinner=True)
def load_data() -> dict:
    # Purpose: Loads all required CSV data into a dictionary for easy access.
    # Returns: dict: Contains dataframes for company summaries, sentiment data, and market summaries.

    data = {}

    # Input Filepath #1 - Company Mode - AI Gen Summary Text Data
    try:
        data["company_summary_data"] = pd.read_csv(company_summary_data_filepath)
    except Exception as e:
        st.error(f"Error loading company summary data: {e}")
        data["company_summary_data"] = pd.DataFrame()

    # Input File #2 - Market Mode - AI Gen Summary Text Data
    data["market_summary_data"] = pd.read_csv("LLM Market Summary v3.csv")

    # Input File #3 - Monthly sentiment data - agg by company / market for company / market mode respectively
    try:
        sa_df = pd.read_csv(sa_monthly_data_filepath)
        sa_df['Year-Month'] = pd.to_datetime(sa_df['Year-Month'], format='%d/%m/%Y', errors='raise')
        sa_df.sort_values('Year-Month', inplace=True)
        data["sa_monthly_data"] = sa_df

    except Exception as e:
        st.error(f"Error loading SA Monthly Data: {e}")
        data["sa_monthly_data"] = pd.DataFrame()

    # Extract unique company list for sidebar use.
    if not data["company_summary_data"].empty:
        data["company_list"] = data["company_summary_data"]['Company'].unique().tolist()
    else:
        data["company_list"] = []
    return data


# =============================================================================
# Sidebar Configuration
# This section contains only 1 function, which is responsible for rendering the sidebar widgets and capturing user inputs.
# =============================================================================
def configure_sidebar(company_summary_data: pd.DataFrame) -> dict:
    # Purpose: Renders the sidebar with 4-5 control boxes, and collects user input selections.
    # Returns: User selections including mode, company, product/market, analysis mode, and filter settings.

    inputs = {}
    with st.sidebar: # Create left-hand side bar with various navigation controls

        # British Gas Logo at the Top of the sidebar
        st.sidebar.image("images/company_logo.png")

        # User Navigation Box 1 of 5
        with st.expander("🏫 Select Competitor", expanded=True):
            inputs["mode"] = st.radio("Select Mode", options=["🎍 Market Mode", "🏢 Company Mode"], index=0)
            prod_option_list = company_summary_data['Product'].unique().tolist()
            if inputs["mode"] == "🏢 Company Mode":
                inputs["selected_company"] = st.radio("Please Select a Company",
                                                       options=company_summary_data['Company'].unique().tolist(),
                                                       index=0)
                prod_option_list = company_summary_data[
                    company_summary_data["Company"] == inputs["selected_company"]
                ]["Product"].unique().tolist()
            else:
                inputs["selected_company"] = None

        # User Navigation Box 2 of 5
        with st.expander(f"🎁 Select {'Product' if inputs['mode'] == '🏢 Company Mode' else 'Market'}", expanded=True):
            if 'All' not in prod_option_list and inputs["mode"] != "🏢 Company Mode":
                prod_option_list.insert(0, "All")
            inputs["selected_product"] = st.radio(
                f"Please Select a {'Product' if inputs['mode'] == '🏢 Company Mode' else 'Market'}",
                options=[PRODUCT_CONFIG.emoji_map.get(product, product) for product in prod_option_list],
                index=0
            )

        if inputs["mode"] == "🏢 Company Mode":
            # Load review data for the selected company (one file per company)
            reviews_filepath = f"Cleaned Reviews {inputs['selected_company']}.csv"
            try:
                inputs["reviews_data"] = pd.read_csv(reviews_filepath)
            except Exception as e:
                st.error(f"Error loading reviews data for {inputs['selected_company']}: {e}")
                inputs["reviews_data"] = pd.DataFrame()
        else:
            inputs["reviews_data"] = None

        # User Navigation Box 3 of 5
        if inputs["mode"] == "🏢 Company Mode":
            prod_name = inputs["selected_product"].split(' ', 1)[-1]
            analysis_options = ["🚁 Overview", "👽 Emerging Trends", "🙋‍♀️ Ask Alice..."] if prod_name != "All" else ["🚁 Overview", "👽 Emerging Trends"]
            with st.expander("🧩 Analysis Mode", expanded=True):
                inputs["analysis_mode"] = st.radio("Please Select Analysis", options=analysis_options, index=0)
        else:
            inputs["analysis_mode"] = "🚁 Overview"

        # User Navigation Box 4 of 5
        with st.expander("⌚ Time Period Settings", expanded=False):
            inputs["filter_year"] = st.selectbox("Pick a Year to Display",
                                                   ("All", "2021", "2022", "2023", "2024", "2025"),
                                                   index=4)

        # User Navigation Box 5 of 5
        with st.expander("🧠 AI Settings", expanded=False):
            inputs["tts_flag"] = st.toggle("Alice Reads Responses Aloud", value=False)
            inputs["dev_mode"] = st.toggle("Dev Mode", value=False)

    return inputs


# ==========================================================================================
# Rendering Functions:
# This section contains functions responsible for displaying various dashboard views based on user inputs.
# ------------------------------------------------------------------------------------------
# 1. render_company_mode: Renders competitor-specific views with detailed sentiment charts and AI analysis.
# 2. render_company_overview: Renders the "Overview" tab for Company Mode, displaying charts and analysis texts.
# 3. render_ask_alice: Provides the interactive "Ask Alice" chatbot interface for query-driven insights in company mode.
# 4. render_market_mode: Renders market-wide sentiment trends and insights based on product/market selection.
# 5. render_dev_mode: Executes additional processing functions when Development Mode is enabled.
# ==========================================================================================
def render_company_mode(data: dict, inputs: dict) -> None:
    """
    Renders the Company Mode view including the header, detailed overview, aspect tabs, and the "Ask Alice" view if selected.

    Args:
        data (dict): Dictionary containing the loaded data.
        inputs (dict): User selections from the sidebar.
    """
    company_name = inputs["selected_company"]
    product_name = inputs["selected_product"].split(' ', 1)[-1]
    analysis_mode = inputs["analysis_mode"]

    # Header: Display company logo and title.
    col1, col2 = st.columns([1, 7])
    with col1:
        st.image(f"images/{company_name.lower()} logo.png", width=100)
    with col2:
        st.markdown(f"# {company_name} Analytics: {product_name}")

    # Filter company summary data for the selected company and product.
    company_summary_data = data["company_summary_data"]
    selected_rows = company_summary_data[
        (company_summary_data["Company"] == company_name) &
        (company_summary_data["Product"] == product_name)
    ]

    # Route to the proper analysis mode.
    if analysis_mode == "🚁 Overview":
        render_company_overview(data, inputs, selected_rows, company_name, product_name)
    elif analysis_mode == "👽 Emerging Trends":
        st.markdown("## Emerging Customer Sentiment Trends")
    elif analysis_mode == "🙋‍♀️ Ask Alice...":
        render_ask_alice(inputs, company_name, product_name)


def render_company_overview(data: dict, inputs: dict, selected_rows: pd.DataFrame, company_name: str, product_name: str) -> None:
    """
    Renders the "Overview" tab for Company Mode, displaying sentiment charts and analysis texts.

    Args:
        data (dict): Loaded data dictionary.
        inputs (dict): User selections.
        selected_rows (pd.DataFrame): Filtered company summary data.
        company_name (str): Selected company name.
        product_name (str): Selected product name.
    """
    tabs_labels = ["✈️ Overview"] + [ASPECT_CONFIG.aspects_map[aspect] for aspect in ASPECT_CONFIG.aspects_map]
    tabs = st.tabs(tabs_labels)

    # Overview Tab
    with tabs[0]:
        chart_toggle = st.toggle("Split Sentiment Into Aspects", value=True,
                                   help="Toggle between Aspect View and overall Sentiment View")
        view = "aspect" if chart_toggle else "sentiment"
        prod_metric = None
        if chart_toggle and product_name == "All":
            prod_metric = st.selectbox(
                "Pick an Aspect to breakdown:",
                ("Overall Sentiment Score", "Appointment Scheduling", "Customer Service",
                 "Response Speed", "Engineer Experience", "Solution Quality", "Value For Money")
            )

        sa_monthly_data = data["sa_monthly_data"]
        # Left Column: Data for the selected company.
        col_left, col_right = st.columns(2)
        with col_left:
            if "all" not in product_name.lower():
                filtered_left = sa_monthly_data[
                    (sa_monthly_data["Company"].str.contains(company_name)) &
                    (sa_monthly_data["Final Product Category"].str.contains(product_name))
                ]
            else:
                filtered_left = sa_monthly_data[sa_monthly_data["Company"].str.contains(company_name)]
            if not filtered_left.empty:
                if product_name.lower() == "all":
                    plot_chart_all_products(product_name, f"{company_name} Sentiment", "", filtered_left, prod_metric, company_name)
                else:
                    plot_chart_2(product_name, company_name, "", filtered_left, view)
            else:
                st.write("No sentiment data available for the selected company and product.")

        # Right Column: Data for British Gas comparison (if applicable).
        with col_right:
            if "British Gas" not in company_name:
                if "all" not in product_name.lower():
                    filtered_right = sa_monthly_data[
                        (sa_monthly_data["Company"].str.contains("British Gas")) &
                        (sa_monthly_data["Final Product Category"].str.contains(product_name))
                    ]
                else:
                    filtered_right = sa_monthly_data[sa_monthly_data["Company"].str.contains("British Gas")]
                if not filtered_right.empty:
                    if product_name.lower() == "all":
                        plot_chart_all_products(product_name, "British Gas Sentiment", "", filtered_right, prod_metric, "British Gas")
                    else:
                        plot_chart_2(product_name, "British Gas", "", filtered_right, view)
                else:
                    st.write("No sentiment data available for British Gas.")
            else:
                st.write("N/A (Selected company is British Gas)")

        # Display AI-generated analysis texts
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🧐 AI Generated Chart Analysis")
        with col2:
            col2a, col2b = st.columns([3, 1])
            with col2a:
                st.markdown("### 🧪 Demographic Analysis", unsafe_allow_html=True)
            with col2b:
                st.markdown("<span class=""button-53"" role=""button"">BETA</span>", unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            overview_row = selected_rows[selected_rows["Aspect"] == "Overview"]
            overview_text = overview_row.iloc[0]["Analysis"] if not overview_row.empty else "Write up..."
            sentiment_difference = int(overview_row.iloc[0]["Sentiment Difference"]) if not overview_row.empty else 0
            if sentiment_difference < 0:
                st.markdown(f"<div class='rounded-block-good'>{overview_text}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='rounded-block-bad'>{overview_text}</div>", unsafe_allow_html=True)
        with col_b:
            demographic_row = selected_rows[selected_rows["Aspect"] == "Demographic"]
            demographic_text = demographic_row.iloc[0]["Analysis"] if not demographic_row.empty else "N/A"
            income_row = selected_rows[selected_rows["Aspect"] == "Income"]
            income_text = income_row.iloc[0]["Analysis"] if not income_row.empty else "N/A"
            st.markdown(
                f"<div class='rounded-block-neutral'>👪 <b>Gender</b>: {demographic_text}<br><br>💸 <b>Income</b>: {income_text}</div>",
                unsafe_allow_html=True
            )
        st.markdown("<hr style='border: 1px solid #0490d7; margin: 20px 0;'>", unsafe_allow_html=True)

    # Render individual aspect tabs if a specific product (not "All") is selected.
    if product_name != "All":
        for idx, aspect in enumerate(ASPECT_CONFIG.aspects_map, start=1):
            with tabs[idx]:
                col_left, col_right = st.columns(2)
                with col_left:
                    plot_aspect_comparison(product_name, aspect, company_name,
                                           f"{ASPECT_CONFIG.aspects_map[aspect]} Compare", "", sa_monthly_data)
                with col_right:
                    aspect_name = ASPECT_CONFIG.aspects_map[aspect].split(' ', 1)[-1]
                    aspect_row = selected_rows[selected_rows["Aspect"] == aspect_name]
                    if not aspect_row.empty:
                        analysis_text = aspect_row.iloc[0]["Analysis"]
                        st.markdown("### 👈 AI Generated Chart Analysis")
                        st.markdown(f"<div class='rounded-block'>{analysis_text}</div>", unsafe_allow_html=True)
                    else:
                        st.write("No analysis available for this aspect.")


def render_ask_alice(inputs: dict, company_name: str, product_name: str) -> None:
    """
    Renders the 'Ask Alice' view for Company Mode.
    Uses the chatbot module to generate AI insights based on user queries and review data.
    Args:
        inputs (dict): User selections from the sidebar (must include 'reviews_data' and 'filter_year').
        company_name (str): The selected company name.
        product_name (str): The selected product name.
    """
    st.markdown("<hr style='border: 1px solid #0490d7; margin: 20px 0;'>", unsafe_allow_html=True)
    query_llm = st.text_area(f"💁‍♀️ **Alice**: *What would you like to know about {company_name}'s {product_name} Insurance?*")
    if st.button("🙋‍♀️ Ask Alice"):
        st.markdown(f"<b>🤔 User: </b>{query_llm}", unsafe_allow_html=True)
        if not query_llm:
            st.markdown("<b>💁‍♀️ Alice</b>: Please enter a question. The query box is currently blank...", unsafe_allow_html=True)
            return

        reviews_data = inputs["reviews_data"]
        filter_year = inputs["filter_year"]
        selected_reviews = chatbot.sample_reviews(reviews_data, product_name, filter_year)
        bg_reviews = None
        if company_name != "British Gas":
            try:
                bg_reviews_data = pd.read_csv("Cleaned Reviews British Gas.csv")
                bg_reviews = chatbot.sample_reviews(bg_reviews_data, product_name, filter_year)
            except Exception as e:
                st.error(f"Error loading British Gas reviews: {e}")
        context = chatbot.prepare_context(selected_reviews, bg_reviews, company_name)
        answer = chatbot.generate_response(context, query_llm, product_name, company_name, bg_reviews)
        st.markdown("<b>💁‍♀️ Alice</b>:", unsafe_allow_html=True)
        st.write(answer)

def render_market_mode(data: dict, inputs: dict) -> None:
    """
    Renders the Market Mode view, showing aggregated sentiment trends and market insights.
    Args:
        data (dict): Loaded data dictionary.
        inputs (dict): User selections from the sidebar.
    """
    product_name = inputs["selected_product"].split(' ', 1)[-1]
    if product_name == "All":
        unique_products = [p for p in data["sa_monthly_data"]["Final Product Category"].unique() if p != "Unknown"]
        tab_names = [PRODUCT_CONFIG.emoji_map.get(product, product) for product in unique_products]
        tabs = st.tabs(tab_names)
        for idx, product in enumerate(unique_products):
            with tabs[idx]:
                plot_product_overall_sentiment(
                    product=product,
                    title=f"{product} Market Sentiment Trends",
                    data=data["sa_monthly_data"]
                )
                year_filter = inputs["filter_year"] if inputs["filter_year"] == "All" else int(inputs["filter_year"])
                prod_strength = data["market_summary_data"][
                    (data["market_summary_data"]["Year"] == year_filter) &
                    (data["market_summary_data"]["Product"] == product) &
                    (data["market_summary_data"]["Aspect"] == "Strength")
                ]
                prod_weakness = data["market_summary_data"][
                    (data["market_summary_data"]["Year"] == year_filter) &
                    (data["market_summary_data"]["Product"] == product) &
                    (data["market_summary_data"]["Aspect"] == "Weakness")
                ]
                prod_improvement = data["market_summary_data"][
                    (data["market_summary_data"]["Year"] == year_filter) &
                    (data["market_summary_data"]["Product"] == product) &
                    (data["market_summary_data"]["Aspect"] == "Improvement")
                ]
                if not prod_strength.empty:
                    st.markdown(f"<div class='rounded-block-good'><div style='text-align:center'><h2>🏆 Our {product} Strengths</h2></div><div style='text-align:left'>{prod_strength.iloc[0]['Analysis']}</div></div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='rounded-block-bad'><div style='text-align:center'><h2>🏮 Our {product} Weaknesses</h2></div><div style='text-align:left'>{prod_weakness.iloc[0]['Analysis']}</div></div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='rounded-block'><div style='text-align:center'><h2>🏗️ {product} Improvement Opportunities</h2></div><div style='text-align:left'>{prod_improvement.iloc[0]['Analysis']}</div></div>", unsafe_allow_html=True)
                else:
                    st.markdown("No market insights available for this product.")
    else:
        st.markdown(f"# Market Sentiment Analytics for {inputs['selected_product']}")
        aspects = list(ASPECT_CONFIG.aspects_map.keys())
        tab_names = [ASPECT_CONFIG.aspects_map[aspect] for aspect in aspects]
        tabs = st.tabs(tab_names)
        for idx, aspect in enumerate(aspects):
            with tabs[idx]:
                plot_chart_3(
                    product=product_name,
                    aspect=aspect,
                    title=f"{ASPECT_CONFIG.aspects_map[aspect]} Market Sentiment Trends",
                    desc="",
                    data=data["sa_monthly_data"]
                )
                year_filter = inputs["filter_year"] if inputs["filter_year"] == "All" else int(inputs["filter_year"])
                filtered_analysis = data["market_summary_data"][
                    (data["market_summary_data"]["Year"] == year_filter) &
                    (data["market_summary_data"]["Product"] == product_name) &
                    (data["market_summary_data"]["Aspect"] == aspect)
                ]
                if not filtered_analysis.empty:
                    analysis_text = filtered_analysis.iloc[0]["Analysis"]
                    st.markdown(f"<div class='rounded-block'>{analysis_text}</div>", unsafe_allow_html=True)
                else:
                    st.markdown("No market insights available for this aspect.")


def render_dev_mode(filter_year: str) -> None:
    # Executes additional data processing functions when Development Mode is enabled.

    data_proc_v01.run(filter_year)

# =============================================================================
# Main Application Workflow
# =============================================================================
def run() -> None:
    # Main function that orchestrates the dashboard workflow.

    load_dotenv()  # Load environment variables

    # Authenticate the user; further execution is halted until login is successful.
    if not auth.login_form():
        return

    # Load all required data.
    data = load_data()

    # Configure sidebar and retrieve user selections.
    inputs = configure_sidebar(data["company_summary_data"])

    # Filter SA Monthly Data by year if a specific year is selected.
    if inputs["filter_year"] != "All":
        try:
            data["sa_monthly_data"] = data["sa_monthly_data"][
                data["sa_monthly_data"]["Year-Month"].dt.year == int(inputs["filter_year"])
            ]
        except Exception as e:
            st.error(f"Error filtering SA Monthly Data by year: {e}")

    # Route to the appropriate view based on the selected mode.
    if inputs["dev_mode"]:
        render_dev_mode(inputs["filter_year"])
    elif inputs["mode"] == "🏢 Company Mode":
        render_company_mode(data, inputs)
    elif inputs["mode"] == "🎍 Market Mode":
        render_market_mode(data, inputs)

run()
