"""
Competitor Intelligence Dashboard
============================================

Overview:
---------
This dashboard provides interactive insights into competitor sentiment and market trends using Streamlit.
It leverages AI-driven analyses and interactive Plotly charts to help users explore two primary modes:
    • Company Mode: Detailed sentiment analysis and comparisons for a selected competitor, including an AI chatbot ("Alice") for insights.
    • Market Mode: Aggregated market sentiment trends and insights across products/markets.
    • Emerging Risk Mode: Tracking emerging risks such as regulatory risk or adverse social trends

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
import numpy as np
# =============================================================================
# Dashboard Setup
# =============================================================================

#Import required python packages
import streamlit as st
from streamlit_modal import Modal
from dotenv import load_dotenv

import actuarial

load_dotenv()  # Load environment variables

import plotly.express as px

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

# Authenticate the user; further execution is halted until login is successful.
if not auth.login_form():
    st.stop()

# ==========================================================================================
# Data Loading and Preprocessing Functions: Functions for loading CSV files and preparing the data
# Function load_data: Aggregates loading of all 3 CSVs files, and return them into a dictionary with keys for easy access.
# ==========================================================================================

@st.cache_data(show_spinner=True)
def load_data(data_prefix) -> dict:
    # Purpose: Loads all required CSV data into a dictionary for easy access.
    # Returns: dict: Contains dataframes for company summaries, sentiment data, and market summaries.

    data = {}

    # Input Filepath #1 - Company Mode - AI Gen Summary Text Data
    try:
        data["company_summary_data"] = pd.read_csv(data_prefix + company_summary_data_filepath)
    except Exception as e:
        st.error(f"Error loading company summary data: {e}")
        data["company_summary_data"] = pd.DataFrame()

    # Input File #2 - Market Mode - AI Gen Summary Text Data
    data["market_summary_data"] = pd.read_csv(data_prefix + market_summary_data_filepath)

    # Input File #3 - Monthly sentiment data - agg by company / market for company / market mode respectively
    try:
        sa_df = pd.read_csv(data_prefix + sa_monthly_data_filepath)
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
# This section is responsible for rendering the 4-5 sidebar widgets and capturing user inputs.
# User selections including mode, company, product/market, analysis mode, and filter settings.
# =============================================================================

inputs = {}
with st.sidebar: # Create left-hand side bar with various navigation controls

    # British Gas Logo at the Top of the sidebar
    st.sidebar.image("images/company_logo.png")

    dept_select = st.segmented_control(label="🤹 Select Department", options=["🧑‍🔧 S&S", "⚡ Energy"], default="🧑‍🔧 S&S")
    if dept_select == "⚡ Energy":
        data_prefix = "energy/"
    else:
        data_prefix = "services/"

    # Load all required data.
    data = load_data(data_prefix)
    company_summary_data = data["company_summary_data"]
    market_summary_data = data["market_summary_data"]

    # User Navigation Box 1 of 5
    with st.expander("🏫 Select Competitor", expanded=True):
        inputs["mode"] = st.radio("Select Mode", options=["🎍 Market Intel", "🏢 Company Intel", "🥷 Emerging Risk"], index=0)
        prod_option_list = company_summary_data['Product'].unique().tolist()
        if inputs["mode"] == "🏢 Company Intel":
            inputs["selected_company"] = st.radio("Please Select a Company",
                                                   options=company_summary_data['Company'].unique().tolist(),
                                                   index=0)
            prod_option_list = company_summary_data[
                company_summary_data["Company"] == inputs["selected_company"]
            ]["Product"].unique().tolist()
        else:
            inputs["selected_company"] = None

    # User Navigation Box 2 of 5
    if inputs["mode"] == "🎍 Market Intel" or inputs["mode"] ==  "🏢 Company Intel":
        with st.expander(f"🎁 Select {'Product' if inputs['mode'] == '🏢 Company Mode' else 'Market'}", expanded=True):
            if 'All' not in prod_option_list and inputs["mode"] != "🏢 Company Intel":
                prod_option_list.insert(0, "All")
            inputs["selected_product"] = st.radio(
                f"Please Select a {'Product' if inputs['mode'] == '🏢 Company Mode' else 'Market'}",
                options=[PRODUCT_CONFIG.emoji_map.get(product, product) for product in prod_option_list],
                index=0
            )

    if inputs["mode"] == "🏢 Company Intel":
        # Load review data for the selected company (one file per company)
        reviews_filepath = f"{data_prefix}Cleaned Reviews {inputs['selected_company']}.csv"
        try:
            inputs["reviews_data"] = pd.read_csv(reviews_filepath)
        except Exception as e:
            st.error(f"Error loading reviews data for {inputs['selected_company']}: {e}")
            inputs["reviews_data"] = pd.DataFrame()
    else:
        inputs["reviews_data"] = None

    # User Navigation Box 3 of 5
    if inputs["mode"] == "🏢 Company Intel":
        prod_name = inputs["selected_product"].split(' ', 1)[-1]
    #     analysis_options = ["🚁 Overview", "👽 Emerging Trends", "🙋‍♀️ Ask Alice..."] if prod_name != "All" else ["🚁 Overview", "👽 Emerging Trends"]
    #     with st.expander("🧩 Analysis Mode", expanded=True):
    #         inputs["analysis_mode"] = st.radio("Please Select Analysis", options=analysis_options, index=0)
    # else:
    #     inputs["analysis_mode"] = "🚁 Overview"

    # User Navigation Box 4 of 5
    with st.expander("⌚ Time Period Settings", expanded=False):
        inputs["filter_year"] = st.selectbox("Pick a Year to Display",
                                               ("All", "2021", "2022", "2023", "2024", "2025"),
                                               index=4)

    # User Navigation Box 5 of 5
    with st.expander("🧠 AI Settings", expanded=False):
        inputs["tts_flag"] = st.toggle("Alice Reads Responses Aloud", value=False)
        inputs["dev_mode"] = st.toggle("Dev Mode", value=False)

    st.markdown('''<style>
    div[data-testid="stModal"] div[role="dialog"] {
        width: 70%;
    }
    </style>''', unsafe_allow_html=True)

    @st.dialog(f"🙎‍♀️ Ask Alice about {inputs["selected_company"]}")
    def show_company_chatbot():
        query_llm = st.text_input(f"💬 Alice: *Please ask me about {inputs["selected_company"]} {prod_name}:* 😁")
        if st.button("🙎‍♀️ Ask Alice"):
            with st.spinner("🤔 Thinking..."):
                reviews_data = inputs["reviews_data"]
                filter_year = inputs["filter_year"]
                selected_reviews = chatbot.sample_reviews(reviews_data, prod_name, filter_year)
                bg_reviews = None
                if inputs["selected_company"] != "British Gas":
                    try:
                        bg_reviews_data = pd.read_csv(f"{data_prefix}Cleaned Reviews British Gas.csv")
                        bg_reviews = chatbot.sample_reviews(bg_reviews_data, prod_name, filter_year)
                    except Exception as e:
                        st.error(f"Error loading British Gas reviews: {e}")
                context = chatbot.prepare_context(selected_reviews, bg_reviews, inputs["selected_company"])
                answer = chatbot.generate_response(context, query_llm, prod_name, inputs["selected_company"], bg_reviews)
                st.markdown("<b>💁‍♀️ Alice</b>:", unsafe_allow_html=True)
                st.write(answer)

    if inputs["mode"] == "🏢 Company Intel":
        if st.button("🙎‍♀️ Ask Alice", key="alice_open"):
            show_company_chatbot()

    def show_emerging_risk_chatbot(emerging_risk: str):
        title = f"🙎‍♀️ Ask Alice about Emerging Risk 🧗"
        @st.dialog(title)
        def show_company_chatbot():
            st.text_input(f"💬 Alice: *Please ask me about any emerging risk:* 😁")
            if st.button("🙎‍♀️ Ask Alice"):
                st.write("Text")

# Filter SA Monthly Data by year if a specific year is selected.
if inputs["filter_year"] != "All":
    try:
        data["sa_monthly_data"] = data["sa_monthly_data"][
            data["sa_monthly_data"]["Year-Month"].dt.year == int(inputs["filter_year"])
        ]
    except Exception as e:
        st.error(f"Error filtering SA Monthly Data by year: {e}")

# Load relevant monthly data
sa_monthly_data = data["sa_monthly_data"]

# ==========================================================================================
# Rendering Functions:
# This section contains functions responsible for displaying various dashboard views based on user inputs.
# ------------------------------------------------------------------------------------------
# 1. render_company_mode: Renders competitor-specific views with detailed sentiment charts and AI analysis.
# 2. render_company_overview: Renders the "Overview" tab for Company Mode, displaying charts and analysis texts.
# 3. render_ask_alice: Provides the interactive "Ask Alice" chatbot interface for query-driven insights in company mode.
# 4. render_market_mode: Renders market-wide sentiment trends and insights based on product/market selection.
# 5. render_dev_mode: Executes additional processing functions when Development Mode is enabled.
# 6. render_emerging_risk:
# ==========================================================================================
def render_company_mode() -> None:
    # Renders the Company Mode view including the header, detailed overview, aspect tabs, and the "Ask Alice" view if selected.

    company_name = inputs["selected_company"]
    product_name = inputs["selected_product"].split(' ', 1)[-1]
    # analysis_mode = inputs["analysis_mode"]

    # Header: Display company logo and title.
    col1, col2 = st.columns([1, 7])
    with col1:
        st.image(f"images/{company_name.lower()} logo.png", width=100)
    with col2:
        st.markdown(f"# {company_name} Analytics: {product_name}")

    # Filter company summary data for the selected company and product.
    selected_rows = company_summary_data[
        (company_summary_data["Company"] == company_name) &
        (company_summary_data["Product"] == product_name)
    ]
    # figure out which aspects we actually have rows for
    available_aspects = [
        asp for asp in ASPECT_CONFIG.aspects_map
        if asp in selected_rows["Aspect"].unique()
    ]
#
#     # Route to the proper analysis mode.
#     if analysis_mode == "🚁 Overview":
#         render_company_overview(selected_rows, company_name, product_name)
#     elif analysis_mode == "👽 Emerging Trends":
#         st.markdown("## Emerging Customer Sentiment Trends")
#     elif analysis_mode == "🙋‍♀️ Ask Alice...":
#         render_ask_alice(company_name, product_name)
#
#
# def render_company_overview(selected_rows: pd.DataFrame, company_name: str, product_name: str) -> None:
#     # Renders the "Overview" tab for Company Mode, displaying sentiment charts and analysis texts.

    tabs_labels = ["✈️ Overview"] + [
        ASPECT_CONFIG.aspects_map[asp] for asp in available_aspects
    ]
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
                f"<div class='rounded-block-neutral'>"
                f"👪 <b>Gender</b>: {demographic_text}<br><br>"
                f"💸 <b>Income</b>: {income_text}</div>",
                unsafe_allow_html=True
            )
        st.markdown("<hr style='border: 1px solid #0490d7; margin: 20px 0;'>", unsafe_allow_html=True)

    # Render individual aspect tabs if a specific product (not "All") is selected.
    if product_name != "All":
        for idx, aspect in enumerate(available_aspects, start=1): #ASPECT_CONFIG.aspects_map, start=1):
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

def render_emerging_risk():
    """
    Renders the Emerging Risk mode using compute_emerging_risk from actuarial.py.
    Flags spikes in:
      • Regulatory Risk  (contract/Ofgem keywords)
      • Social Risk      (green/eco keywords)
      • Brand Risk       (switching‑intent keywords)
    Compares the last 7 days vs the prior 90‑day baseline, and shows:
      – a gauge (pct of baseline)
      – a 12‑week sparkline of raw counts
      – top 3 sub‑risks per category
    """

    # ─── Helpers ────────────────────────────────────────────────────────────
    @st.cache_data
    def load_reviews() -> pd.DataFrame:
        df = pd.read_csv("annotated_reviews.csv", parse_dates=["Date"])
        return df

    def style_chart(fig: go.Figure) -> go.Figure:
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="white", plot_bgcolor="white",
            margin=dict(l=0, r=0, t=30, b=0),
            hovermode="x unified"
        )
        return fig

    # ─── Load & global filters ──────────────────────────────────────────────
    df = load_reviews()

    # ─── Tabs ───────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "✈️ Overview",
        "🌱 Net Zero / Heat Pumps",
        "👩‍⚖️ Regulatory Risk",
        "🎭 Brand Risk",
        "🥊 Social Trends Risk"
    ])

    # ─── Tab 1: Emerging Risk Overview ───────────────────────────────────────
    with tabs[0]:

        #st.markdown("<h3 style='text-align:center'> 🥷 Emerging Risk Real-Time Monitoring </h3>", unsafe_allow_html=True)

        # Define your keyword→weight maps
        reg_keywords   = {"ofgem":1, "contract":1, "terms":0.5, "conditions":0.5}
        green_keywords = {"green":1, "renewable":1, "eco":1, "carbon":0.5, "solar":0.5}
        sw_keywords    = {"will switch":1, "looking to switch":1,
                          "switched to":1, "moved to":1, "consider switching":1}

        # Compute emerging risk for each
        reg   = actuarial.compute_emerging_risk(df, reg_keywords, date_col="Date", review_col="Review", sentiment_col="sent_vader_n")
        social= actuarial.compute_emerging_risk(df, green_keywords)
        brand = actuarial.compute_emerging_risk(df, sw_keywords)

        df['green_flag']  = (df['kw_green'] > 0).astype(int)
        df['switch_flag'] = df['switching'].isin(['Intent','Outcome']).astype(int)

        # Turn rate_ratio → percent of baseline
        metrics = [
            ("🧑‍⚖️ Regulatory Risk", reg, "reg_ref_flag"),
            ("🌱 Social Risk", social, "green_flag"),
            ("🤡 Brand Risk", brand, "switch_flag"),
        ]

        # Sub‑risk cards
        card_cols = st.columns(3)
        for (col, (title, res, flag_col)) in zip(card_cols, metrics):
            # compute our two KPIs
            base_avg = round(float(res["base_count"])*7/90, 1)
            recent_avg = res["recent_count"]
            change_pct = (res["rate_ratio"]-1) * 100
            drivers = res["drivers"]  # list of {keyword, recent_count, ratio,...}
            sent_diff = (res["recent_sentiment"] - res["base_sentiment"]) * 100

            # pick gradient color: green if below 100%, red if above
            gradient = (
                "linear-gradient(135deg,#28a745,#85e085)"  # green–light
                if change_pct <= -10
                else "linear-gradient(135deg,#dc3545,#e68585)"  # red–light
            )

            with col:

                fill = min(abs(change_pct), 100)
                x_pos = 50 if change_pct >= 0 else 50 - fill

                s_fill = min(abs(sent_diff), 100)
                s_x = 50 if sent_diff >= 0 else 50 - s_fill

                st.markdown(f"""
                <div style="
                    background: {gradient};
                    padding: 20px;
                    margin-bottom: 9px;
                    border-radius: 10px;
                    color: white;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.7);
                ">
                  <h4 style="margin:0 0 10px; text-align:center">{title}</h3>

                  <!-- Mentions change -->
                  <p style="font-size:32px; text-align: {'right' if change_pct > 0 else 'left'}; margin:5px 0;">{change_pct:+.0f}%</p>
                  <div style="position:relative; margin:4px 0;">
                    <span style="position:absolute; left:0; top:-1em;">🙊←</span>
                    <span style="position:absolute; right:0; top:-1em;">→📣</span>
                      <span style="position:absolute; left:50%; transform:translateX(-50%); top:-1em;">🔻</span>
                    <svg width="100%" height="12">
                      <rect width="100%" height="12" fill="white" opacity="0.2"/>
                      <line x1="50%" x2="50%" y1="0" y2="12" stroke="white" stroke-width="1"/>
                      <rect x="{x_pos:.1f}%" width="{fill:.1f}%"
                            height="12" fill="white"/>
                      <line x1="50%" x2="50%" y1="0" y2="12" stroke="navy" stroke-width="2"/>
                    </svg>
                  </div>
                  <p style="font-size:14px; margin:8px 0 16px; text-align: center;">
                    Baseline Chatter: {base_avg}, Recent Avg: {recent_avg}
                  </p>
                
                  <!-- Sentiment shift (same idea) -->
                  <p style="font-size:28px; text-align: {'right' if sent_diff > 0 else 'left'}; margin:5px 0;">{sent_diff:+.1f} pts</p>
                  <div style="position:relative; margin:4px 0;">
                    <span style="position:absolute; left:0; top:-1em;">🤬←</span>
                    <span style="position:absolute; right:0; top:-1em;">→😁</span>
                      <span style="position:absolute; left:50%; transform:translateX(-50%); top:-1em;">🔻</span>
                    <svg width="100%" height="12">
                      <rect width="100%" height="12" fill="white" opacity="0.2"/>
                      <line x1="50%" x2="50%" y1="0" y2="12" stroke="white" stroke-width="1"/>
                      <rect x="{s_x:.1f}%" width="{s_fill:.1f}%" height="12" fill="white"/>
                      <line x1="50%" x2="50%" y1="0" y2="12" stroke="navy" stroke-width="2"/>
                    </svg>
                  </div>
                  <p style="font-size:14px; margin:8px 0 0; text-align: center;">
                    Baseline Sentiment: {(res["base_sentiment"]*100).round(1)}, Recent Avg: {(res["recent_sentiment"]*100).round(1)}
                  </p>
                </div>
                """, unsafe_allow_html=True)

                # --- Sparkline (last 12 weeks) ---
                weekly = (
                    df
                    .groupby(pd.Grouper(key="Date", freq="W-MON"))[flag_col]
                    .sum()
                    .iloc[-12:]  # last 12 weeks
                    .reset_index(name="Count")
                )
                spark_fig = go.Figure(go.Scatter(
                    x=weekly["Date"], y=weekly["Count"],
                    mode="lines", line={'width': 2, 'color': "#0490d7"}
                ))
                # Add subtle megaphone at top‑right and quiet emoji at bottom‑right
                spark_fig.add_annotation(
                    text="↑ 📢",
                    xref="paper", yref="paper",
                    x=0.07, y=0.99,
                    showarrow=False,
                    font=dict(size=16)
                )
                spark_fig.add_annotation(
                    text="↓ 🙊",
                    xref="paper", yref="paper",
                    x=0.07, y=0.01,
                    showarrow=False,
                    font=dict(size=16)
                )
                spark_fig.update_layout(
                    height=80,
                    margin=dict(t=0, b=0, l=0, r=0),
                    xaxis_visible=False,
                    yaxis_visible=False
                )
                st.plotly_chart(style_chart(spark_fig), use_container_width=True)

                # --- Sentiment Sparkline (last 12 weeks) ---
                sentiment_weekly = (
                    df
                    .groupby(pd.Grouper(key="Date", freq="W-MON"))["sent_vader_n"]
                    .mean()
                    .iloc[-12:]
                    .reset_index(name="AvgSentiment")
                )

                sent_fig = go.Figure(go.Scatter(
                    x=sentiment_weekly["Date"],
                    y=sentiment_weekly["AvgSentiment"].round(3)*100,
                    mode="lines",
                    line={'width': 2, 'color': "#ffa500"},  # e.g. orange for sentiment
                ))

                # Add sad face at bottom‑right and happy face at top‑right
                sent_fig.add_annotation(
                    text="↓ 🤬",
                    xref="paper", yref="paper",
                    x=0.07, y=0.01,
                    showarrow=False,
                    font=dict(size=16)
                )
                sent_fig.add_annotation(
                    text="↑ 😁",
                    xref="paper", yref="paper",
                    x=0.07, y=0.99,
                    showarrow=False,
                    font=dict(size=16)
                )

                sent_fig.update_layout(
                    height=80,
                    margin=dict(t=0, b=0, l=0, r=0),
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False)
                )

                st.plotly_chart(style_chart(sent_fig), use_container_width=True, key=f"spark_sent_{title}")

                st.markdown("---")

                st.markdown(f"<h4 style='text-align:center'> {title[:-5]} Drivers</h4>", unsafe_allow_html=True)
                # now your driver list tiles below, same style as above
                drv_count = 0
                for drv in drivers[:3]:  # top 3
                    drv_count += 1
                    kw = drv["keyword"].title()
                    last_cnt = drv["recent_count"]
                    pct_change = drv["ratio"] * 100
                    sent_change = drv["sent_diff"] * 100
                    bar_color = "#dc3545" if pct_change > 150 else "#28a745"

                    if drv_count == 1:
                        kw_title = f"🥇 {kw}"
                    elif drv_count == 2:
                        kw_title = f"🥈 {kw}"
                    elif drv_count == 3:
                        kw_title = f"🥉 {kw}"

                    st.markdown(f"""
                    <div style="
                        background: {bar_color};
                        padding: 12px;
                        border-radius: 8px;
                        color: white;
                        margin-top: 12px;
                    ">
                      <strong style="font-size:18px;">{kw_title}</strong><br>
                      This week: {last_cnt} mentions {"📣" if pct_change > 150 else "💬"}<br>
                      <small>~{pct_change:.0f}% of baseline, Sentiment Shift: {sent_change:+.1f} pts <br></small>
                      <small>🧠 AI Gen: This topic drives the spike because...</small>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("---")
                st.markdown(f"<h4 style='text-align:center'> 🧠 AI Gen Actionables </h4>", unsafe_allow_html=True)
                st.write("Based on online chatter, the root cause of the above flagged issues are...")
                st.write("Recomended actions the business can take right now to address the root cause of these issues are..")

    # ─── Tab 2: Heat Pumps ─────────────────────────────────────────────────────
    with tabs[1]:
        st.title("🌱 Net Zero / Heat Pumps")
        st.markdown("Use green‑chatter trends to time sustainable product launches.")
        green_ts = df.groupby(df["Date"].dt.to_period("M"))["kw_green"].mean().reset_index()
        fig_hp = go.Figure(go.Scatter(
            x=green_ts["Date"].dt.to_timestamp(), y=green_ts["kw_green"], mode="lines"
        ))
        fig_hp.update_layout(title="Avg Green Mentions by Month",
                             yaxis_title="Avg # green keywords")
        st.plotly_chart(style_chart(fig_hp), use_container_width=True)

    # ─── Tab 3: Regulatory Risk ─────────────────────────────────────────────────
    with tabs[2]:
        st.title("👩‍⚖️ Regulatory Risk")
        st.metric("This week vs baseline", f"{reg['rate_ratio']*100:.1f}%", delta=f"{(reg['rate_ratio']-1)*100:.1f}%")
        df_reg_ts = (
            df[df["Review"].str.contains("|".join(reg_keywords.keys()), case=False)]
              .set_index("Date")
              .resample("W-MON")
              .size()
              .reset_index(name="Count")
        )
        fig_reg = go.Figure(go.Scatter(x=df_reg_ts["Date"], y=df_reg_ts["Count"], mode="lines+markers"))
        fig_reg.update_layout(title="% Weekly Regulatory Mentions")
        st.plotly_chart(style_chart(fig_reg), use_container_width=True)

    # ─── Tab 4: Brand Risk ───────────────────────────────────────────────────────
    with tabs[3]:
        st.title("🎭 Brand Risk")
        st.metric("This week vs baseline", f"{brand['rate_ratio']*100:.1f}%", delta=f"{(brand['rate_ratio']-1)*100:.1f}%")
        df_brand_ts = (
            df[df["Review"].str.contains("|".join(sw_keywords.keys()), case=False)]
              .set_index("Date")
              .resample("W-MON")
              .size()
              .reset_index(name="Count")
        )
        fig_br = go.Figure(go.Scatter(x=df_brand_ts["Date"], y=df_brand_ts["Count"], mode="lines+markers"))
        fig_br.update_layout(title="% Weekly Switching Mentions")
        st.plotly_chart(style_chart(fig_br), use_container_width=True)

    # ─── Tab 5: Social Trends Risk ──────────────────────────────────────────────
    with tabs[4]:
        st.title("🥊 Social Trends Risk")
        st.metric("This week vs baseline", f"{social['rate_ratio']*100:.1f}%", delta=f"{(social['rate_ratio']-1)*100:.1f}%")
        df_soc_ts = (
            df[df["Review"].str.contains("|".join(green_keywords.keys()), case=False)]
              .set_index("Date")
              .resample("W-MON")
              .size()
              .reset_index(name="Count")
        )
        fig_st = go.Figure(go.Scatter(x=df_soc_ts["Date"], y=df_soc_ts["Count"], mode="lines+markers"))
        fig_st.update_layout(title="% Weekly Green Mentions")
        st.plotly_chart(style_chart(fig_st), use_container_width=True)

def render_emerging_risk_2() -> None:
    # Renders the "Overview" tab for Company Mode, displaying sentiment charts and analysis texts.

    tabs_labels = ["✈️ Overview", "🌱 Net Zero / Heat Pumps", "👩‍⚖️ Regulatory Risk", "🎭 Brand Risk", "🥊 Social Trends Risk"]
    tabs = st.tabs(tabs_labels)
    with tabs[0]:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("Reg Risk KPI")
        with col2:
            st.write("Brand Risk KPI")
        with col3:
            st.write("Social Risk KPI")

    with tabs[1]:
        st.write("CHART 1: Avg Green mentions over time... this shows social trends around green solutions\n\n")
        st.write("CHART 2: Green mentions by provider... this shows which providers are selling more / less green solutions")
        st.write("CHART 3: Sentiment by Green Topic over time... Shows social trends of what people do or don't like about green solutions like heatpumps, includes a paragraph talking about interesting emerging trends over the last month or so, or maybe last week")

    with tabs[2]:
        col1, col2 = st.columns([1,3])
        with col1:
            st.write("Online market chatter around regulatory risk has spiked in the last week compared to the average for the last 3 months")
        with col2:
            st.write("CHART: % of chatter with regulatory risk references over time")

    with tabs[3]:
        st.write("This tab tracks risks to our brand, such as big accounts with a lot of followers saying something about us online, or something like an important / government offical etc...")

    with tabs[4]:
        st.write("This tab tracks emerging risks in terms of customer complaints for our business, customer complaints general to the industry, and any new emerging topic for which chatter has recently increased a lot over the last week compared to the last 3 month average")

def render_ask_alice(company_name: str, product_name: str) -> None:
    # Renders the 'Ask Alice' view for Company Mode - AI generated insights based on user queries and review data.

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

def render_market_mode() -> None:
    # Renders the Market Mode view, showing aggregated sentiment trends and market insights.

    product_name = inputs["selected_product"].split(' ', 1)[-1]
    if product_name == "All":
        unique_products = [prod for prod in sa_monthly_data["Final Product Category"].unique() if prod != "Unknown"]
        tab_names = [PRODUCT_CONFIG.emoji_map.get(product, product) for product in unique_products]
        tabs = st.tabs(tab_names)
        for idx, product in enumerate(unique_products):
            with tabs[idx]:
                plot_product_overall_sentiment(
                    product=product,
                    title=f"{product} Market Sentiment Trends",
                    data=sa_monthly_data
                )
                year_filter = inputs["filter_year"] if inputs["filter_year"] == "All" else int(inputs["filter_year"])
                selected_rows = market_summary_data[
                    (market_summary_data["Year"] == year_filter) &
                    (market_summary_data["Product"] == product)
                    ]
                prod_strength = selected_rows[market_summary_data["Aspect"] == "Strength"]
                prod_weakness = selected_rows[market_summary_data["Aspect"] == "Weakness"]
                prod_improvement = selected_rows[market_summary_data["Aspect"] == "Improvement"]

                if not prod_strength.empty:
                    st.markdown(f"<div class='rounded-block-good'><div style='text-align:center'><h2>🏆 Our {product} Strengths</h2></div><div style='text-align:left'>{prod_strength.iloc[0]['Analysis']}</div></div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='rounded-block-bad'><div style='text-align:center'><h2>🏮 Our {product} Weaknesses</h2></div><div style='text-align:left'>{prod_weakness.iloc[0]['Analysis']}</div></div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='rounded-block'><div style='text-align:center'><h2>🏗️ {product} Improvement Opportunities</h2></div><div style='text-align:left'>{prod_improvement.iloc[0]['Analysis']}</div></div>", unsafe_allow_html=True)
                else:
                    st.markdown("No market insights available for this product.")
    else:
        st.markdown(f"# Market Sentiment Analytics for {inputs['selected_product']}")
        #aspects = list(ASPECT_CONFIG.aspects_map.keys())
        available_aspects = [
            asp for asp in ASPECT_CONFIG.aspects_map
            if asp in market_summary_data["Aspect"].unique()
        ]
        tab_names = [ASPECT_CONFIG.aspects_map[aspect] for aspect in available_aspects] #aspects]
        tabs = st.tabs(tab_names)
        for idx, aspect in enumerate(available_aspects): #aspects):
            with tabs[idx]:
                plot_chart_3(
                    product=product_name,
                    aspect=aspect,
                    title=f"{ASPECT_CONFIG.aspects_map[aspect]} Market Sentiment Trends",
                    desc="",
                    data=sa_monthly_data
                )
                year_filter = inputs["filter_year"] if inputs["filter_year"] == "All" else int(inputs["filter_year"])
                selected_rows = market_summary_data[
                    (market_summary_data["Year"] == year_filter) &
                    (market_summary_data["Product"] == product_name)
                ]
                filtered_analysis = selected_rows[market_summary_data["Aspect"] == aspect]
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

    # Route to the appropriate view based on the selected mode.
    if inputs["dev_mode"]:
        render_dev_mode(inputs["filter_year"])
    elif inputs["mode"] == "🏢 Company Intel":
        render_company_mode()
    elif inputs["mode"] == "🎍 Market Intel":
        render_market_mode()
    elif inputs["mode"] == "🥷 Emerging Risk":
        render_emerging_risk()

run()
