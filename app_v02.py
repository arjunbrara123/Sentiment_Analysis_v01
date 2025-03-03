"""
Competitor Analytics Dashboard
================================
This Streamlit application provides a comprehensive dashboard for competitor analytics by visualizing sentiment 
and market trends across different products and companies. The app supports two primary modes:
  - "Market Mode": Displays aggregated sentiment trends and market insights across products.
  - "Company Mode": Provides detailed sentiment analysis for a selected competitor, including comparisons with British Gas.

Key Features:
---------------
1. Data Loading & Processing:
   - Loads product-level text summaries (LLM generated) and monthly sentiment analysis data from CSV files.
   - Loads market summary data with insights categorized into Strengths, Weaknesses, Improvement Opportunities, and Growth.
   - Processes and cleans data, including date parsing and percentage value conversion.

2. Page Configuration & Styling:
   - Configures the Streamlit page with title, favicon, and layout settings.
   - Loads custom CSS from "style.css" to apply styling across the dashboard.

3. Sidebar Controls:
   - Displays the company logo.
   - Provides interactive widgets to select competitors, products/markets, analysis mode, and time period settings.
   - Offers toggles for AI settings, including a chatbot ("Alice") that uses OpenAI for data-driven responses.

4. Visualization & Charting:
   - Uses Plotly for dynamic charting to display sentiment trends.
   - Supports different chart types for overall sentiment and aspect-specific breakdowns.
   - Includes a comparison view between the selected company and British Gas when applicable.

5. AI Chatbot Integration:
   - Integrates with OpenAI to process user queries based on social media and review data.
   - Provides actionable insights and recommendations with a conversational tone.
   - Offers different query modes (Overview, Emerging Trends, Drilldown) for tailored responses.

6. Caching & Performance:
   - Uses Streamlit caching (@st.cache_data) to optimize data loading and processing.

Required File Inputs:
-----------------------
1) prod_summary_data: CSV file containing product-level LLM text summaries (both overall and aspect-specific).
2) sa_monthly_data: CSV file with monthly sentiment analysis data at company/product level.
3) reviews: CSV file with review data for chatbot queries.
4) market_summary_data: CSV file with market insights.

Additional Settings:
--------------------
- The environment variable SECRET_HASH is used for secure operations.
- The script tracks the start time to monitor execution duration.
- In development mode, additional data processing functions can be run (using data_proc_v01).

Usage:
------
- Ensure that all required CSV files and assets (e.g., "company_logo.png", "style.css") are in the correct locations.
- Set the necessary environment variables.
- Run the application with Streamlit (e.g., `streamlit run app.py`).

Dependencies:
-------------
- Python standard libraries: os, datetime, hashlib.
- Third-party libraries: pandas, streamlit, plotly, OpenAI API client.

"""

# Import required packages
import os
from datetime import datetime
from charts import *
from openai import OpenAI
import hashlib

# Set Streamlit Page Config
st.set_page_config(
    page_title="🕵️ Competitor Analytics",
    page_icon="favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded",
)
SECRET_HASH = os.getenv("SECRET_HASH")
START_TIME = datetime.now()
import data_proc_v01

# Load and inject CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def hash_password(password: str) -> str:
    """Hashes the password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

# --- Configuration: Set your credentials here ---
VALID_USERNAME = "admin"
VALID_PASSWORD = "password123"

# --- Initialize session state for authentication ---
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

# --- Login Form ---
if not st.session_state["authenticated"]:

    col1, col2, col3, col4, col5 = st.columns([3,1,3,1,3])
    with col1:
        st.image("bgil-competitors.png")
    with col5:
        st.write("")
        st.write("")
        st.write("")
        st.image("ai-logo.jpg")
    with col2:
        st.write("")
    with col4:
        st.write("")
    with col3:
        st.image("bgil-alice-logo1.png")
        with st.form(key="login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")

        if submitted:
            if username == VALID_USERNAME and password == VALID_PASSWORD:
                st.session_state["authenticated"] = True
                st.success("Login successful!")
                st.rerun()  # Refresh the page to remove the login form
            else:
                st.error("Incorrect username or password")
                st.stop()  # Stop execution if login fails
        else:
            st.stop()  # Stop execution until the form is submitted


# Load required data
@st.cache_data
def load_agg_data(input_filepath):

    # Read in the data
    data = pd.read_csv(input_filepath)

    # Check the data in the date column is as expected
    try:

        # Clean Percentage column
        if 'Percentage' in data.columns:
            # Remove '%' and convert to float
            data['Percentage'] = data['Percentage'].str.replace('%', '', regex=False).astype(float)

        data['Year-Month'] = pd.to_datetime(data['Year-Month'], format='%d/%m/%Y', errors='raise')
        data = data.sort_values('Year-Month')

        # If clean, return the data set
        return data

    except ValueError as e:
        print(f"Error encountered: {e}")
        return 0
# Load Market Summary Data
@st.cache_data
def load_market_summary(input_filepath):
    """Loads the market summary data and ensures column names are standardized."""
    data = pd.read_csv(input_filepath)

    # Ensure column names are stripped of leading/trailing spaces
    data.columns = [col.strip() for col in data.columns]

    return data

# Input File #1 - This contains text summaries at a product / company level, both overall and for each aspect
prod_summary_data = pd.read_csv("LLM Prod Level Summary v2x.csv")
company_list = prod_summary_data['Company'].unique().tolist()

# Input File #2 - This contains the monthly sentiment / aspect score at a product / company level
sa_monthly_data = pd.read_csv("LLM SA Monthly Data.csv")

# Input File #3 - Load Market Summary Data
market_summary_data = load_market_summary("LLM Market Summary v2.csv")

# Create a container with a white background for the sidebar controls
with st.sidebar:
    st.sidebar.image("company_logo.png")

    with st.expander("🏫 Select Competitor", expanded=True):
        mode = st.radio(
            "Select Mode",
            options=["🎍 Market Mode", "🏢 Company Mode"],
            index=0
        )

        prod_option_list = prod_summary_data['Product'].unique().tolist()

        if mode == "🏢 Company Mode":
            selected_company = st.radio(
                "Please Select a Company",
                options=company_list,
                index=0
            )
            company_name = selected_company #.split(' ', 1)[-1]
            prod_option_list = prod_summary_data[prod_summary_data['Company'] == selected_company]['Product'].unique().tolist()
        else:
            selected_company = ""

    with st.expander(("🎁 Select Product" if mode == "🏢 Company Mode" else "🛖 Select Market"), expanded=True):
        if 'All' not in prod_option_list and mode != "🏢 Company Mode":
            prod_option_list.insert(0, "All")
        selected_product = st.radio(
            "Please Select a " + (" Product" if mode == "🏢 Company Mode" else " Market"),
            options = [PRODUCT_CONFIG.emoji_map.get(product, product) for product in prod_option_list],
            index=0
        )

    if mode == "🏢 Company Mode":
        input_Raw_Comments_Text_data = "Cleaned Reviews " + company_name + ".csv"
        reviews_data = pd.read_csv(input_Raw_Comments_Text_data)

    product_name = selected_product.split(' ', 1)[-1]
    if product_name != "All" and mode == "🏢 Company Mode":
        analysis_mode_options = ["🚁 Overview", "👽 Emerging Trends", "🙋‍♀️ Ask Alice..."]
    else:
        analysis_mode_options = ["🚁 Overview", "👽 Emerging Trends"]

    if mode == "🏢 Company Mode":
        with st.expander("🧩 Analysis Mode", expanded=True):
            analysis_mode = st.radio(
                "Please Select Analysis",
                options=analysis_mode_options,
                index=0
            )
    else:
        analysis_mode = "🚁 Overview"

    with st.expander("⌚ Time Period Settings", expanded=False):
        filter_year = st.selectbox(
            "Pick a Year to Display",
            ("All", "2021", "2022", "2023", "2024", "2025"),
        index=4)

    if filter_year != "All":
        sa_monthly_data = sa_monthly_data[sa_monthly_data['Year-Month'].str[-4:] == str(filter_year)]

    with st.expander("🧠 AI Settings", expanded=False):
        tts_flag = st.toggle("Alice Reads Responses Aloud", value=False)
        dev_mode = st.toggle("Dev Mode", value=False)
        dev_flag = False
        if dev_mode:
            dev_flag = True
        else:
            dev_pass = ""

# Main dashboard layout
if mode == "🏢 Company Mode" and not dev_flag:

    #st.markdown(f"# {company_name} - Competitor Analytics")
    st.markdown(f"# {company_name} Analytics: {product_name}")

    if analysis_mode == "🚁 Overview":

        #company_tabs = st.tabs(["✈️ Overview"] + [aspects_map[aspect] for aspect in aspects_map])  # aspects_map defined earlier
        company_tabs = st.tabs(["✈️ Overview"] + [ASPECT_CONFIG.aspects_map[aspect] for aspect in ASPECT_CONFIG.aspects_map])  # aspects_map defined earlier

        with company_tabs[0]:  # "Overview" tab
            # When toggled on, it represents "Aspect View". When off, it represents "Sentiment View".
            chart_toggle = st.toggle("Split Sentiment Into Aspects", value=True,
                                     help="Toggle between Aspect View (all aspects) and Sentiment View (overall sentiment)")
            if chart_toggle and product_name == "All":
                prod_metric = st.selectbox("Pick an Aspect to breakdown:",
                             ("Overall Sentiment Score", "Appointment Scheduling", "Customer Service", "Response Speed", "Engineer Experience", "Solution Quality", "Value For Money")
                             )
            view = "aspect" if chart_toggle else "sentiment"

            # Create two columns for side-by-side display
            col1, col2 = st.columns(2)

            # Left Column: Selected Company's Summary
            with col1:

                # Plot sentiment graph for the selected company and product
                if "all" not in product_name.lower():
                    filtered_data_left = sa_monthly_data[
                        (sa_monthly_data["Company"].str.contains(company_name)) &
                        (sa_monthly_data["Final Product Category"].str.contains(product_name))
                    ]
                else:
                    filtered_data_left = sa_monthly_data[
                        (sa_monthly_data["Company"].str.contains(company_name))
                    ]
                if not filtered_data_left.empty:
                    if "all" == product_name.lower():
                        plot_chart_all_products(product_name, f"{company_name} Sentiment", "", filtered_data_left, prod_metric, company_name)
                    else:
                        plot_chart_2(product_name, company_name, "", filtered_data_left, view)
                else:
                    st.write("No sentiment data available for the selected company and product.")

            # Right Column: British Gas Summary (or blank if British Gas is selected)
            with col2:
                if "British Gas" not in selected_company:
                    # Filter sentiment data for British Gas and the same product
                    if "all" not in product_name.lower():
                        filtered_data_right = sa_monthly_data[
                            (sa_monthly_data["Company"].str.contains("British Gas")) &
                            (sa_monthly_data["Final Product Category"].str.contains(product_name))
                            ]
                    else:
                        filtered_data_right = sa_monthly_data[
                            (sa_monthly_data["Company"].str.contains("British Gas"))
                            ]
                    # Plot sentiment graph for British Gas
                    if not filtered_data_right.empty:
                        if "all" == product_name.lower():
                            plot_chart_all_products(product_name, f"British Gas Sentiment", "", filtered_data_right,
                                                    prod_metric, "British Gas")
                        else:
                            plot_chart_2(product_name, f"British Gas", "", filtered_data_right, view)
                    else:
                        st.write("No sentiment data available for British Gas.")
                else:
                    st.write("N/A (Selected company is British Gas)")

            # plot_aspect_comparison_hist(product_name, "Sentiment Score", company_name,
            #                             f"",
            #                             "", sa_monthly_data)

            st.markdown("<hr style='border: 1px solid #0490d7; margin: 20px 0;'>", unsafe_allow_html=True)

        # Only create the aspect tabs if a specific product is selected (not "All")
        if product_name != "All":
            # Now, create a tab for each aspect.
            aspect_tab_names = [ASPECT_CONFIG.aspects_map[aspect] for aspect in ASPECT_CONFIG.aspects_map]
            # Since we already have our tabs container with "Overview" + aspect names,
            for idx, aspect in enumerate(ASPECT_CONFIG.aspects_map, start=1):
                with company_tabs[idx]:
                    # Create a two-column layout
                    col1, col2 = st.columns(2)
                    with col1:
                        # Plot the aspect comparison chart.
                        plot_aspect_comparison(product_name, aspect, company_name,
                                               f"{ASPECT_CONFIG.aspects_map[aspect]} Compare",
                                               "", sa_monthly_data)
                    with col2:
                        # Retrieve and display the analysis text for this aspect.
                        selected_rows = prod_summary_data[
                            (prod_summary_data["Company"] == company_name) &
                            (prod_summary_data["Product"] == product_name)
                            ]
                        aspect_name = ASPECT_CONFIG.aspects_map[aspect].split(' ', 1)[-1]  # adjust if needed
                        aspect_row = selected_rows[selected_rows["Aspect"] == aspect_name]
                        aspect_score = int(filtered_data_left[aspect_name + "_sentiment_score"].mean())
                        if company_name == "British Gas":
                            aspect_difference = ""
                        else:
                            aspect_row = selected_rows[selected_rows["Aspect"] == aspect_name]
                            if not aspect_row.empty:
                                sentiment_difference = aspect_row.iloc[0]["Sentiment Difference"]
                                aspect_difference = -sentiment_difference if not pd.isna(sentiment_difference) else ""
                            else:
                                aspect_difference = ""
                        if not aspect_row.empty:
                            analysis_text = aspect_row.iloc[0]["Analysis"]
                            st.markdown("### 👈 AI Generated Chart Analysis")
                            st.markdown(f"<div class='rounded-block'>{analysis_text}</div>", unsafe_allow_html=True)
                        else:
                            st.write("No analysis available for this aspect.")

                    # plot_aspect_comparison_hist(product_name, aspect + "_sentiment_score", company_name,
                    #                        f"", #{aspects_map[aspect]} Compare",
                    #                        "", sa_monthly_data)

    elif analysis_mode == "👽 Emerging Trends":
        st.markdown("## Emerging Customer Sentiment Trends")

    elif analysis_mode == "🙋‍♀️ Ask Alice...":

        # st.markdown("### 💁‍♀️ Hi there, please let me know how to best respond to your query...")
        # col1, col2 = st.columns([3,2])
        # with col1:
        #     alice_mode = st.radio(
        #         "Select Social Media / Online Data to base responses on:",
        #         (f"👩‍✈️✈️ **Overview Mode**: *Let's Look Across Your Whole Selected Year*: {filter_year}", "🧑‍🚀🛰️ **Emerging Risk Mode**: *Explore Emerging Trends in Recent Months*", "👩‍🚒🚒 **Drilldown Mode**: *Focus in on just one, very Specific Month...*"),
        #         index=0,
        #     )
        #     alice_selected = alice_mode.split(':', 1)[0].replace("**", "")
        #     if alice_selected == "👩‍🚒🚒 Drilldown Mode":
        #         filter_llm_month = st.selectbox(
        #             "Please select a specific month...",
        #             reviews_data["Year-Month"].unique()
        #         )
        # with col2:
        #     st.markdown(f"""<b><u>ALICE Query Settings</b></u><br>
        #     - <b>🏭 Selected Company</b>: {selected_company}<br>
        #     - <b>🎁 Selected Product Line</b>: {selected_product}<br>
        #     - <b>🧩 Analysis Mode</b>: {alice_mode.split(':', 1)[0].replace("**", "")}<br>
        #     - <b>⏲️ Time Period</b>: {pd.to_datetime(filter_llm_month, dayfirst=True).strftime("%B %Y") if alice_selected == "👩‍🚒🚒 Drilldown Mode" else filter_year}<br>
        #     """,unsafe_allow_html=True)

        alice_selected = "👩‍✈️✈️ Overview Mode"

        st.markdown("<hr style='border: 1px solid #0490d7; margin: 20px 0;'>", unsafe_allow_html=True)

        query_llm = st.text_area("💬 Enter your specific query here...")
        client = OpenAI()

        # Filter reviews for product
        filtered_reviews = reviews_data[reviews_data["Final Product Category"] == product_name]

        # Set review limit (e.g., due to API constraints)
        REVIEW_LIMIT = 1000

        # Filter reviews based on sampling method
        if alice_selected == "👩‍🚒🚒 Drilldown Mode":
            filtered_reviews = reviews_data[reviews_data["Year-Month"] == filter_llm_month]
        elif alice_selected == "👩‍✈️✈️ Overview Mode":
            # Sample proportionally from each month
            monthly_counts = reviews_data["Year-Month"].value_counts()
            sample_sizes = (monthly_counts / monthly_counts.sum() * REVIEW_LIMIT).astype(int)
            filtered_reviews = pd.concat([
                reviews_data[reviews_data["Year-Month"] == month].sample(
                    n=min(sample_sizes[month], len(reviews_data[reviews_data["Year-Month"] == month])),
                    random_state=42
                )
                for month in monthly_counts.index
            ])
        elif alice_selected == "🧑‍🚀🛰️ Emerging Risk Mode":
            # Sort by date and take the most recent
            filtered_reviews = reviews_data.sort_values(by="Date", ascending=False).head(REVIEW_LIMIT)

        OPENAI_SYSTEM_PROMPT = f"""
        You are a customer experience expert at {selected_company} Insurance. 
        You speak in a warm, friendly, and conversational tone, occasionally adding light humor or puns to keep engagement high. 
        Your task is to analyse the company's social media data to provide well-reasoned, data-driven insights. 
        Focus on the biggest themes or commonalities between a large volume of comments as opposed to relying or referencing any one specific comment, unless there is one comment which is interesting and representative of how a very large number of comments feel.
        Remember the audience is {selected_company} senior members, so try and phrase things that doesn't show the company in an overly negative light or in a way that might offend anyone.
        When you answer a question, do the following:

        1. Summarise the key data points from the social media context that support your answer, accounting for what product line it is too.
        2. Provide 2-3 concrete, actionable recommendations for the business, explaining briefly why these recommendations follow from the data and given product line.
        3. Use a short concluding paragraph or bullet points to tie everything together. 
        4. If the data is insufficient to answer the question, or the question is not relevant to {selected_company}, politely say so and explain why.

        Always keep the conversation focused on {selected_company} Insurance, and ensure the user knows all your recommendations are rooted in the data you've been given.
        """
        
        def llm_inference(query, context):
            try:
                # st.write(context)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": (
                            OPENAI_SYSTEM_PROMPT
                        )},
                        {"role": "user",
                         "content": (
                             f"Product Line: {product_name}\n\n"
                             f"Social Media Data: {context}\n\n"
                             f"Question: {query}\n\n"
                             "Please reference the social media data directly in your answer, "
                             "and offer data-driven actionable steps British Gas Insurance can take."
                         )
                         }
                    ],
                    temperature=0.5,
                    max_tokens=1000,
                    frequency_penalty=1.5,
                    presence_penalty=1.5,
                )

                # Parse the response content
                answer = response.choices[0].message.content.strip()
                return answer

            except Exception as e:
                return f"Inferencing Failed, Error: {str(e)}"


        if st.button("🙋‍♀️ Ask Alice"):
            st.markdown(f"<b>🤔 User: </b>{query_llm}", unsafe_allow_html=True)
            if len(query_llm) == 0:
                st.markdown(f"<b>💁‍♀️ Alice</b>: Please enter a question, the query box is currently blank...",
                            unsafe_allow_html=True)
            else:
                answer = llm_inference(query_llm, filtered_reviews)
                st.markdown(f"<b>💁‍♀️ Alice</b>:", unsafe_allow_html=True)
                st.write(answer)
                elapsed_time = datetime.now() - START_TIME
                hours, remainder = divmod(elapsed_time.total_seconds(), 3600)
                minutes, seconds = divmod(remainder, 60)

elif mode == "🎍 Market Mode" and not dev_flag:

    if product_name == "All":
        unique_products = [p for p in sa_monthly_data["Final Product Category"].unique() if p != "Unknown"]
        tab_names = [PRODUCT_CONFIG.emoji_map.get(product, product) for product in unique_products]
        tabs = st.tabs(tab_names)

        for idx, product in enumerate(unique_products):
            with tabs[idx]:
                # Plot overall sentiment for this product
                plot_product_overall_sentiment(
                    product=product,
                    title=f"{product} Market Sentiment Trends",
                    data=sa_monthly_data
                )

                # Display analysis text from CSV
                year_filter = filter_year if filter_year == "All" else int(filter_year)
                filtered_analysis = market_summary_data[
                    (market_summary_data["Year"] == year_filter) &
                    (market_summary_data["Product"] == product) &
                    (market_summary_data["Aspect"] == "Overall")
                ]
                prod_strength = market_summary_data[
                    (market_summary_data["Year"] == year_filter) &
                    (market_summary_data["Product"] == product) &
                    (market_summary_data["Aspect"] == "Strength")
                ]
                prod_weakness = market_summary_data[
                    (market_summary_data["Year"] == year_filter) &
                    (market_summary_data["Product"] == product) &
                    (market_summary_data["Aspect"] == "Weakness")
                ]
                prod_improvement = market_summary_data[
                    (market_summary_data["Year"] == year_filter) &
                    (market_summary_data["Product"] == product) &
                    (market_summary_data["Aspect"] == "Improvement")
                ]
                prod_growth = market_summary_data[
                    (market_summary_data["Year"] == year_filter) &
                    (market_summary_data["Product"] == product) &
                    (market_summary_data["Aspect"] == "Growth")
                ]

                if not prod_strength.empty:
                    st.markdown(f"<div class='rounded-block-good'><div style='text-align:center'><h2>🏆 Our {product} Strengths</h2><div><div style='text-align:left'>{prod_strength.iloc[0]["Analysis"]}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='rounded-block-bad'><div style='text-align:center'><h2>🏮 Our {product} Weaknesses</h2><div><div style='text-align:left'>{prod_weakness.iloc[0]["Analysis"]}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='rounded-block'><div style='text-align:center'><h2>🏗️ {product} Improvement Opportunities</h2><div><div style='text-align:left'>{prod_improvement.iloc[0]["Analysis"]}</div>", unsafe_allow_html=True)
                    #st.markdown(f"<div class='rounded-block-neutral'><div style='text-align:center'><h2>🌱 Growing the {product} Customer Base</h2><div><div style='text-align:left'>{prod_growth.iloc[0]["Analysis"]}</div>", unsafe_allow_html=True)
                else:
                    st.markdown("No market insights available for this product.")

    else:
        st.markdown(f"# Market Sentiment Analytics for {selected_product}")

        # Dynamically generate tabs using the aspects_map dictionary - 6 aspects
        aspects = list(ASPECT_CONFIG.aspects_map.keys())
        tab_names = [ASPECT_CONFIG.aspects_map[aspect] for aspect in aspects]
        tabs = st.tabs(tab_names)

        # Loop over the aspects and fill each tab with its respective chart and a placeholder
        for idx, aspect in enumerate(aspects):
            with tabs[idx]:
                # Call the new plotting function; the title can be dynamically built
                plot_chart_3(
                    product=product_name,
                    aspect=aspect,
                    title=f"{ASPECT_CONFIG.aspects_map[aspect]} Market Sentiment Trends",
                    desc="",
                    data=sa_monthly_data
                )

                # Retrieve the corresponding analysis from the market summary
                filtered_analysis = market_summary_data[
                    (market_summary_data["Year"] == (filter_year if filter_year == "All" else int(filter_year))) &  # Only use data from that year
                    (market_summary_data["Product"] == product_name) &  # Match product
                    (market_summary_data["Aspect"] == aspect)  # Match aspect
                ]

                # Display the analysis if available
                if not filtered_analysis.empty:
                    analysis_text = filtered_analysis.iloc[0]["Analysis"]
                    st.markdown(f"<div class='rounded-block'>{analysis_text}", unsafe_allow_html=True)
                else:
                    st.markdown("No market insights available for this aspect.")

elif dev_flag:
    data_proc_v01.run(filter_year)
