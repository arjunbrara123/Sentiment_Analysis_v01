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

6. Caching & Performance:
   - Uses Streamlit caching (@st.cache_data) to optimize data loading and processing.

Required File Inputs:
-----------------------
1) company_summary_data: AI Generated summaries for 'Company Mode'.
2) sa_monthly_data: Monthly sentiment data - aggregated up by company/market for company/market mode.
3) reviews: Reviews for a specific company (One file per company) - Used for 'Ask Alice' mode only
4) market_summary_data: AI Generated summaries for 'Market Mode'.

Additional Settings:
--------------------
- The environment variable SECRET_HASH is used for secure operations.
- The script tracks the start time to monitor execution duration.
- In development mode, additional data processing functions can be run (using data_proc_v01).

"""

# Import required packages
import os
from datetime import datetime
from charts import *
from openai import OpenAI
import hashlib

#from dotenv import load_dotenv
#load_dotenv()

# Set Streamlit Page Config
st.set_page_config(
    page_title="Competitor Intelligence App 🔬",
    page_icon="images/favicon.ico",
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

# --- Initialize session state for authentication ---
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

# --- Login Form ---
if not st.session_state["authenticated"]:

    col1, col2, col3, col4, col5 = st.columns([3,1,3,1,3])
    with col1:
        st.image("images/bgil_competitors_transparent.png")
    with col5:
        st.write("")
        st.write("")
        st.write("")
        st.image("images/ai_logo_transparent.png")
    with col3:

        st.image("images/bgil-alice-logo1.png")
        with st.form(key="login_form"):
            username = st.text_input("👤 **Username**", placeholder="Your Username Here...", label_visibility="visible", help="💡 Please enter your username here. If you do not have one, please contact the underwriting team for access to this tool.")
            password = st.text_input("🤐 **Password**", type="password", placeholder="Your Password Here...", label_visibility="visible", help="💡 If you have forgotten your password, please contact the underwriting team to reset this.")
            submitted = st.form_submit_button(" 🔐 Login")

        if submitted:
            if username == VALID_USERNAME and hash_password(password) == "a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3":
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

# Input File #1 - Company Mode Summary Text Data
company_summary_data = pd.read_csv("LLM Prod Level Summary v3.csv")
company_list = company_summary_data['Company'].unique().tolist()

# Input File #2 - Monthly sentiment data - agg by company / market for company / market mode respectively
sa_monthly_data = pd.read_csv("LLM SA Monthly Data.csv")

# Input File #3 - Market Mode Summary Text Data
market_summary_data = load_market_summary("LLM Market Summary v3.csv")

# Create a container with a white background for the sidebar controls
with st.sidebar:
    st.sidebar.image("images/company_logo.png")

    with st.expander("🏫 Select Competitor", expanded=True):
        mode = st.radio(
            "Select Mode",
            options=["🎍 Market Mode", "🏢 Company Mode"],
            index=0
        )

        prod_option_list = company_summary_data['Product'].unique().tolist()

        if mode == "🏢 Company Mode":
            selected_company = st.radio(
                "Please Select a Company",
                options=company_list,
                index=0
            )
            company_name = selected_company #.split(' ', 1)[-1]
            prod_option_list = company_summary_data[company_summary_data['Company'] == selected_company]['Product'].unique().tolist()
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
            #dev_pass = st.text_input("Enter password", type="password")
            #if hash_password(dev_pass) == SECRET_HASH:
            dev_flag = True
        else:
            dev_pass = ""

# Main dashboard layout
if mode == "🏢 Company Mode" and not dev_flag:

    st.markdown(f"# {company_name} Analytics: {product_name}")
    if analysis_mode == "🚁 Overview":

        company_tabs = st.tabs(["✈️ Overview"] + [ASPECT_CONFIG.aspects_map[aspect] for aspect in ASPECT_CONFIG.aspects_map])  # aspects_map defined earlier

        # Import the 'Company Mode' rows containing LLM summaries relevant to this company
        selected_rows = company_summary_data[
            (company_summary_data["Company"] == company_name) &
            (company_summary_data["Product"] == product_name)
            ]

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

            # plot_aspect_comparison_boxplot(product_name, "Sentiment Score", company_name,
            #                             f"BoxPlots Sentiment Comparison",  # {aspects_map[aspect]} Compare",
            #                             "", sa_monthly_data)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 🧐 AI Generated Chart Analysis")
            with col2:
                col2a, col2b = st.columns([3,1])
                with col2a:
                    st.markdown("### 🧪 Demographic Analysis", unsafe_allow_html=True)
                with col2b:
                    st.markdown("<span class=""button-53"" role=""button"">BETA</span>", unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                overview_row = selected_rows[selected_rows["Aspect"] == "Overview"]
                overview_text = overview_row.iloc[0]["Analysis"]
                if company_name == "British Gas:":
                    st.markdown(f"<div class='rounded-block'>{"Write up..." if len(overview_text) == 0 else overview_text}</div>", unsafe_allow_html=True)
                else:
                    sentiment_difference = int(overview_row.iloc[0]["Sentiment Difference"])
                    if sentiment_difference < 0:
                        st.markdown(f"<div class='rounded-block-good'>{"Write up..." if len(overview_text) == 0 else overview_text}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(
                            f"<div class='rounded-block-bad'>{"Write up..." if len(overview_text) == 0 else overview_text}</div>",
                            unsafe_allow_html=True)
            with col2:
                demographic_row = selected_rows[selected_rows["Aspect"] == "Demographic"]
                demographic_text = demographic_row.iloc[0]["Analysis"]
                income_row = selected_rows[selected_rows["Aspect"] == "Income"]
                income_text = income_row.iloc[0]["Analysis"]

                st.markdown(
                    f"<div class='rounded-block-neutral'>👪 <b>Gender</b>:  {demographic_text}<br><br>💸 <b>Income</b>:  {income_text}</div>",
                    unsafe_allow_html=True)

            st.markdown("<hr style='border: 1px solid #0490d7; margin: 20px 0;'>", unsafe_allow_html=True)

        # Only create the aspect tabs if a specific product is selected (not "All")
        if product_name != "All":
            # Create a tab for each aspect.
            aspect_tab_names = [ASPECT_CONFIG.aspects_map[aspect] for aspect in ASPECT_CONFIG.aspects_map]
            # iterate over the aspect tabs using index starting from 1.
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
                        aspect_name = ASPECT_CONFIG.aspects_map[aspect].split(' ', 1)[-1]  # adjust if needed
                        aspect_row = selected_rows[selected_rows["Aspect"] == aspect_name]
                        aspect_score = int(filtered_data_left[aspect_name + "_sentiment_score"].mean())
                        if company_name == "British Gas":
                            aspect_difference = ""
                        else:
                            #aspect_difference = int(filtered_data_left[aspect_col + "_sentiment_score"].mean()) - int(filtered_data_right[aspect_col + "_sentiment_score"].mean())
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

                            #st.markdown(analysis_text, unsafe_allow_html=True)
                        else:
                            st.write("No analysis available for this aspect.")

                    # plot_aspect_comparison_hist(product_name, aspect + "_sentiment_score", company_name,
                    #                        f"", #{aspects_map[aspect]} Compare",
                    #                        "", sa_monthly_data)

                        # col1, col2 = st.columns([4,5])
                        # with col1:
                        #     st.write("")
                        # with col2:
                        #     st.metric(company_name, aspect_score, aspect_difference)

    elif analysis_mode == "👽 Emerging Trends":
        st.markdown("## Emerging Customer Sentiment Trends")

    elif analysis_mode == "🙋‍♀️ Ask Alice...":

        st.markdown("<hr style='border: 1px solid #0490d7; margin: 20px 0;'>", unsafe_allow_html=True)

        query_llm = st.text_area(f"💁‍♀️ **Alice**: *What would you like to know about {company_name}'s {product_name} Insurance?*")
        client = OpenAI()

        def sample_reviews(selected_reviews):

            # Filter reviews for product
            filtered_reviews = selected_reviews[selected_reviews["Final Product Category"] == product_name]

            # Filter by year if it’s not "All"
            if filter_year != "All":
                filtered_reviews = filtered_reviews[filtered_reviews["Year-Month"].str[-4:] == str(filter_year)]

            # Set review limit (e.g., due to API constraints)
            REVIEW_LIMIT = 50 #200

            # Sample proportionally from each month
            monthly_counts = reviews_data["Year-Month"].value_counts()
            sample_sizes = (monthly_counts / monthly_counts.sum() * REVIEW_LIMIT).astype(int)
            filtered_reviews = pd.concat([
                filtered_reviews[filtered_reviews["Year-Month"] == month].sample(
                    n=min(sample_sizes[month], len(filtered_reviews[filtered_reviews["Year-Month"] == month])),
                    random_state=42
                )
                for month in monthly_counts.index
            ])

            return filtered_reviews

        selected_reviews = sample_reviews(reviews_data)

        # If the company isn’t British Gas, load British Gas reviews too
        if company_name != "British Gas":
            bg_reviews_data = pd.read_csv("Cleaned Reviews British Gas.csv")
            bg_reviews = sample_reviews(bg_reviews_data)
        else:
            bg_reviews = None

        def prepare_context(selected_reviews, bg_reviews):
            context = f"Selected Company ({selected_company}) Reviews:\n{selected_reviews.to_string(index=False)}\n\n"

            if bg_reviews is not None:
                context += f"British Gas Reviews:\n{bg_reviews.to_string(index=False)}\n\n"

            return context


        def generate_response(context, query, product, selected_company, bg_reviews):
            client = OpenAI()  # Assumes your API key is set up in your environment

            system_prompt = f"""
            You are a commercial strategy expert at British Gas Insurance. 
            Your task is to analyse the provided social media data and reviews to provide well-reasoned, data-driven insights.
            Please provide your analysis in bullet points and only focus on the key ask of the question.
            """

            
          
            if bg_reviews is not None:
                system_prompt += f"\n\nYou are comparing British Gas with {selected_company} for the {product} product line."
            else:
                system_prompt += f"\n\nYou are analyzing reviews for British Gas's {product} product line."

            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",  # Use your preferred model
                    #model="o3-mini",  # Use your preferred model
                    messages=[
                        {"role": "system", "content": system_prompt},
                        #{"role": "developer", "content": system_prompt},
                        {"role": "user", "content": f"Social Media Data: {context}\n\nQuestion: {query}"}
                    ],
                    temperature=0.3,  # Lower for more focused responses
                    max_tokens=1000,  # Reduce to 500-750 for shorter answers if desired
                    #max_completion_tokens=1000,  # Reduce to 500-750 for shorter answers if desired
                    #reasoning_effort='medium',
                    frequency_penalty=1.5,
                    presence_penalty=1.5
                )

                answer = response.choices[0].message.content.strip()
                return answer
            except Exception as e:
                return f"Oops, something went wrong! Error: {str(e)}"


        if st.button("🙋‍♀️ Ask Alice"):
            st.markdown(f"<b>🤔 User: </b>{query_llm}", unsafe_allow_html=True)
            if len(query_llm) == 0:
                st.markdown(f"<b>💁‍♀️ Alice</b>: Please enter a question, the query box is currently blank...",
                            unsafe_allow_html=True)
            else:
                context = prepare_context(selected_reviews, bg_reviews)
                answer = generate_response(context, query_llm, product_name, selected_company, bg_reviews)
                st.markdown(f"<b>💁‍♀️ Alice</b>:", unsafe_allow_html=True)
                st.write(answer)

elif mode == "🎍 Market Mode" and not dev_flag:

    if product_name == "All":
        # Get unique products (excluding "All")
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
                prod_summary = market_summary_data[
                    (market_summary_data["Year"] == year_filter) &
                    (market_summary_data["Product"] == product) &
                    (market_summary_data["Aspect"] == "Summary")
                ]

                if not prod_strength.empty:
                    #analysis_text = filtered_analysis.iloc[0]["Analysis"]
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
                #st.markdown(f"## {aspects_map[aspect]}")

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
    #print("Test 01")
