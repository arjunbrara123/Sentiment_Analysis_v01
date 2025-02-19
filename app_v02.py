# Import required packages
import os
from datetime import datetime

from charts import *
from openai import OpenAI
import hashlib

from dotenv import load_dotenv
load_dotenv()

# Required File Inputs
# 1) prod_summary_data - LLM product-level text summaries at company / product level, at an overall and aspect level.
# 2) sa_monthly_data   - LLM sentiment analysis monthly data, also at company / product level, at an overall and aspect level.
# 3) reviews           - Feeding into the chatbot to answer questions based from

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

# Input File #1 - This contains text summaries at a product / company level, both overall and for each aspect
prod_summary_data = pd.read_csv("LLM Prod Level Summary.csv")
company_list = prod_summary_data['Company'].unique().tolist()

# Input File #2 - This contains the monthly sentiment / aspect score at a product / company level
sa_monthly_data = pd.read_csv("LLM SA Monthly Data.csv")

# Create a container with a white background for the sidebar controls
with st.sidebar:
    st.sidebar.image("company_logo.png")

    with st.expander("🏫 Select Competitor / Product", expanded=False):
        mode = st.radio(
            "Select Mode",
            options=["🏢 Company Mode", "🎍 Market Mode"],
            index=0
        )

        prod_option_list = prod_summary_data['Product'].unique().tolist()

        if mode == "🏢 Company Mode":
            selected_company = st.radio(
                "Please Select a Company",
                options=company_list,
                index=0
            )
            prod_option_list = prod_summary_data[prod_summary_data['Company'] == selected_company]['Product'].unique().tolist()
        else:
            selected_company = ""

        selected_product = st.radio(
            "Please Select a Product",
            options = prod_option_list,
            index=0
        )

    if mode == "🏢 Company Mode":
        company_name = selected_company.split(' ', 1)[-1]
        input_Raw_Comments_Text_data = "Cleaned Reviews " + company_name + ".csv"
        reviews_data = pd.read_csv(input_Raw_Comments_Text_data)

    product_name = selected_product.split(' ', 1)[-1]
    if product_name == "All":
        analysis_mode_options = ["🚁 Overview", "👽 Emerging Trends"]
    else:
        analysis_mode_options = ["🚁 Overview", "👽 Emerging Trends", "🙋‍♀️ Ask Alice..."]

    with st.expander("🧩 Analysis Mode", expanded=False):
        analysis_mode = st.radio(
            "Please Select Analysis",
            options=analysis_mode_options,
            index=0
        )

    with st.expander("🕜 Time Period Settings", expanded=False):
        filter_year = st.selectbox(
            "Pick a Year to Display",
            ("All", "2021", "2022", "2023", "2024", "2025"),
        index=4)
        print(f"Year Select: {filter_year}")

    if filter_year != "All":
        sa_monthly_data = sa_monthly_data[sa_monthly_data['Year-Month'].str[-4:] == str(filter_year)]

    with st.expander("⚙️ Settings", expanded=False):
        tts_flag = st.toggle("Alice Reads Responses Aloud", value=False)
        dev_mode = st.toggle("Dev Mode", value=False)
        dev_flag = False
        if dev_mode:
            dev_pass = st.text_input("Enter password", type="password")
            if hash_password(dev_pass) == SECRET_HASH:
                dev_flag = True
        else:
            dev_pass = ""

# Main dashboard layout
if mode == "🏢 Company Mode" and not dev_flag:

    #st.markdown(f"# {company_name} - Competitor Analytics")
    st.markdown(f"# Competitor Analytics")
    st.markdown(f"**Selected Product Line:** {selected_product}")

    if analysis_mode == "🚁 Overview":

        # When toggled on, it represents "Aspect View". When off, it represents "Sentiment View".
        chart_toggle = st.toggle("Split Sentiment Into Aspects", value=True,
                                 help="Toggle between Aspect View (all aspects) and Sentiment View (overall sentiment)")

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
                    plot_chart_2(product_name, f"British Gas", "", filtered_data_right, view)
                else:
                    st.write("No sentiment data available for British Gas.")

            else:
                st.write("N/A (Selected company is British Gas)")

        # Add divider
        st.markdown("<hr style='border: 1px solid #0490d7; margin: 20px 0;'>", unsafe_allow_html=True)

        st.markdown("## Aspect Details")

        selected_row = prod_summary_data[
            (prod_summary_data["Company"] == selected_company) &
            (prod_summary_data["Product"] == selected_product)
        ]

        if not selected_row.empty:
            # Get the first (and assumed only) row for the selected company and product.
            row = selected_row.iloc[0]
            st.write("Company: " + company_name)

            # Use aspects_map directly to loop through the aspects.
            for aspect_col, aspect_display in aspects_map.items():
                with st.expander(aspect_display):
                    plot_aspect_comparison(product_name, aspect_col, company_name,
                                           f"{aspect_display} Comparison", "",
                                           sa_monthly_data)
                    col1, col2 = st.columns([1, 5])
                    with col1:
                        aspect_score = int(filtered_data_left[aspect_col + "_sentiment_score"].mean())
                        if company_name == "British Gas":
                            aspect_difference = ""
                        else:
                            aspect_difference = int(filtered_data_left[aspect_col + "_sentiment_score"].mean()) - int(filtered_data_right[aspect_col + "_sentiment_score"].mean())
                        st.metric(company_name, aspect_score, aspect_difference)
                    with col2:
                        st.markdown(row[aspect_col], unsafe_allow_html=True)


        else:
            st.write("No aspect details available.")

    elif analysis_mode == "👽 Emerging Trends":
        st.markdown("## Emerging Customer Sentiment Trends")

    elif analysis_mode == "🙋‍♀️ Ask Alice...":

        st.markdown(
            f"<div style='font-size: 35px; text-align: center; color: #012973; margin-bottom: 10px;'><b>🔬 Ask Alice a more specific sentiment analysis question</b><br></div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<div style='text-align: center; color: #012973; margin-bottom: 15px;'>This selection allows you to ask specific questions focused on a specific month, and specific product group, which Alice will answer based on the sentiment and online chatter data available for the specific month and product group.</div>",
            unsafe_allow_html=True
        )
        col1, col2 = st.columns([2, 2])
        with col1:
            filter_llm_month = st.selectbox(
                "Please select a specific month...",
                reviews_data["Year-Month"].unique()
            )
        with col2:
            filter_llm_type = st.selectbox(
                "Please select a specific customer segment",
                ("Everyone", "Tbc...    "),
            )
        query_llm = st.text_area("Enter your more specific query here...")
        client = OpenAI()
        #convo = []

        # Debugging: Print unique values of Year-Month and Final Product Category
        print("Unique Year-Month values:", reviews_data["Year-Month"].unique())
        print(filter_llm_month)
        # Filter relevant data
        filtered_reviews = reviews_data[
            reviews_data["Year-Month"] == filter_llm_month]
        filtered_reviews = filtered_reviews[
                               filtered_reviews["Final Product Category"] == product_name][:1000].values.tolist()
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
                    model="gpt-4o-mini",  # "o1-mini", #"gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": (
                            OPENAI_SYSTEM_PROMPT
                        )},
                        {"role": "user",
                         "content": (
                             f"Month: {filter_llm_month}\n\n"
                             f"Product Line: {product_name}\n\n"
                             f"Social Media Data: {context}\n\n"
                             f"Question: {query}\n\n"
                             "Please reference the social media data directly in your answer, "
                             "and offer actionable steps British Gas Insurance can take."
                         )
                         }
                    ],
                    temperature=0.5,
                    max_tokens=1000,
                    # max_completion_tokens=250,
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
                pd.DataFrame([[datetime.now(), f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}", query_llm,
                               filter_llm_month, product_name, "Group Placeholder", answer]]).to_csv(
                    'LLM_Runlog.csv', mode='a', index=False, header=False)


elif mode == "🎍 Market Mode" and not dev_flag:
    st.markdown(f"# Market Sentiment Analytics for {selected_product}")

    # Dynamically generate tabs using the aspects_map dictionary - 6 aspects
    aspects = list(aspects_map.keys())
    tab_names = [aspects_map[aspect] for aspect in aspects]
    tabs = st.tabs(tab_names)

    # Loop over the aspects and fill each tab with its respective chart and a placeholder
    for idx, aspect in enumerate(aspects):
        with tabs[idx]:
            st.markdown(f"## {aspects_map[aspect]}")
            # Call the new plotting function; the title can be dynamically built
            plot_chart_3(
                product=product_name,
                aspect=aspect,
                title=f"{aspects_map[aspect]} Sentiment Trends for {selected_product}",
                desc="",
                data=sa_monthly_data
            )
            # Placeholder for additional summary information for this aspect
            st.write("Placeholder for additional summary information...")

elif dev_flag:
    data_proc_v01_openai.run()
# Import required packages
import os
from datetime import datetime

from charts import *
from openai import OpenAI
import hashlib

from dotenv import load_dotenv
load_dotenv()

# Required File Inputs
# 1) prod_summary_data - LLM product-level text summaries at company / product level, at an overall and aspect level.
# 2) sa_monthly_data   - LLM sentiment analysis monthly data, also at company / product level, at an overall and aspect level.
# 3) reviews           - Feeding into the chatbot to answer questions based from

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

# Input File #1 - This contains text summaries at a product / company level, both overall and for each aspect
prod_summary_data = pd.read_csv("LLM Prod Level Summary.csv")
company_list = prod_summary_data['Company'].unique().tolist()

# Input File #2 - This contains the monthly sentiment / aspect score at a product / company level
sa_monthly_data = pd.read_csv("LLM SA Monthly Data.csv")

# Create a container with a white background for the sidebar controls
with st.sidebar:
    st.sidebar.image("company_logo.png")

    with st.expander("🏫 Select Competitor / Product", expanded=False):
        mode = st.radio(
            "Select Mode",
            options=["🏢 Company Mode", "🎍 Market Mode"],
            index=0
        )

        prod_option_list = prod_summary_data['Product'].unique().tolist()

        if mode == "🏢 Company Mode":
            selected_company = st.radio(
                "Please Select a Company",
                options=company_list,
                index=0
            )
            prod_option_list = prod_summary_data[prod_summary_data['Company'] == selected_company]['Product'].unique().tolist()
        else:
            selected_company = ""

        selected_product = st.radio(
            "Please Select a Product",
            options = prod_option_list,
            index=0
        )

    if mode == "🏢 Company Mode":
        company_name = selected_company.split(' ', 1)[-1]
        input_Raw_Comments_Text_data = "Cleaned Reviews " + company_name + ".csv"
        reviews_data = pd.read_csv(input_Raw_Comments_Text_data)

    product_name = selected_product.split(' ', 1)[-1]
    if product_name == "All":
        analysis_mode_options = ["🚁 Overview", "👽 Emerging Trends"]
    else:
        analysis_mode_options = ["🚁 Overview", "👽 Emerging Trends", "🙋‍♀️ Ask Alice..."]

    with st.expander("🧩 Analysis Mode", expanded=False):
        analysis_mode = st.radio(
            "Please Select Analysis",
            options=analysis_mode_options,
            index=0
        )

    with st.expander("🕜 Time Period Settings", expanded=False):
        filter_year = st.selectbox(
            "Pick a Year to Display",
            ("All", "2021", "2022", "2023", "2024", "2025"),
        index=4)
        print(f"Year Select: {filter_year}")

    if filter_year != "All":
        sa_monthly_data = sa_monthly_data[sa_monthly_data['Year-Month'].str[-4:] == str(filter_year)]

    with st.expander("⚙️ Settings", expanded=False):
        tts_flag = st.toggle("Alice Reads Responses Aloud", value=False)
        dev_mode = st.toggle("Dev Mode", value=False)
        dev_flag = False
        if dev_mode:
            dev_pass = st.text_input("Enter password", type="password")
            if hash_password(dev_pass) == SECRET_HASH:
                dev_flag = True
        else:
            dev_pass = ""

# Main dashboard layout
if mode == "🏢 Company Mode" and not dev_flag:

    #st.markdown(f"# {company_name} - Competitor Analytics")
    st.markdown(f"# Competitor Analytics")
    st.markdown(f"**Selected Product Line:** {selected_product}")

    if analysis_mode == "🚁 Overview":

        # When toggled on, it represents "Aspect View". When off, it represents "Sentiment View".
        chart_toggle = st.toggle("Split Sentiment Into Aspects", value=True,
                                 help="Toggle between Aspect View (all aspects) and Sentiment View (overall sentiment)")

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
                    plot_chart_2(product_name, f"British Gas", "", filtered_data_right, view)
                else:
                    st.write("No sentiment data available for British Gas.")

            else:
                st.write("N/A (Selected company is British Gas)")

        # Add divider
        st.markdown("<hr style='border: 1px solid #0490d7; margin: 20px 0;'>", unsafe_allow_html=True)

        st.markdown("## Aspect Details")

        selected_row = prod_summary_data[
            (prod_summary_data["Company"] == selected_company) &
            (prod_summary_data["Product"] == selected_product)
        ]

        if not selected_row.empty:
            # Get the first (and assumed only) row for the selected company and product.
            row = selected_row.iloc[0]
            st.write("Company: " + company_name)

            # Use aspects_map directly to loop through the aspects.
            for aspect_col, aspect_display in aspects_map.items():
                with st.expander(aspect_display):
                    plot_aspect_comparison(product_name, aspect_col, company_name,
                                           f"{aspect_display} Comparison", "",
                                           sa_monthly_data)
                    col1, col2 = st.columns([1, 5])
                    with col1:
                        aspect_score = int(filtered_data_left[aspect_col + "_sentiment_score"].mean())
                        if company_name == "British Gas":
                            aspect_difference = ""
                        else:
                            aspect_difference = int(filtered_data_left[aspect_col + "_sentiment_score"].mean()) - int(filtered_data_right[aspect_col + "_sentiment_score"].mean())
                        st.metric(company_name, aspect_score, aspect_difference)
                    with col2:
                        st.markdown(row[aspect_col], unsafe_allow_html=True)


        else:
            st.write("No aspect details available.")

    elif analysis_mode == "👽 Emerging Trends":
        st.markdown("## Emerging Customer Sentiment Trends")

    elif analysis_mode == "🙋‍♀️ Ask Alice...":

        st.markdown(
            f"<div style='font-size: 35px; text-align: center; color: #012973; margin-bottom: 10px;'><b>🔬 Ask Alice a more specific sentiment analysis question</b><br></div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<div style='text-align: center; color: #012973; margin-bottom: 15px;'>This selection allows you to ask specific questions focused on a specific month, and specific product group, which Alice will answer based on the sentiment and online chatter data available for the specific month and product group.</div>",
            unsafe_allow_html=True
        )
        col1, col2 = st.columns([2, 2])
        with col1:
            filter_llm_month = st.selectbox(
                "Please select a specific month...",
                reviews_data["Year-Month"].unique()
            )
        with col2:
            filter_llm_type = st.selectbox(
                "Please select a specific customer segment",
                ("Everyone", "Tbc...    "),
            )
        query_llm = st.text_area("Enter your more specific query here...")
        client = OpenAI()
        #convo = []

        # Debugging: Print unique values of Year-Month and Final Product Category
        print("Unique Year-Month values:", reviews_data["Year-Month"].unique())
        print(filter_llm_month)
        # Filter relevant data
        filtered_reviews = reviews_data[
            reviews_data["Year-Month"] == filter_llm_month]
        filtered_reviews = filtered_reviews[
                               filtered_reviews["Final Product Category"] == product_name][:1000].values.tolist()
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
                    model="gpt-4o-mini",  # "o1-mini", #"gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": (
                            OPENAI_SYSTEM_PROMPT
                        )},
                        {"role": "user",
                         "content": (
                             f"Month: {filter_llm_month}\n\n"
                             f"Product Line: {product_name}\n\n"
                             f"Social Media Data: {context}\n\n"
                             f"Question: {query}\n\n"
                             "Please reference the social media data directly in your answer, "
                             "and offer actionable steps British Gas Insurance can take."
                         )
                         }
                    ],
                    temperature=0.5,
                    max_tokens=1000,
                    # max_completion_tokens=250,
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
                pd.DataFrame([[datetime.now(), f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}", query_llm,
                               filter_llm_month, product_name, "Group Placeholder", answer]]).to_csv(
                    'LLM_Runlog.csv', mode='a', index=False, header=False)


elif mode == "🎍 Market Mode" and not dev_flag:
    st.markdown(f"# Market Sentiment Analytics for {selected_product}")

    # Dynamically generate tabs using the aspects_map dictionary - 6 aspects
    aspects = list(aspects_map.keys())
    tab_names = [aspects_map[aspect] for aspect in aspects]
    tabs = st.tabs(tab_names)

    # Loop over the aspects and fill each tab with its respective chart and a placeholder
    for idx, aspect in enumerate(aspects):
        with tabs[idx]:
            st.markdown(f"## {aspects_map[aspect]}")
            # Call the new plotting function; the title can be dynamically built
            plot_chart_3(
                product=product_name,
                aspect=aspect,
                title=f"{aspects_map[aspect]} Sentiment Trends for {selected_product}",
                desc="",
                data=sa_monthly_data
            )
            # Placeholder for additional summary information for this aspect
            st.write("Placeholder for additional summary information...")

elif dev_flag:
    data_proc_v01.run()
