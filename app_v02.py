    # Import required packages
from charts import *
from getpass import getuser
from sklearn.linear_model import LinearRegression
from openai import OpenAI
import plotly.express as px
from datetime import datetime

#from dotenv import load_dotenv
#load_dotenv()

# Set Streamlit Page Config
st.set_page_config(
    page_title="Sentiment Analytics Dashboard",
    page_icon="favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded",
)

START_TIME = datetime.now()

# Load and inject CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background: linear-gradient(to left, #0490d7, white);
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load required data
#@st.cache_data
def load_agg_data(input_filepath):

    # Read in the data
    data = pd.read_csv(input_filepath)

    # Check the data in the date column is as expected
    try:
        data['Year-Month'] = pd.to_datetime(data['Year-Month'], format='%d/%m/%Y', errors='raise')
        #data['Year-Month'] = data['Year-Month'].dt.to_timestamp()
        data = data.sort_values('Year-Month')

        # If clean, return the data set
        return data

    except ValueError as e:
        print(f"Error encountered: {e}")
        return 0

# Create a container with a white background for the sidebar controls
with st.sidebar:
    st.sidebar.image("company_logo.png")
    st.sidebar.markdown(f"**Current User:** {getuser()}")

    with st.expander("🏫 Select Company", expanded=False):
        selected_company = st.radio(
            "Please Select a Company",
            options=["British Gas", "HomeServe"],
            index=0
        )

    input_LLM_Tabs_Summary_data = "LLM Summary Data " + selected_company + ".csv"
    input_SA_Monthly_data = "SA Monthly Data " + selected_company + ".csv"
    input_Raw_Comments_Text_data = "Cleaned Reviews " + selected_company + ".csv"

    data = load_agg_data(input_SA_Monthly_data)
    tab_data = pd.read_csv(input_LLM_Tabs_Summary_data)
    reviews_data = pd.read_csv(input_Raw_Comments_Text_data)
    insight_list = tab_data["Type"].unique().tolist()


    with st.expander("🕵️ Insight Mode", expanded=True):
        lob_filter = st.radio(
            "Please Select an Insights Mode",
            options=["🚁 Overview"] + insight_list + ["💁‍♀️ Ask Alice..."],
            index=0,
            help="Select an option to change the grouping of tabs displayed on the main window"
        )

    with st.expander("🕜 Time Period Settings", expanded=False):

        show_projections = st.checkbox("Show Future Projections", value=False,
                           help="Enable this to view predicted sentiment trends based on historical data.")
        projection_end_date = st.date_input("Set Projection End Date (Default 3 Months)",
                                value=data["Year-Month"].max().date() + pd.DateOffset(months=3),
                                min_value=data["Year-Month"].max().date(),
                                )
        proj_method = st.selectbox(
            "How would you like to project values forward?",
            ("Holt-Winters (Recommended)", "Linear Regression - All Data", "Linear Regression - Last 3M", "Linear Regression - Last 6M") #, "Spline", "Time Series")
        )

        min_date = pd.to_datetime(data['Year-Month'].min(), format='%Y%m').date()
        max_date = projection_end_date if show_projections else data["Year-Month"].max().date()

        # Generate future projections dynamically
        if show_projections:

            future_data_list = []  # Store projected data for each product type

            # Prepare numeric columns for regression
            numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()

            #!!!! Remove once moving to monthly csv
            #grouping_cols = ["Year-Month", "Final Product Category"]
            #monthly_data = data[grouping_cols + numeric_columns].groupby(grouping_cols).mean().reset_index()
            #data = monthly_data

            # Label encode the Year-Month column for regression
            data["Month_Index"] = (data["Year-Month"] - data["Year-Month"].min()).dt.days // 30

            # Iterate over each product category
            for product_type in data["Final Product Category"].unique():

                product_data = data[data["Final Product Category"] == product_type]
                product_data = product_data.reset_index(drop=True)

                # Apply the filtering logic based on `proj_method`
                if proj_method == "Linear Regression - Last 3M":
                    recent_months = product_data["Year-Month"].nlargest(3)  # Get the most recent 3 months
                    product_data = product_data[product_data["Year-Month"].isin(recent_months)]
                elif proj_method == "Linear Regression - Last 6M":
                    recent_months = product_data["Year-Month"].nlargest(6)  # Get the most recent 6 months
                    product_data = product_data[product_data["Year-Month"].isin(recent_months)]

                # Prepare future dates
                future_dates = pd.date_range(
                    start=product_data["Year-Month"].max() + pd.DateOffset(months=1),
                    end=max_date,
                    freq="MS"  # Monthly intervals
                )

                # Create a future Month Index for projections
                last_month_index = product_data["Month_Index"].max()
                future_month_indexes = np.arange(last_month_index + 1, last_month_index + len(future_dates) + 1)

                # Perform linear regression for each numeric column
                for col in numeric_columns:
                    # Prepare the model and data
                    X = product_data[["Month_Index"]]
                    y = product_data[col]
                    if proj_method == "Holt-Winters (Recommended)":
                        model = ExponentialSmoothing(product_data[col], trend="add", seasonal=None, damped_trend=True)
                        fitted_model = model.fit()
                        future_values = fitted_model.forecast(len(future_dates))
                    else:
                        model = LinearRegression().fit(X, y)
                        future_values = model.predict(future_month_indexes.reshape(-1, 1))

                    # Create a DataFrame for projections
                    future_data = pd.DataFrame({
                        "Year-Month": future_dates,
                        col: future_values,
                        "Final Product Category": product_type,
                        "Projected": 1
                    })

                    # Append to the list
                    future_data_list.append(future_data)

            # Combine future projections with the original data
            future_data_combined = pd.concat(future_data_list, ignore_index=True)
            combined_data = pd.concat([data, future_data_combined], ignore_index=True)

        else:
            combined_data = data

        start_date, end_date = st.slider(
            "Select Time Period for Analysis:",
            min_value=min_date,
            max_value=max_date,
            value=(combined_data['Year-Month'].min().date(), combined_data['Year-Month'].max().date()),
            help="Adjust the time range to filter the sentiment data.",
        )

        # Filter data based on the selected range
        filtered_data = combined_data[
            (combined_data["Year-Month"] >= pd.Timestamp(start_date)) &
            (combined_data["Year-Month"] <= pd.Timestamp(end_date))
            ]

    with st.expander("⚙️ Settings", expanded=False):
        tts_flag = st.toggle("Alice Reads Responses Aloud", value=False)

        print("3=================")
        print(filtered_data.columns)
        print(filtered_data.head())
        print("3=================")

# Dynamically display tabs based on the selected option
if lob_filter == "🚁 Overview":

    # We create two sub-tabs for the Overview
    overview_tabs = st.tabs(["🚁 Summary", "🧪 Lapse Analysis", "📚 User Guide"])

    # 1. SUMMARY TAB
    with overview_tabs[0]:
        st.markdown("## Overall Sentiment Summary")

        # Sentiment Score by Product over Time
        plot_chart_1(
            "Sentiment Score",
            "Sentiment Score by Product Over Time",
            "text",
            filtered_data  # Ensure your data is loaded and available
        )

        st.write("""
        Below is a high-level view of our **Promoters**, **Detractors**, and **Emerging Trends** based on the 
        latest AI generated themes from social media, discussion forums, and any available online reviews. You can see the distribution of each category in the treemaps below, followed by 
        brief descriptions summarizing each sub-topic (e.g., 'Engineer Experience' for Promoters, or 'Billing Errors' 
        for Detractors).
        """)

        if "Percentage" in tab_data.columns:
            tab_data.loc[tab_data.index, "Percentage"] = tab_data["Percentage"].str.replace("%", "", regex=False).astype(float)

        # Separate out the three main categories from tab_data
        overview_categories = {
            "Promoters": {"icon": "🥳", "filter": "Promoters"},
            "Detractors": {"icon": "🤬", "filter": "Detractors"},
            "Emerging Trends": {"icon": "👽", "filter": "Emerging"},
        }

        # Create three columns for Promoters, Detractors, Emerging Trends
        cols = st.columns(len(overview_categories))

        # Loop through each category and process data
        for i, (category, details) in enumerate(overview_categories.items()):
            cat_data = tab_data[tab_data["Type"].str.contains(details["filter"], case=False, na=False)]

            with cols[i]:
                st.markdown(f"### {details['icon']} {category}")

                if not cat_data.empty:
                    # Create treemap
                    fig_treemap = px.treemap(
                        cat_data,
                        path=["Tab Headline"],
                        values="Percentage",
                        color="Percentage",
                        title=""
                    )
                    fig_treemap.update_layout(
                        margin=dict(l=10, r=10, t=40, b=10),
                        template="plotly_white"
                    )
                    st.plotly_chart(fig_treemap, use_container_width=True)

                    # Summarize each row dynamically
                    for _, row in cat_data.iterrows():
                        st.markdown(f"**{row['Tab Emoji']} {row['Tab Headline']} ~{int(row['Percentage'])}%**")
                        st.write(f"{row['Tab Description']}")
                else:
                    st.write(f"No {category} data available.")

        # Optional: Additional placeholders or summaries below the three columns
        st.markdown("---")
        st.markdown("### Additional Comments")
        st.write("""
        - Use the **Promoters** data to reinforce what's going well and replicate success across the business.
        - Address key concerns highlighted in the **Detractors** data with urgency to maintain brand reputation.
        - Keep a close eye on **Emerging Trends** to stay ahead of shifting sentiment and industry developments.

        This overview provides a snapshot of what’s driving customer happiness, frustration, and future risk factors.
        For deeper analysis, explore the specific insight modes in the sidebar or ask a custom question via **Ask Alice**.
        """)

    with overview_tabs[1]:
        st.markdown("""
        # Lapse Rate Analysis
        """, unsafe_allow_html=True)

        print("4=================")
        print(filtered_data.columns)
        print(filtered_data.head())
        print("4=================")

        # Tab 1 Chart 1: Sentiment Score by Product over Time
        plot_chart_1(
            "Lapse Probability",
            "Cancellation Sentiment by Product Over Time",
            "text",
            filtered_data  # Ensure your data is loaded and available
        )

    # 4. USER GUIDE TAB
    with overview_tabs[2]:
        st.markdown("## User Guide")

        st.markdown("""
            Welcome to the **Sentiment Analytics Dashboard**, a cutting-edge tool leveraging **state-of-the-art AI models** to extract actionable insights from real-time online data. 
            This platform is designed to turn digital noise into digital gold...

            <hr style="border: 1px solid #0490d7; margin: 20px 0;">
        """, unsafe_allow_html=True)

        st.write("""

        Here's how to use each section:

        **1. Insight Mode**  
        Select different insight modes (Promoters, Detractors, etc.) from the sidebar. Each mode shows relevant 
        tabs and data visualisations along with AI-generated summaries and recommended actions.

        **2. Time Period Settings**  
        Adjust the slider to filter data by date range. If you enable 'Show Future Projections', additional 
        forecast data is appended up to your chosen end date.

        **3. Company Selection**  
        Choose the company whose data you want to see (e.g., British Gas, Domestic & General, etc.).

        **~ Ask Alice**  
        Navigate to 'Ask Alice...' in the 'Insights' sidebar section to pose any custom questions about the social media / online data. 
        This is helpful for more specific or ad-hoc queries regarding customer sentiment, product segments, or time periods.

        **Best Practices**  
        - Use the filters to narrow down the data to the most relevant subset for your question.  
        - In 'Ask Alice', be as specific as possible (include time ranges, product details, or specific segments) 
          to get the most accurate AI-generated responses.  
        - Refer to the visualizations often; they can reveal trends or patterns the text summaries may not fully capture.

        We hope this guide helps you navigate and leverage the dashboard effectively!
        """)

elif lob_filter in insight_list:
    # Filter for Promoters
    insight_tabs = tab_data[tab_data["Type"] == lob_filter]

    # Create tabs dynamically using the Tab Name and Tab Emoji columns
    tabs = st.tabs([f"{row['Tab Emoji']} {row['Tab Name']}" for _, row in insight_tabs.iterrows()])

    # Loop through each tab and populate its content
    for idx, tab in enumerate(tabs):
        with tab:
            # Get the data for the current tab
            row = insight_tabs.iloc[idx]

            # Display the main title and description
            st.markdown(
                f"<div style='font-size: 35px; text-align: center; color: #012973; margin-bottom: 10px;'><b>{row['Tab Emoji']} {row['Tab Headline']}</b><br></div>",
                unsafe_allow_html=True
            )

            st.markdown(
                f"<div style='text-align: center; color: #012973;'>{row['Tab Description']}<br></div>",
                unsafe_allow_html=True
            )

            # Display the chart
            plot_chart_1(
                row["Chart Column"],
                row["Chart Title"],
                "text",
                filtered_data  # Ensure your data is loaded and available
            )

            # Display chart analysis
            st.write(row["Chart Analysis"])

            # Add another divider
            st.markdown("<hr>", unsafe_allow_html=True)

            # Display Sentiment Summary and Recommended Actions in two columns
            col1, col2 = st.columns([2, 2])
            with col1:
                st.markdown("<div style='font-size: 25px; text-align: center; color: #012973; margin-bottom: 10px;;'><b>🕵️AI Generated Online Sentiment Summary</b></div>", unsafe_allow_html=True)
                st.markdown(
                    f"<div class='rounded-block'>{row['Sentiment Summary']}</div>",
                    unsafe_allow_html=True
                )
            with col2:
                st.markdown(
                    f"<div style='font-size: 25px; text-align: center; color: #012973; margin-bottom: 10px;'><b>🏗️AI Generated Recommended Actions</b></div>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"<div class='rounded-block'>{row['Recommended Actions']}</div>",
                    unsafe_allow_html=True
                )

            # Add a divider
            st.markdown("<hr>", unsafe_allow_html=True)

            # Display Key Drivers dynamically
            st.markdown("### Key Drivers")
            driver_columns = [col for col in insight_tabs.columns if col.startswith("Driver")]
            for driver_col in driver_columns:
                if pd.notnull(row[driver_col]):
                    # Split driver data into title and text
                    title, text = row[driver_col].split(";", 1)
                    st.markdown(f"**{title.strip()}**")
                    st.write(text.strip())

elif lob_filter == "💁‍♀️ Ask Alice...":

    st.markdown(
        f"<div style='font-size: 35px; text-align: center; color: #012973; margin-bottom: 10px;'><b>🔬 Ask Alice a more specific sentiment analysis question</b><br></div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div style='text-align: center; color: #012973; margin-bottom: 15px;'>This selection allows you to ask specific questions focused on a specific month, and specific product group, which Alice will answer based on the sentiment and online chatter data available for the specific month and product group.</div>",
        unsafe_allow_html=True
    )
    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        filter_llm_month = st.selectbox(
            "Please select a specific month...",
            reviews_data["Year-Month"].unique() #("01/05/2024", "01/06/2024", "01/07/2024", "01/08/2024", "01/09/2024")  #("YY-MMM 01", "YY-MMM 02", "YY-MMM 03"),
        )
    with col2:
        filter_llm_prod = st.selectbox(
            "Please select a specific product group",
            reviews_data["Final Product Category"].unique(),
        )
    with col3:
        filter_llm_type = st.selectbox(
            "Please select a specific customer segment",
            ("Everyone", "Tbc...    "),
        )
    query_llm = st.text_area("Enter your more specific query here...")
    client = OpenAI()
    convo = []

    # Filter relevant data

    filtered_reviews = reviews_data[
        reviews_data["Year-Month"] == filter_llm_month]
    filtered_reviews = filtered_reviews[
        filtered_reviews["Final Product Category"] == filter_llm_prod][:1000].values.tolist()

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
            #st.write(context)
            response = client.chat.completions.create(
                model="gpt-4o-mini", #"o1-mini", #"gpt-4o-mini",
                messages=[
                    {"role": "system", "content": (
                        OPENAI_SYSTEM_PROMPT
                    )},
                    {"role": "user",
                     "content": (
                         f"Month: {filter_llm_month}\n\n"
                         f"Product Line: {filter_llm_prod}\n\n"
                         f"Social Media Data: {context}\n\n"
                         f"Question: {query}\n\n"
                         "Please reference the social media data directly in your answer, "
                         "and offer actionable steps British Gas Insurance can take."
                     )
                    }
                ],
                temperature=0.5,
                max_tokens=1000,
                #max_completion_tokens=250,
            )

            # Parse the response content
            answer = response.choices[0].message.content.strip()
            return answer

        except Exception as e:
            return f"Inferencing Failed, Error: {str(e)}"

    if st.button("🙋‍♀️ Ask Alice"):
        st.markdown(f"<b>🤔 User: </b>{query_llm}", unsafe_allow_html=True)
        if len(query_llm) == 0:
            st.markdown(f"<b>💁‍♀️ Alice</b>: Please enter a question, the query box is currently blank...", unsafe_allow_html=True)
        else:
            answer = llm_inference(query_llm,filtered_reviews)
            st.markdown(f"<b>💁‍♀️ Alice</b>:", unsafe_allow_html=True)
            st.write(answer)
            elapsed_time = datetime.now() - START_TIME
            hours, remainder = divmod(elapsed_time.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)
            pd.DataFrame([[datetime.now(),f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}",query_llm,filter_llm_month,filter_llm_prod,"Group Placeholder",answer]]).to_csv('LLM_Runlog.csv', mode='a', index=False, header=False)

            if tts_flag:
                audio_response = client.audio.speech.create(
                    model="tts-1",
                    voice="nova",
                    input=answer
                )
                audio_response.write_to_file("llm_answer.mp3")
                with open("llm_answer.mp3", "rb") as audio_file:
                    st.audio(audio_file, format='audio/mp3')


