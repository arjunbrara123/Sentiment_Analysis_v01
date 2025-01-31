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
    page_title="🕵️ Competitor Analytics",
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

    with st.expander("🏫 Select Competitor", expanded=False):
        selected_company = st.radio(
            "Please Select a Company",
            options=["British Gas", "HomeServe"],
            index=0
        )

    with st.expander("🎁 Select Product", expanded=False):
        selected_product = st.radio(
            "Please Select a Product",
            options=["🌍 All", "🚿 Gas Products", "⚡ Energy", "🪠 Plumbing & Drains"],
            index=0
        )

    with st.expander("🧩 Analysis Mode", expanded=False):
        analysis_mode = st.radio(
            "Please Select Analysis",
            options=["🚁 Overview", "🥳 Strengths", "⚠️ Weaknesses", "👽 Emerging Trends", "🧱 Standardised Insights"],
            index=0
        )


    input_LLM_Tabs_Summary_data = "LLM Summary Data " + selected_company + ".csv"
    input_SA_Monthly_data = "SA Monthly Data " + selected_company + ".csv"
    input_Raw_Comments_Text_data = "Cleaned Reviews " + selected_company + ".csv"

    data = load_agg_data(input_SA_Monthly_data)
    tab_data = pd.read_csv(input_LLM_Tabs_Summary_data)
    reviews_data = pd.read_csv(input_Raw_Comments_Text_data)
    insight_list = tab_data["Type"].unique().tolist()

# Main dashboard layout
st.markdown(f"# {selected_company} Sentiment Analytics")
st.markdown(f"**Selected Product Line:** {selected_product}")

if analysis_mode == "🚁 Overview":
    # Treemap visualization for overview
    st.markdown("## Sentiment Overview by Theme")

    fig = px.treemap(tab_data,
                     path=['Type', 'Tab Headline'],
                     values='Percentage',
                     color='Percentage',
                     color_continuous_scale='Blues')
    st.plotly_chart(fig, use_container_width=True)

elif analysis_mode == "🥳 Strengths":
    st.markdown("## Key Strength Drivers")

    # Filter and display promoters
    promoters = tab_data[tab_data["Type"].str.contains("Promoter")]
    for _, row in promoters.iterrows():
        with st.expander(f"{row['Tab Emoji']} {row['Tab Headline']} ({row['Percentage']}%)"):
            st.markdown(f"**Summary**\n{row['Sentiment Summary']}")
            st.markdown(f"**Recommended Actions**\n{row['Recommended Actions']}")

            # Show associated chart
            plot_chart_1(
                row["Chart Column"],
                row["Chart Title"],
                "text",
                data
            )

elif analysis_mode == "⚠️ Weaknesses":
    st.markdown("## Key Pain Points")

    # Filter and display detractors
    detractors = tab_data[tab_data["Type"].str.contains("Detractor")]
    for _, row in detractors.iterrows():
        with st.expander(f"{row['Tab Emoji']} {row['Tab Headline']} ({row['Percentage']}%)"):
            st.markdown(f"**Issue Summary**\n{row['Sentiment Summary']}")
            st.markdown(f"**Mitigation Plan**\n{row['Recommended Actions']}")

            # Show trend chart
            plot_chart_1(
                row["Chart Column"],
                row["Chart Title"],
                "text",
                data
            )

elif analysis_mode == "👽 Emerging Trends":
    st.markdown("## Emerging Customer Sentiment Trends")

    trends = tab_data[tab_data["Type"].str.contains("Emerging")]
    cols = st.columns(2)
    for idx, (_, row) in enumerate(trends.iterrows()):
        with cols[idx % 2]:
            st.markdown(f"### {row['Tab Emoji']} {row['Tab Headline']}")
            st.write(f"**{row['Percentage']}%** of recent mentions")
            st.markdown(f"*{row['Chart Analysis']}*")
            st.markdown("---")

elif analysis_mode == "💬 Ask Alice...":
    st.markdown("## AI Insights Assistant")

    query = st.text_input("Ask Alice about customer sentiment patterns:")
    if query:
        # Simplified AI response (would integrate with your existing Alice logic)
        response = f"Alice's analysis for {selected_product}:\n\n- Positive trend in engineer responsiveness\n- Ongoing issues with billing clarity\n- Recommendation: Prioritize billing system audit"
        st.info(response)
