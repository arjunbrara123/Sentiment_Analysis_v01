# Import required packages
import charts
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

# Add this with other data loading (after reviews_data = ...)
prod_summary_data = pd.read_csv("LLM Prod Level Summary.csv")
#prod_summary_data['Company'] = prod_summary_data['Company'].str.strip()  # Clean whitespace
company_list = prod_summary_data['Company'].unique().tolist()
sa_monthly_data = pd.read_csv("LLM SA Monthly Data.csv")

# Create a container with a white background for the sidebar controls
with st.sidebar:
    st.sidebar.image("company_logo.png")
    st.sidebar.markdown(f"**Current User:** {getuser()}")

    with st.expander("🏫 Select Competitor / Product", expanded=False):
        selected_company = st.radio(
            "Please Select a Company",
            options=company_list,
            index=0
        )

        selected_product = st.radio(
            "Please Select a Product",
            options = prod_summary_data[prod_summary_data['Company'] == selected_company]['Product'].unique().tolist(),
            #options=["🌍 All", "🚿 Gas Products", "⚡ Energy", "🪠 Plumbing & Drains"],
            index=0
        )

    company_name = selected_company.split(' ', 1)[-1]
    product_name = selected_product.split(' ', 1)[-1]
    input_LLM_Tabs_Summary_data = "LLM Summary " + "British Gas.csv" #+ company_name + ".csv"
    #input_SA_Monthly_data = "SA Monthly Data " + company_name + ".csv"
    input_Raw_Comments_Text_data = "Cleaned Reviews " + company_name + ".csv"

    #data = load_agg_data(input_SA_Monthly_data)
    tab_data = pd.read_csv(input_LLM_Tabs_Summary_data)
    #insight_list = tab_data["Type"].unique().tolist()
    reviews_data = pd.read_csv(input_Raw_Comments_Text_data)

    with st.expander("🧩 Analysis Mode", expanded=False):
        analysis_mode = st.radio(
            "Please Select Analysis",
            options=["🚁 Overview", "👽 Emerging Trends", "🗺️ Customer Journey", "💬 Ask Alice..."],
            index=0
        )

    with st.expander("🕜 Time Period Settings", expanded=False):
        st.write("Test")

    with st.expander("⚙️ Settings", expanded=False):
        tts_flag = st.toggle("Alice Reads Responses Aloud", value=False)

# Main dashboard layout
st.markdown(f"# {company_name} Sentiment Analytics")
st.markdown(f"**Selected Product Line:** {selected_product}")

if analysis_mode == "🚁 Overview":
    st.markdown("## Side-by-Side Product Overview")

    # Filter the summary for the selected company and product
    selected_summary = prod_summary_data[
        (prod_summary_data["Company"] == selected_company) &
        (prod_summary_data["Product"] == selected_product)
    ]["Summary"].values

    # Filter the summary for British Gas and the same product
    british_gas_summary = prod_summary_data[
        (prod_summary_data["Company"].str.contains("British Gas")) &
        (prod_summary_data["Product"] == selected_product)
    ]["Summary"].values

    # Create two columns for side-by-side display
    col1, col2 = st.columns(2)

    # Left Column: Selected Company's Summary
    with col1:
        st.markdown(f"### {company_name} - {selected_product}")
        if selected_summary.size > 0:
            st.write(selected_summary[0])
        else:
            st.write("No summary available for the selected company and product.")

    # Right Column: British Gas Summary (or blank if British Gas is selected)
    with col2:
        st.markdown(f"### Equivalent BG Product Comparison")
        if "British Gas" not in selected_company:
            if british_gas_summary.size > 0:
                st.write(british_gas_summary[0])
            else:
                st.write("No British Gas summary available for the selected product.")

        else:
            st.write("N/A (Selected company is British Gas)")

    # Add divider
    st.markdown("<hr style='border: 1px solid #0490d7; margin: 20px 0;'>", unsafe_allow_html=True)

    # Create two columns for side-by-side display
    col1, col2 = st.columns(2)

    # Left Column: Selected Company's Summary
    with col1:
        #st.markdown(f"### {company_name} - {selected_product}")

        # Plot sentiment graph for the selected company and product
        filtered_data_left = sa_monthly_data[
            (sa_monthly_data["Company"].str.contains(company_name)) &
            (sa_monthly_data["Final Product Category"].str.contains(product_name))
        ]
        if not filtered_data_left.empty:

            #st.markdown("### Sentiment Trends Over Time")
            plot_chart_2(product_name, company_name + " " + product_name + " Sentiment", "", filtered_data_left)
        else:
            st.write("No sentiment data available for the selected company and product.")

    # Right Column: British Gas Summary (or blank if British Gas is selected)
    with col2:
        #st.markdown(f"### Equivalent BG Product Comparison")
        if "British Gas" not in selected_company:

            # Filter sentiment data for British Gas and the same product
            filtered_data_right = sa_monthly_data[
                (sa_monthly_data["Company"].str.contains("British Gas")) &
                (sa_monthly_data["Final Product Category"].str.contains(product_name))
                ]
            # Plot sentiment graph for British Gas
            if not filtered_data_right.empty:

                #st.markdown("### Sentiment Trends Over Time")
                plot_chart_2(product_name, f"vs BG {product_name} Sentiment Score", "", filtered_data_right)
            else:
                st.write("No sentiment data available for British Gas.")

        else:
            st.write("N/A (Selected company is British Gas)")

    # Add divider
    st.markdown("<hr style='border: 1px solid #0490d7; margin: 20px 0;'>", unsafe_allow_html=True)

    st.markdown("## Aspects")

    promoters = tab_data[tab_data["Type"].str.contains("Promoter")]
    for _, row in promoters.iterrows():
        with st.expander(f"{row['Tab Emoji']} {row['Tab Headline']} ({row['Percentage']}%)"):
            st.markdown(f"**Summary**\n{row['Sentiment Summary']}")
            st.markdown(f"**Recommended Actions**\n{row['Recommended Actions']}")

elif analysis_mode == "👽 Emerging Trends":
    st.markdown("## Emerging Customer Sentiment Trends")

    # trends = tab_data[tab_data["Type"].str.contains("Emerging")]
    # cols = st.columns(2)
    # for idx, (_, row) in enumerate(trends.iterrows()):
    #     with cols[idx % 2]:
    #         st.markdown(f"### {row['Tab Emoji']} {row['Tab Headline']}")
    #         st.write(f"**{row['Percentage']}%** of recent mentions")
    #         st.markdown(f"*{row['Chart Analysis']}*")
    #         st.markdown("---")

elif analysis_mode == "💬 Ask Alice...":
    st.markdown("## AI Insights Assistant")

    query = st.text_input("Ask Alice about customer sentiment patterns:")
    if query:
        # Simplified AI response (would integrate with your existing Alice logic)
        response = f"Alice's analysis for {selected_product}:\n\n- Positive trend in engineer responsiveness\n- Ongoing issues with billing clarity\n- Recommendation: Prioritize billing system audit"
        st.info(response)
