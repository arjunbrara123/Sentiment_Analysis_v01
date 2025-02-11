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
company_list = prod_summary_data['Company'].unique().tolist()
sa_monthly_data = pd.read_csv("LLM SA Monthly Data.csv")

# Create a container with a white background for the sidebar controls
with st.sidebar:
    st.sidebar.image("company_logo.png")
    st.sidebar.markdown(f"**Current User:** {getuser()}")


    # New mode selection radio button at the very top

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
        input_LLM_Tabs_Summary_data = "LLM Summary " + "British Gas.csv" #+ company_name + ".csv"
        #input_SA_Monthly_data = "SA Monthly Data " + company_name + ".csv"
        input_Raw_Comments_Text_data = "Cleaned Reviews " + company_name + ".csv"

        #data = load_agg_data(input_SA_Monthly_data)
        tab_data = pd.read_csv(input_LLM_Tabs_Summary_data)
        #insight_list = tab_data["Type"].unique().tolist()
        reviews_data = pd.read_csv(input_Raw_Comments_Text_data)

    product_name = selected_product.split(' ', 1)[-1]

    with st.expander("🧩 Analysis Mode", expanded=False):
        analysis_mode = st.radio(
            "Please Select Analysis",
            options=["🚁 Overview", "👽 Emerging Trends", "💬 Ask Alice..."],
            index=0
        )

    with st.expander("🕜 Time Period Settings", expanded=False):
        st.write("Test")

    with st.expander("⚙️ Settings", expanded=False):
        tts_flag = st.toggle("Alice Reads Responses Aloud", value=False)

# Main dashboard layout
if mode == "🏢 Company Mode":

    #st.markdown(f"# {company_name} - Competitor Analytics")
    st.markdown(f"# Competitor Analytics")
    st.markdown(f"**Selected Product Line:** {selected_product}")
    #st.markdown(f"# {company_name}")

    if analysis_mode == "🚁 Overview":
        #st.markdown("## Side-by-Side Product Overview")

        # Filter the summary for the selected company and product
        selected_summary = prod_summary_data[
            (prod_summary_data["Company"] == selected_company) &
            (prod_summary_data["Product"] == selected_product)
        ]

        # Filter the summary for British Gas and the same product
        british_gas_summary = prod_summary_data[
            (prod_summary_data["Company"].str.contains("British Gas")) &
            (prod_summary_data["Product"] == selected_product)
        ]

        # Create two columns for side-by-side display
        col1, col2 = st.columns(2)

        # Left Column: Selected Company's Summary
        with col1:
            #st.markdown(f"## {company_name} ")# - {selected_product}")
            st.markdown(f"<h2 style='text-decoration: underline;'>{company_name}</h2>", unsafe_allow_html=True)
            #if selected_summary.size > 0:
            st.markdown('### Strengths')
            st.write(selected_summary["Strengths"].values[0])
            st.markdown('### Weaknesses')
            st.write(selected_summary["Weaknesses"].values[0])
            #else:
            #    st.write("No summary available for the selected company and product.")

        # Right Column: British Gas Summary (or blank if British Gas is selected)
        with col2:
            #st.markdown(f"## British Gas")
            st.markdown("<h2 style='text-decoration: underline;'>British Gas</h2>", unsafe_allow_html=True)
            if "British Gas" not in selected_company:
                # if british_gas_summary.size > 0:
                #     st.write(british_gas_summary[0])
                # else:
                #     st.write("No British Gas summary available for the selected product.")
                st.markdown('### Strengths')
                st.write(british_gas_summary["Strengths"].values[0])
                st.markdown('### Weaknesses')
                st.write(british_gas_summary["Weaknesses"].values[0])
            else:
                st.write("N/A (Selected company is British Gas)")

        # Add divider
        st.markdown("<hr style='border: 1px solid #0490d7; margin: 20px 0;'>", unsafe_allow_html=True)

        # When toggled on, it represents "Aspect View". When off, it represents "Sentiment View".
        chart_toggle = st.toggle("Split Sentiment Into Aspects", value=True,
                                 help="Toggle between Aspect View (all aspects) and Sentiment View (overall sentiment)")
        if chart_toggle:
            view = "aspect"
        else:
            view = "sentiment"

        # Create two columns for side-by-side display
        col1, col2 = st.columns(2)

        # Left Column: Selected Company's Summary
        with col1:
            #st.markdown(f"### {company_name} - {selected_product}")

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

                #st.markdown("### Sentiment Trends Over Time")
                plot_chart_2(product_name, company_name + " " + product_name + " Sentiment", "", filtered_data_left, view)
            else:
                st.write("No sentiment data available for the selected company and product.")

        # Right Column: British Gas Summary (or blank if British Gas is selected)
        with col2:
            #st.markdown(f"### Equivalent BG Product Comparison")
            if "British Gas" not in selected_company:

                print("-----")
                print(product_name.lower)
                print("=====")
                # Filter sentiment data for British Gas and the same product
                if "all" not in product_name.lower():
                    filtered_data_right = sa_monthly_data[
                        (sa_monthly_data["Company"].str.contains("British Gas")) &
                        (sa_monthly_data["Final Product Category"].str.contains(product_name))
                        ]
                    print("Case 01")
                else:
                    print("Case e3")
                    filtered_data_right = sa_monthly_data[
                        (sa_monthly_data["Company"].str.contains("British Gas"))
                        ]
                # Plot sentiment graph for British Gas
                if not filtered_data_right.empty:

                    #st.markdown("### Sentiment Trends Over Time")
                    plot_chart_2(product_name, f"vs BG {product_name} Sentiment Score", "", filtered_data_right, view)
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

            # # Define the fixed list of aspect names (these must match exactly the CSV column headers)
            # fixed_aspects = [
            #     "Appointment Scheduling",
            #     "Customer Service",
            #     "Response Speed",
            #     "Engineer Experience",
            #     "Solution Quality",
            #     "Value For Money"
            # ]
            #
            # # Loop through each aspect and create an expander with its text
            # for aspect in fixed_aspects:
            #     with st.expander(aspect):
            #         st.write(row[aspect])

            # Use aspects_map directly to loop through the aspects.
            for aspect_col, aspect_display in aspects_map.items():
                with st.expander(aspect_display):
                    st.write(row[aspect_col])
        else:
            st.write("No aspect details available.")

        # aspect_summaries = tab_data[tab_data["Type"].str.contains("Promoter")]
        # for _, row in aspect_summaries.iterrows():
        #     with st.expander(f"{row['Tab Emoji']} {row['Tab Headline']} ({row['Percentage']}%)"):
        #         st.markdown(f"**Summary**\n{row['Sentiment Summary']}")
        #         st.markdown(f"**Recommended Actions**\n{row['Recommended Actions']}")

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

elif mode == "🎍 Market Mode":
    st.markdown(f"# Market Sentiment Analytics for {selected_product}")

    # Dynamically generate tabs using the aspects_map dictionary.
    # aspects_map keys hold the aspect keys and values hold the display names (with emojis).
    aspects = list(aspects_map.keys())
    tab_names = [aspects_map[aspect] for aspect in aspects]

    # Create 6 tabs (one for each aspect)
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
