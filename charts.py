import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import savgol_filter
from cats import *
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# === Helper Functions ===
# Smoothing function for charts
def smooth_data(series, window=3, method='rolling_mean', polyorder=2):

    if window == 1:
        return series # No smoothing applied

    if method == 'rolling_mean':
        return series.rolling(window=window, min_periods=1).mean() # Apply rolling mean smoothing

    elif method == 'savgol':
        # Check window is odd and greater than polyorder
        if window < 3:
            raise ValueError("Window size must be at least 3 for Savitzky–Golay filter.")
        if window % 2 == 0:
            window += 1  # Make window size odd

        if polyorder >= window:
            raise ValueError("Polynomial order must be less than window size.")

        # Apply Savitzky–Golay filter
        smoothed = savgol_filter(series, window_length=window, polyorder=polyorder)
        return pd.Series(smoothed, index=series.index)

    else:
        raise ValueError("Unsupported smoothing method. Choose 'rolling_mean' or 'savgol'.")


def filter_chart_data(data, title, filters, unique_key):

    # Create a container so all related widgets and summary appear together
    container = st.expander("🕵️ " + title + " - Expand for Data Filters ↓", expanded=False)
    with container:

        # Dynamically create product checkbox columns
        selected_products = product_categories
        if 'product' in filters:
            selected_products = []
            cols = st.columns(len(product_categories))
            for product, col in zip(product_categories, cols):
                if col.checkbox(product, value=True, key=f"{unique_key}_checkbox_{product}"):
                    selected_products.append(product)
            filtered_data = data[data['Final Product Category'].isin(selected_products)]
        else:
            filtered_data = data

        # Add the emotion dropdown
        selected_emotion = 'All'
        if 'emotion' in filters:
            emotion_options = ['All'] + emotion_categories
            selected_emotion = st.selectbox(
                "Select Emotion Filter:", emotion_options,
                index=0,  # Default to 'All'
                key=f"{unique_key}_emotion_select"
            )

        # Add a unique key to the slider to avoid conflicts
        if 'time' in filters:
            start_date, end_date = st.slider(
                "Select Time Period for Analysis:",
                min_value=data['Year-Month'].min().date(),
                max_value=data['Year-Month'].max().date(),
                value=(data['Year-Month'].min().date(), data['Year-Month'].max().date()),
                help="Adjust the time range to filter the sentiment data.",
                key=f"{unique_key}_date_slider"
            )
            filtered_period_data = filtered_data[
                (filtered_data['Year-Month'] >= pd.Timestamp(start_date)) &
                (filtered_data['Year-Month'] <= pd.Timestamp(end_date))
            ]
        else:
            filtered_period_data = filtered_data

        if selected_emotion != 'All':
            filtered_period_data = filtered_period_data[filtered_period_data['Strongest Emotion'] == selected_emotion]

        # Summarize filtered data
        product_counts = filtered_period_data['Final Product Category'].value_counts()
        total_count = len(filtered_period_data)
        emotion_averages = {emotion: filtered_period_data[emotion].mean() for emotion in emotion_categories}
        average_rating = filtered_period_data['Rating'].mean()

        # Format summaries
        product_count_text = ", ".join(
            f"{product}: {count} ({count / total_count * 100:.2f}%)"
            for product, count in product_counts.items()
        )

        emotion_average_text = ", ".join(
            f"{emotion.capitalize()}: {avg:.2f}" for emotion, avg in emotion_averages.items()
        )

        summary_markdown = f"""
        **Entries in Filtered Data:**
        - {product_count_text}
        - **Total Entries:** {total_count}

        **Average Metrics Across Filtered Data:**
        - **Average Rating:** {average_rating:.2f}
        - **Average Emotion Scores:** {emotion_average_text}
        """

        st.markdown(summary_markdown)

    return filtered_period_data, selected_products  # Return the filtered dataset for charting

def style_chart(fig):
    fig.update_layout(
        title_font=dict(family="Arial Black", size=24, color="#012973"),
        font=dict(family="Arial", size=14, color="#012973"),
        title_y=0.93, title_x=0.0,
        paper_bgcolor="white", plot_bgcolor="white",
        legend_title = " Legend",
        template="plotly_white", height=600, hovermode="x",
        legend=dict(bgcolor="white", bordercolor="#0490d7", borderwidth=1),
        xaxis=dict(title_font=dict(family="Arial, sans-serif", size=16, color="#012973")),
        yaxis=dict(title_font=dict(family="Arial, sans-serif", size=16, color="#012973")),
    )

    # Add a border around the entire chart
    fig.update_layout(
        shapes=[
            dict(type="rect", xref="paper", yref="paper",
                    x0=0, y0=0, x1=1, y1=1,
                    line=dict(color="#0490d7", width=1, dash="dot"),
                    fillcolor="rgba(0,0,0,0)"
            )
        ]
    )

    return fig

def add_red_line(fig):
    fig.add_shape(
        x0=0, y0=0, x1=1, y1=0,
        type="line", yref="y", xref="paper",
        line=dict(color="red", width=2, dash="dot")
    )
    return fig

def plot_chart_1(group_var, title, desc, data):
    """
    Renders a chart based on the specified parameters.

    Args:
        chart_column (str): Column name for chart data.
        chart_title (str): Title of the chart.
        chart_type (str): Type of the chart (e.g., 'text') - this is still in development
        data (DataFrame): Data to visualize.
    """

    # Aggregate data into grouping for plot
    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
    data_grouped = data.groupby(["Year-Month", "Final Product Category"], as_index=False)[numeric_cols].mean()

    # Create empty chart object
    fig = go.Figure()

    prods = data_grouped["Final Product Category"].unique().tolist()

    # Separate projected data
    if 'Projected' not in data_grouped:
        data_grouped["Projected"] = 0

    # Create a line for each product
    for product in prods:
        df_prod = data_grouped[data_grouped['Final Product Category'] == product]
        df_prod_act = df_prod[df_prod['Projected'] != 1]
        df_prod_pred = df_prod[df_prod['Projected'] == 1]

        # Ensure continuity between actual and predicted
        if not df_prod_act.empty and not df_prod_pred.empty:
            last_actual_row = df_prod_act.iloc[[-1]]  # Get the last row of actual data
            df_prod_pred = pd.concat([last_actual_row, df_prod_pred])

        fig.add_trace(
            go.Scatter(
                x=df_prod_act['Year-Month'], y=df_prod_act[group_var],
                mode='lines+markers', name=product_emoji_map[product],
                line=dict(color=product_colours[product], width=1),
                hovertemplate=("<b>Sentiment Score:</b> %{y:.2f}<br>")
            )
        )

        # Add predicted data as dashed line
        if not df_prod_pred.empty:
            fig.add_trace(
                go.Scatter(
                    x=df_prod_pred['Year-Month'], y=df_prod_pred[group_var],
                    mode='lines+markers', name=f"{product_emoji_map[product]} (Predicted)",
                    line=dict(color=product_colours[product], width=2, dash='dot'),
                    marker=dict(symbol='circle-open', size=10),
                    hovertemplate="<i>Predicted Value</i>: %{y:.2f}<br>",
                    showlegend=False,
                )
            )

    # Add chart labelling
    fig.update_layout(
        title=title, #yaxis_range=[-0.99, 0.99],
        xaxis_title="Month & Year", yaxis_title="Sentiment Score",
    )

    # Add chart styling and display
    fig = style_chart(fig)
    fig = add_red_line(fig)
    st.plotly_chart(fig, use_container_width=True)

def plot_chart_2(product, title, desc, data, view="aspect"):
    """
    Plots sentiment trends for different service aspects over time for a selected product.

    Args:
        product (str): The product category to filter the data.
        title (str): Title of the chart.
        desc (str): Chart description (not currently used, but available for future expansion).
        data (DataFrame): The full dataset containing sentiment scores.
    """

    # Define aspect sentiment score column names
    if view == "aspect":
        aspect_columns = [f"{aspect}_sentiment_score" for aspect in aspects_map.keys()]
    elif view == "sentiment":
        aspect_columns = ["Sentiment Score"]

    # Filter data for the selected product only
    if "all" not in product.lower():
        data_filtered = data[data["Final Product Category"] == product]
    else:
        data_filtered = data

    # Ensure we have valid data
    if data_filtered.empty:
        st.warning(f"No data available for {product}. Please select a different product.")
        return

    # Group by time (Year-Month) and calculate mean sentiment scores for each aspect
    data_grouped = data_filtered.groupby("Year-Month", as_index=False)[aspect_columns].mean()
    data_grouped['Year-Month'] = pd.to_datetime(data_grouped['Year-Month'], format='%d/%m/%Y', errors='raise')

    # Sort by chronological order
    data_grouped = data_grouped.sort_values("Year-Month")

    # Create a Plotly figure
    fig = go.Figure()
    print("View: " + view)
    if view == "aspect":
        # Add a line trace for each aspect
        for aspect, aspect_column in zip(aspects, aspect_columns):
            fig.add_trace(
                go.Scatter(
                    x=data_grouped["Year-Month"],
                    y=data_grouped[aspect_column],
                    mode="lines+markers",
                    name=aspects_map[aspect],  # Legend will display the aspect name
                    hovertemplate=f"<b>{aspect} Sentiment Score:</b> %{{y:.2f}}<br>",
                    line=dict(width=2)  # Keep the lines clean and simple
                )
            )
    elif view == "sentiment":
        fig.add_trace(
            go.Scatter(
                x=data_grouped["Year-Month"],
                y=data_grouped["Sentiment Score"],
                mode="lines+markers",
                name="Overall Sentiment",
                hovertemplate="<b>Overall Sentiment Score:</b> %{y:.2f}<br>",
                line=dict(width=2)
            )
        )


    # Update chart aesthetics using the existing style function
    fig.update_layout(
        title=title,
        xaxis_title="Month & Year",
        #yaxis_title="Sentiment Score",
        legend_title="Service Aspects",
        legend=dict(
            orientation="h",  # Horizontal legend
            yanchor="top",
            y=-0.2,  # Position below the chart
            xanchor="center",
            x=0.5
        )
    )

    # Set the y-axis range to have a maximum of 95
    fig.update_yaxes(range=[-60, 95])

    fig = style_chart(fig)  # Apply styling
    fig = add_red_line(fig)  # Add reference line at y=0

    # Display the chart
    st.plotly_chart(fig, use_container_width=True)


def plot_chart_3(product, aspect, title, desc, data):
    """
    Plots a multiline sentiment trend chart for a single service aspect over time,
    where each line represents a competitor for the selected product.

    Args:
        product (str): The product category to filter the data.
        aspect (str): The service aspect key (e.g., "Appointment Scheduling").
        title (str): Title of the chart.
        desc (str): Chart description (for future use).
        data (DataFrame): The full dataset containing sentiment scores.
    """
    # Determine the sentiment column name for the specified aspect
    sentiment_column = f"{aspect}_sentiment_score"

    # Filter data for the selected product only
    data_filtered = data[data["Final Product Category"] == product]
    print(product)

    # Check for valid data
    if data_filtered.empty:
        st.warning(f"No data available for {product}. Please select a different product.")
        return

    # Ensure 'Year-Month' is in datetime format and sort the data
    data_filtered['Year-Month'] = pd.to_datetime(data_filtered['Year-Month'], format='%d/%m/%Y', errors='raise')
    data_filtered = data_filtered.sort_values("Year-Month")

    # Create a Plotly figure
    fig = go.Figure()

    # Get the list of all competitors in the filtered data
    competitor_list = data_filtered["Company"].unique()

    # For each competitor, group data by 'Year-Month' and calculate the mean sentiment score
    for competitor in competitor_list:
        competitor_data = data_filtered[data_filtered["Company"] == competitor]
        competitor_grouped = competitor_data.groupby("Year-Month", as_index=False)[sentiment_column].mean()
        competitor_grouped = competitor_grouped.sort_values("Year-Month")

        # Add a trace for this competitor
        fig.add_trace(
            go.Scatter(
                x=competitor_grouped["Year-Month"],
                y=competitor_grouped[sentiment_column],
                mode="lines+markers",
                name=competitor,
                hovertemplate=f"<b>{competitor} {aspect} Sentiment Score:</b> %{{y:.2f}}<br>",
                line=dict(width=2)
            )
        )

    # Update chart layout to mimic plot_chart_2 styling
    fig.update_layout(
        title=title,
        xaxis_title="Month & Year",
        # yaxis_title="Sentiment Score",
        legend_title="Competitors",
        legend=dict(
            orientation="h",  # Horizontal legend
            yanchor="top",
            y=-0.2,  # Position below the chart
            xanchor="center",
            x=0.5
        )
    )

    # Set the y-axis range as before
    fig.update_yaxes(range=[-60, 95])

    # Apply existing styling functions (assumed to be defined elsewhere)
    fig = style_chart(fig)
    fig = add_red_line(fig)

    # Display the chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)


def plot_aspect_comparison(product, aspect, company, title, desc, data, height=200):
    """
    Plots a small line chart comparing the sentiment trends for a single aspect
    between the selected company and British Gas over time.

    Args:
        product (str): The product category to filter the data.
        aspect (str): The aspect to plot (e.g., "Appointment Scheduling").
        company (str): The selected company. If this is "British Gas", only one line is plotted.
        title (str): Title of the chart.
        desc (str): Chart description (currently not used).
        data (DataFrame): The full dataset containing sentiment scores.
        height (int): Fixed height for the chart (in pixels). Default is 200.
    """
    # Determine the column name for this aspect (e.g., "Appointment Scheduling_sentiment_score")
    aspect_col = f"{aspect}_sentiment_score"

    # Filter the data by product.
    if "all" not in product.lower():
        company_data = data[
            (data["Company"].str.contains(company)) &
            (data["Final Product Category"].str.contains(product))
            ]
        # Only filter for British Gas if the selected company is not British Gas.
        if company != "British Gas":
            bg_data = data[
                (data["Company"].str.contains("British Gas")) &
                (data["Final Product Category"].str.contains(product))
                ]
        else:
            bg_data = None
    else:
        company_data = data[data["Company"].str.contains(company)]
        if company != "British Gas":
            bg_data = data[data["Company"].str.contains("British Gas")]
        else:
            bg_data = None

    # Return if no data for the selected company.
    if company_data.empty:
        return

    # Group the selected company data by "Year-Month" for the aspect.
    company_grouped = company_data.groupby("Year-Month", as_index=False)[aspect_col].mean()
    company_grouped["Year-Month"] = pd.to_datetime(company_grouped["Year-Month"], format='%d/%m/%Y', errors='raise')
    company_grouped = company_grouped.sort_values("Year-Month")

    # If applicable, group British Gas data.
    if bg_data is not None and not bg_data.empty:
        bg_grouped = bg_data.groupby("Year-Month", as_index=False)[aspect_col].mean()
        bg_grouped["Year-Month"] = pd.to_datetime(bg_grouped["Year-Month"], format='%d/%m/%Y', errors='raise')
        bg_grouped = bg_grouped.sort_values("Year-Month")
    else:
        bg_grouped = None

    # Create the Plotly figure.
    fig = go.Figure()

    # Add trace for the selected company (in maroon).
    fig.add_trace(
        go.Scatter(
            x=company_grouped["Year-Month"],
            y=company_grouped[aspect_col],
            mode="lines+markers",
            name=company,
            line=dict(color="maroon", width=2),
            hovertemplate=f"<b>{company} {aspect} Sentiment:</b> %{{y:.2f}}<br>"
        )
    )

    # If the selected company is not British Gas, add a trace for British Gas (in blue).
    if company != "British Gas" and bg_grouped is not None:
        fig.add_trace(
            go.Scatter(
                x=bg_grouped["Year-Month"],
                y=bg_grouped[aspect_col],
                mode="lines+markers",
                name="British Gas",
                line=dict(color="blue", width=2),
                hovertemplate=f"<b>British Gas {aspect} Sentiment:</b> %{{y:.2f}}<br>"
            )
        )

    # Update chart layout with fixed height.
    fig.update_layout(
        title=title,
        xaxis_title="Month & Year",
        yaxis_title="Sentiment Score",
        height=height,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )

    # Apply additional styling (if you have style_chart and add_red_line functions).
    fig = style_chart(fig)
    fig = add_red_line(fig)

    # Display the chart.
    st.plotly_chart(fig, use_container_width=True)


def plot_chart_all_products(product, title, desc, data, metric, company, height=200):
    """
    Plots a line chart that shows one trace per product (from 'Final Product Category')
    for a given company. The metric used for plotting is chosen via a dropdown.

    Args:
        product (str): The product indicator (should be "all" in this view).
        title (str): Title of the chart.
        desc (str): Description (unused currently).
        data (DataFrame): The full sentiment dataset.
        metric (str): The chosen metric, either "Overall Sentiment Score" or one of the aspect names.
        company (str): The company to filter by.
        height (int): The fixed height for the chart (default 200 pixels).
    """
    # Determine the sentiment column to use.
    if metric == "Overall Sentiment Score":
        sentiment_column = "Sentiment Score"
    else:
        sentiment_column = f"{metric}_sentiment_score"

    # Filter the data for the specified company.
    filtered_data = data[data["Company"].str.contains(company)]
    if filtered_data.empty:
        return

    # Ensure the "Year-Month" column is datetime (if not already).
    filtered_data["Year-Month"] = pd.to_datetime(filtered_data["Year-Month"], format='%d/%m/%Y', errors='raise')

    # Group by "Year-Month" and "Final Product Category", computing the mean of the sentiment column.
    grouped_data = filtered_data.groupby(["Year-Month", "Final Product Category"], as_index=False)[
        sentiment_column].mean()
    grouped_data = grouped_data.sort_values("Year-Month")

    # Create a Plotly figure.
    fig = go.Figure()

    # Loop over each unique product (from "Final Product Category") and add a trace.
    for product_cat in grouped_data["Final Product Category"].unique():
        product_data = grouped_data[grouped_data["Final Product Category"] == product_cat]
        # Use your product_colours mapping (if available) for colour; default to None for Plotly's color.
        colour = product_colours.get(product_cat, None)

        fig.add_trace(
            go.Scatter(
                x=product_data["Year-Month"],
                y=product_data[sentiment_column],
                mode="lines+markers",
                name=product_cat,
                line=dict(color=colour, width=2),
                hovertemplate=f"<b>{product_cat} {metric}:</b> %{{y:.2f}}<br>"
            )
        )

    # Update layout.
    fig.update_layout(
        title=title,
        xaxis_title="Month & Year",
        yaxis_title=metric,
        height=height,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )

    # Apply your standard styling functions.
    fig = style_chart(fig)
    fig = add_red_line(fig)

    # Display the chart.
    st.plotly_chart(fig, use_container_width=True)
