"""
Charts Module
=============

This module provides charting functions used in the Competitor Analytics Dashboard.
Below is a list of the available chart functions along with their purpose, outputs,
and where they are used in the dashboard:

1. plot_chart_all_products(product, title, desc, data, metric, company, height=200)
   - Dashboard Usage: Called in Company Mode Overview (left/right columns) when the selected product is "All".
   - Purpose: Plots one trace per product category for a specified company.
   - Output: Renders a Plotly line chart (via Streamlit) showing overall sentiment or a chosen aspect metric.

2. plot_chart_2(product, title, desc, data, view="aspect")
   - Dashboard Usage: Called in Company Mode Overview when a specific (non "All") product is selected.
   - Purpose: Plots sentiment trends over time for a specific product.
     Can toggle between an overall sentiment view and an aspect breakdown view.
   - Output: Renders a Plotly line chart (via Streamlit).

3. plot_aspect_comparison(product, aspect, company, title, desc, data, height=500)
   - Dashboard Usage: Called in Company Mode’s aspect tabs to compare the selected company versus British Gas.
   - Purpose: Compares the sentiment trend for a single aspect between the selected company and British Gas.
   - Output: Renders a small Plotly line chart (via Streamlit) comparing the two companies.

4. plot_chart_3(product, aspect, title, desc, data)
   - Dashboard Usage: Called in Market Mode (when a specific product is selected) for each aspect’s market trends.
   - Purpose: Plots market sentiment trends for a given aspect, with one line per competitor.
   - Output: Renders a Plotly line chart (via Streamlit) showing competitor sentiment trends.

5. plot_product_overall_sentiment(product, title, data, height=400)
   - Dashboard Usage: Called in Market Mode when “All” products are selected to display overall sentiment trends.
   - Purpose: Plots overall sentiment trends for a product across all companies.
   - Output: Renders a Plotly line chart (via Streamlit) showing each company’s overall sentiment.

6. plot_comparison_hist(product, aspect, company, title, desc, data, height=200, metric=None, companies=None)
   - Dashboard Usage: Called in both Company Mode Overview (for overall sentiment comparisons) and in aspect tabs when a histogram view is desired.
   - Purpose: Creates a distribution plot (histogram/density plot) comparing sentiment scores. By default, it compares a single aspect’s sentiment between the selected company and British Gas, but by passing a different metric (e.g., "Sentiment Score") and an optional list of companies, it can also compare overall sentiment across multiple companies.
   - Output: Renders a Plotly histogram/density chart (via Streamlit) with configurable bin size and smooth appearance.

Note:
- This module assumes that the external constants (e.g. product_emoji_map, product_colours, insurer_colours,
  aspects_map, aspects, aspect_colours, etc.) are defined in the module "cats".
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff

# Import external configurations and colour mappings from config module
from config import PRODUCT_CONFIG, COMPANY_CONFIG, ASPECT_CONFIG, EMOTION_CONFIG


# --- Helper Functions ---

def apply_chart_style(fig, title="", xaxis_title="Month & Year", yaxis_title="Sentiment Score", height=600, legendpos="h"):
    """
    Applies consistent styling to the Plotly figure.
    Uses Arial fonts, blue border (#0490d7) with dash-dot, and adds a red dashed reference line at y=0.
    """
    fig.update_layout(
        title=title,
        title_font=dict(family="Arial Black", size=24, color="#012973"),
        font=dict(family="Arial", size=14, color="#012973"),
        title_y=0.96,
        title_x=0.0,
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend_title="",
        template="plotly_white",
        height=height,
        hovermode="closest",
        legend=dict(bgcolor="white", bordercolor="#0490d7", borderwidth=0, orientation="h"),
        xaxis=dict(
            title="", #xaxis_title,
            title_font=dict(family="Arial, sans-serif", size=16, color="#012973")
        ),
        yaxis=dict(
            title=yaxis_title,
            title_font=dict(family="Arial, sans-serif", size=16, color="#012973")
        ),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    # Add a blue dashed-dot border around the entire chart
    fig.update_layout(
        shapes=[
            dict(
                type="rect", xref="paper", yref="paper",
                x0=0, y0=0, x1=1, y1=1,
                line=dict(color="#0490d7", width=1, dash="dot"),
                fillcolor="rgba(0,0,0,0)"
            )
        ]
    )
    # Add red dashed reference line at y=0
    fig.add_shape(
        type="line", xref="paper", yref="y",
        x0=0, y0=0, x1=1, y1=0,
        line=dict(color="red", width=2, dash="dot")
    )
    return fig

def create_line_trace(df, x_col, y_col, label, color=None, dash=None, line_width=2, hovertemplate=None, marker_symbol="circle", showlegend=True):
    """
    Creates a Plotly Scatter trace for line charts.
    If 'color' is None, Plotly's default colour cycle is used.
    """
    line_opts = {"width": line_width}
    if dash:
        line_opts["dash"] = dash
    if color is not None:
        line_opts["color"] = color

    return go.Scatter(
        x=df[x_col],
        y=df[y_col],
        mode="lines+markers",
        name=label,
        line=line_opts,
        marker=dict(symbol=marker_symbol),
        hovertemplate=hovertemplate or "<b>%{y:.2f}</b><br>",
        showlegend=showlegend
    )

def group_and_sort(data, group_cols, agg_col, date_col="Year-Month", date_format="%d/%m/%Y"):
    """
    Groups data by the specified columns, aggregates the target column by mean,
    converts the date column to datetime, and returns a DataFrame sorted by the date.
    """
    data[date_col] = pd.to_datetime(data[date_col], format=date_format, errors="raise")
    grouped = data.groupby(group_cols, as_index=False)[agg_col].mean()
    return grouped.sort_values(date_col)

# --- Chart Functions ---

def plot_chart_all_products(product, title, desc, data, metric, company, height=200):
    """
    Plots one trace per product category for a specified company.

    Parameters:
        product (str): Expected to be "all" (case-insensitive) in this view.
        title (str): Chart title.
        desc (str): Unused description placeholder.
        data (DataFrame): Full sentiment dataset.
        metric (str): Chosen metric (e.g., "Overall Sentiment Score" or an aspect name).
        company (str): Company name to filter data.
        height (int): Chart height in pixels.

    Dashboard Usage: Used in Company Mode Overview (left/right columns) when product is "All".
    """
    sentiment_column = "Sentiment Score" if metric == "Overall Sentiment Score" else f"{metric}_sentiment_score"

    filtered_data = data[data["Company"].str.contains(company)]
    filtered_data = filtered_data[filtered_data['Final Product Category'] != 'Unknown']
    if filtered_data.empty:
        st.write("No data available.")
        return

    filtered_data["Year-Month"] = pd.to_datetime(filtered_data["Year-Month"], format='%d/%m/%Y', errors='raise')
    grouped = filtered_data.groupby(["Year-Month", "Final Product Category"], as_index=False)[sentiment_column].mean()
    grouped = grouped.sort_values("Year-Month")

    fig = go.Figure()
    for prod_cat in grouped["Final Product Category"].unique():
        prod_data = grouped[grouped["Final Product Category"] == prod_cat]
        label = PRODUCT_CONFIG.emoji_map.get(prod_cat, prod_cat)
        color = PRODUCT_CONFIG.colours.get(prod_cat, None)
        fig.add_trace(create_line_trace(
            df=prod_data,
            x_col="Year-Month",
            y_col=sentiment_column,
            label=label,
            color=color,
            hovertemplate=f"%{{y:.2f}}<br>"
        ))
    fig = apply_chart_style(fig, title=title, yaxis_title=metric, height=height)
    st.plotly_chart(fig, use_container_width=True)

def plot_chart_2(product, title, desc, data, view="aspect"):
    """
    Plots sentiment trends over time for a specific product.

    Parameters:
        product (str): Product name to filter data.
        title (str): Chart title.
        desc (str): Unused description placeholder.
        data (DataFrame): Full sentiment dataset.
        view (str): "aspect" for breakdown by aspect; otherwise, overall sentiment.

    Dashboard Usage: Used in Company Mode Overview when a specific (non "All") product is selected.
    """
    if "all" not in product.lower():
        data = data[data["Final Product Category"].str.contains(product)]
    if data.empty:
        st.warning("No sentiment data available for the selected product.")
        return

    data["Year-Month"] = pd.to_datetime(data["Year-Month"], format='%d/%m/%Y', errors='raise')
    data = data.sort_values("Year-Month")
    fig = go.Figure()
    if view == "aspect":
        for aspect in ASPECT_CONFIG.aspects_map:
            col = f"{aspect}_sentiment_score"
            label = ASPECT_CONFIG.aspects_map.get(aspect, aspect)
            color = ASPECT_CONFIG.aspect_colours.get(aspect, "grey")
            fig.add_trace(create_line_trace(
                df=data,
                x_col="Year-Month",
                y_col=col,
                label=label,
                color=color,
                hovertemplate="<b>%{y:.2f}</b><br>"
            ))
    else:
        fig.add_trace(create_line_trace(
            df=data,
            x_col="Year-Month",
            y_col="Sentiment Score",
            label="Overall Sentiment",
            color="blue" if title == "British Gas" else "maroon",
            hovertemplate="<b>%{y:.2f}</b><br>"
        ))
    fig = apply_chart_style(fig, title=title)
    fig.update_yaxes(range=[-30, 95])
    st.plotly_chart(fig, use_container_width=True)

def plot_aspect_comparison(product, aspect, company, title, desc, data, height=500):
    """
    Compares a single aspect's sentiment trend between the selected company and British Gas.

    Parameters:
        product (str): Product name to filter data.
        aspect (str): Aspect key (e.g. "Appointment Scheduling").
        company (str): Selected company name.
        title (str): Chart title.
        desc (str): Unused description placeholder.
        data (DataFrame): Full sentiment dataset.
        height (int): Chart height in pixels.

    Dashboard Usage: Used in Company Mode’s aspect tabs to compare the selected company vs British Gas.
    """
    aspect_col = f"{aspect}_sentiment_score"
    company_data = data[(data["Company"].str.contains(company)) &
                        (data["Final Product Category"].str.contains(product))]
    if company.lower() != "british gas":
        bg_data = data[(data["Company"].str.contains("British Gas")) &
                       (data["Final Product Category"].str.contains(product))]
    else:
        bg_data = None

    if company_data.empty:
        st.warning(f"No data available for {company}.")
        return

    company_grouped = group_and_sort(company_data, ["Year-Month"], aspect_col)
    if bg_data is not None and not bg_data.empty:
        bg_grouped = group_and_sort(bg_data, ["Year-Month"], aspect_col)
    else:
        bg_grouped = None

    fig = go.Figure()
    fig.add_trace(create_line_trace(
        df=company_grouped,
        x_col="Year-Month",
        y_col=aspect_col,
        label=company,
        color="maroon",
        hovertemplate=f"<b>{company} {aspect}:</b> %{{y:.2f}}<br>"
    ))
    if bg_grouped is not None:
        fig.add_trace(create_line_trace(
            df=bg_grouped,
            x_col="Year-Month",
            y_col=aspect_col,
            label="British Gas",
            color="blue",
            hovertemplate=f"<b>British Gas {aspect}:</b> %{{y:.2f}}<br>"
        ))
    fig = apply_chart_style(fig, title=title, height=height)
    st.plotly_chart(fig, use_container_width=True)

def plot_chart_3(product, aspect, title, desc, data, height=500):
    """
    Plots market sentiment trends for a specific aspect by competitor.

    Parameters:
        product (str): Product name to filter data.
        aspect (str): Aspect key (e.g. "Appointment Scheduling").
        title (str): Chart title.
        desc (str): Unused description placeholder.
        data (DataFrame): Full sentiment dataset.

    Dashboard Usage: Used in Market Mode (non "All") to show aspect sentiment trends by competitor.
    """
    sentiment_col = f"{aspect}_sentiment_score"
    filtered_data = data[data["Final Product Category"].str.contains(product)]
    if filtered_data.empty:
        st.warning("No data available for the selected product.")
        return

    filtered_data["Year-Month"] = pd.to_datetime(filtered_data["Year-Month"], format='%d/%m/%Y', errors='raise')
    filtered_data = filtered_data.sort_values("Year-Month")
    fig = go.Figure()
    for competitor in filtered_data["Company"].unique():
        comp_data = filtered_data[filtered_data["Company"] == competitor]
        comp_grouped = group_and_sort(comp_data, ["Year-Month"], sentiment_col)
        # Fetch the colour from insurer_colours, default to "gray" if not found.
        color = COMPANY_CONFIG.insurer_colours.get(competitor, "gray")
        # Set extra wide line if competitor is British Gas.
        line_width = 4 if competitor == "British Gas" else 2
        # Let Plotly use its default cycle for competitor lines.
        fig.add_trace(create_line_trace(
            df=comp_grouped,
            x_col="Year-Month",
            y_col=sentiment_col,
            label=competitor,
            color=color,
            line_width=line_width,
            hovertemplate=f"%{{y:.2f}}<br>"
        ))
    fig = apply_chart_style(fig, title=title, height=height)
    st.plotly_chart(fig, use_container_width=True)

def plot_product_overall_sentiment(product, title, data, height=500):
    """
    Plots overall sentiment trends for a product across all companies.

    Parameters:
        product (str): Product name.
        title (str): Chart title.
        data (DataFrame): Full sentiment dataset.
        height (int): Chart height in pixels.

    Dashboard Usage: Used in Market Mode when "All" products are selected to display overall sentiment trends.
    """
    product_data = data[data["Final Product Category"] == product]
    if product_data.empty:
        st.write("No data available for this product.")
        return

    fig = go.Figure()
    for company in product_data["Company"].unique():
        comp_data = product_data[product_data["Company"] == company]
        comp_grouped = group_and_sort(comp_data, ["Year-Month"], "Sentiment Score")
        # Use insurer_colours for companies
        color = COMPANY_CONFIG.insurer_colours.get(company, "gray")
        line_width = 4 if company == "British Gas" else 2
        fig.add_trace(create_line_trace(
            df=comp_grouped,
            x_col="Year-Month",
            y_col="Sentiment Score",
            label=company,
            color=color,
            line_width=line_width,
            hovertemplate=f"<b>{company}</b><br>Month: %{{x}}<br>Sentiment: %{{y:.2f}}<br>"
        ))
    fig = apply_chart_style(fig, title=title, height=height)
    fig.update_yaxes(range=[-40, 95])
    st.plotly_chart(fig, use_container_width=True)

def plot_aspect_comparison_hist(product, aspect_col, company, title, desc, data, height=200, metric=None, companies=None):
    """
    Creates a distribution plot (histogram/density plot) comparing a single aspect's
    sentiment scores between the selected company and British Gas for a given product.

    Parameters:
        product (str): The product name to filter the data.
        aspect (str): The aspect key (e.g., "Appointment Scheduling").
        company (str): The selected company.
        title (str): The chart title.
        desc (str): An unused description (placeholder).
        data (DataFrame): The full sentiment dataset.
        height (int): Chart height in pixels.

    Dashboard Usage:
        Use this function in place of plot_aspect_comparison when you want a histogram-style
        comparison rather than a line chart. The colours remain consistent:
        - Selected company: maroon
        - British Gas: blue
    """

    # Filter data for the selected product.
    product_data = data[data["Final Product Category"].str.contains(product)]
    if product_data.empty:
        st.warning("No data available for the selected product.")
        return

    # Filter data for the selected company.
    company_data = product_data[product_data["Company"].str.contains(company)]

    # If the selected company is not British Gas, also filter for British Gas.
    if company.lower() != "british gas":
        bg_data = product_data[product_data["Company"].str.contains("British Gas")]
    else:
        bg_data = None

    # Extract the relevant aspect sentiment scores and drop missing values.
    company_values = company_data[aspect_col].dropna().tolist()
    group_labels = []
    hist_data = []
    colors = []

    if company_values:
        hist_data.append(company_values)
        group_labels.append(company)  # Could also use an emoji label if desired.
        colors.append("maroon")

    if bg_data is not None and not bg_data.empty:
        bg_values = bg_data[aspect_col].dropna().tolist()
        if bg_values:
            hist_data.append(bg_values)
            group_labels.append("British Gas")
            colors.append("blue")

    # If no valid data exists for either group, show a warning.
    if not hist_data:
        st.warning("No valid sentiment scores available to plot.")
        return

    # Create the distribution plot.
    # Here we use a fixed bin_size; adjust as necessary.
    fig = ff.create_distplot(hist_data, group_labels, bin_size=3, colors=colors)

    # Update layout to include title and set height.
    fig.update_layout(
        title=title,
        height=height,
        xaxis=dict(
            title="👈🤬                    ←😟                    Overall Sentiment Score                    😊→                    🥳👉",
            title_font=dict(family="Arial, sans-serif", size=16, color="#012973")
        ),
        yaxis=dict(
            title='Prob Density',
            title_font=dict(family="Arial, sans-serif", size=16, color="#012973")
        ),
        margin=dict(l=50, r=50, t=50, b=50),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Arial", size=14, color="#012973")
    )

    # Display the figure in Streamlit.
    st.plotly_chart(fig, use_container_width=True)
