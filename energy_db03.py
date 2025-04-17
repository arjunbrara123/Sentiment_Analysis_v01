# app.py
"""
Streamlit Dashboard for Energy Reviews Analysis
------------------------------------------------
This dashboard reads in `annotated_reviews.csv` and presents:

📈 Overview        – Review volumes & share‑of‑voice
💬 Sentiment       – Positivity/negativity distributions & trends
📝 Characteristics  – Review structure & customer segments over time
🔍 Topics          – What customers discuss most (raw vs TF–IDF)
⭐ Quality Score    – Composite actionability metric distribution & trends
🌡️ Heat Pumps      – Green/Eco interest as a proxy for heat‑pump launch

Every visualization includes plain‑English insights and suggested next steps.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# -------------------------------------------------------------------
# Page config & CSS
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Energy Reviews Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)
if Path("style.css").exists():
    st.markdown(f"<style>{Path('style.css').read_text()}</style>", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['Year-Month'])
    return df

def style_chart(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="white", plot_bgcolor="white",
        title_font_size=20, font_size=14,
        legend_title_text="", hovermode="x unified"
    )
    return fig

# -------------------------------------------------------------------
# Load data & global filters
# -------------------------------------------------------------------
df = load_data("annotated_reviews.csv")

st.sidebar.header("Global Filters")
providers = st.sidebar.multiselect(
    "Providers",
    options=sorted(df['Company'].unique()),
    default=sorted(df['Company'].unique())
)
df = df[df['Company'].isin(providers)]

# Default date range: Jan 2023–Jan 2025
dmin, dmax = df['Year-Month'].min(), df['Year-Month'].max()
default_start = max(dmin, pd.Timestamp("2023-01-01"))
default_end   = min(dmax, pd.Timestamp("2025-01-31"))

d1, d2 = st.sidebar.date_input(
    "Date Range",
    value=(default_start, default_end),
    min_value=dmin,
    max_value=dmax
)
df = df[(df['Year-Month'] >= pd.to_datetime(d1)) & (df['Year-Month'] <= pd.to_datetime(d2))]

# Precompute monthly totals
monthly = df.groupby('Year-Month').size().reset_index(name='Reviews')

# -------------------------------------------------------------------
# Tabs
# -------------------------------------------------------------------
tabs = st.tabs([
    "📈 Overview",
    "💬 Sentiment",
    "📝 Characteristics",
    "🔍 Topics",
    "⭐ Quality Score",
    "🌡️ Heat Pumps"
])

# -------------------------------------------------------------------
# Tab 1: Overview
# -------------------------------------------------------------------
with tabs[0]:
    st.header("📈 Overview")
    st.markdown("""
    **Purpose:** Executive snapshot of review volume and share‑of‑voice.  
    **Action:** Align support capacity to volume peaks; benchmark top competitors.
    """)

    # Total reviews by provider
    counts = df['Company'].value_counts().reset_index()
    counts.columns = ['Company','Reviews']
    counts['Pct'] = (counts['Reviews']/len(df)*100).round(0)
    fig1 = px.bar(
        counts, x='Company', y='Reviews',
        title="Total Reviews by Provider",
        hover_data={'Pct':':.0f%'}
    )
    st.plotly_chart(style_chart(fig1), use_container_width=True)
    st.markdown(f"- **Octopus** leads with {counts.loc[counts['Company']=='Octopus','Pct'].values[0]}% of all reviews.")

    # Overall monthly volume
    fig2 = px.line(
        monthly, x='Year-Month', y='Reviews',
        title="Total Monthly Review Volume"
    )
    st.plotly_chart(style_chart(fig2), use_container_width=True)
    st.markdown("Spikes align with billing cycles or outages—plan staffing & communications accordingly.")

    # Monthly volume by provider
    vol_cmp = df.groupby(['Year-Month','Company']).size().reset_index(name='Reviews')
    fig3 = px.line(
        vol_cmp, x='Year-Month', y='Reviews', color='Company',
        title="Monthly Review Volume by Provider"
    )
    st.plotly_chart(style_chart(fig3), use_container_width=True)
    st.markdown("Compare share‑of‑voice trends to assess marketing or service event impacts.")

# -------------------------------------------------------------------
# Tab 2: Sentiment
# -------------------------------------------------------------------
with tabs[1]:
    st.header("💬 Sentiment Analysis")
    st.markdown("""
    **Purpose:** Gauge customer tone with:
    - **VADER** (social‑media optimised)  
    - **TextBlob** (general polarity)  
    **Action:** Use VADER dips for language alerts; TextBlob for deeper satisfaction shifts.
    """)

    # Distribution histogram
    melt = df[['sent_tb','sent_vader']].melt(var_name='Method', value_name='Score')
    fig4 = px.histogram(
        melt, x='Score', color='Method', barmode='overlay', nbins=50,
        title="Sentiment Score Distributions"
    )
    fig4.update_yaxes(title="Count")
    st.plotly_chart(style_chart(fig4), use_container_width=True)

    # VADER trend
    vader_trend = df.groupby(['Year-Month','Company'])['sent_vader_n'].mean().reset_index()
    fig5 = go.Figure()
    for comp in vader_trend['Company'].unique():
        d = vader_trend[vader_trend['Company']==comp]
        width = 4 if comp=='British Gas' else 1
        color = 'blue' if comp=='British Gas' else None
        fig5.add_trace(go.Scatter(
            x=d['Year-Month'], y=d['sent_vader_n'], mode='lines',
            name=comp, line=dict(color=color, width=width)
        ))
    fig5.update_layout(
        title="VADER Sentiment Over Time by Provider",
        yaxis_title="VADER [0=neutral→1=positive]"
    )
    st.plotly_chart(style_chart(fig5), use_container_width=True)
    st.markdown("Thick blue = British Gas. Drops signal rising negativity—trigger rapid intervention.")

    # TextBlob trend
    tb_trend = df.groupby(['Year-Month','Company'])['sent_tb_n'].mean().reset_index()
    fig6 = go.Figure()
    for comp in tb_trend['Company'].unique():
        d = tb_trend[tb_trend['Company']==comp]
        width = 4 if comp=='British Gas' else 1
        color = 'blue' if comp=='British Gas' else None
        fig6.add_trace(go.Scatter(
            x=d['Year-Month'], y=d['sent_tb_n'], mode='lines',
            name=comp, line=dict(color=color, width=width)
        ))
    fig6.update_layout(
        title="TextBlob Sentiment Over Time by Provider",
        yaxis_title="TextBlob [0=negative→1=positive]"
    )
    st.plotly_chart(style_chart(fig6), use_container_width=True)
    st.markdown("Use alongside VADER: TextBlob confirms underlying satisfaction shifts vs language intensity.")

# -------------------------------------------------------------------
# Tab 3: Characteristics
# -------------------------------------------------------------------
with tabs[2]:
    st.header("📝 Review Characteristics")
    st.markdown("""
    **Purpose:** Track review length, payment, fuel/tariff mix, switching intent, contact channels,
    and regulatory mentions over time.  
    **Action:** Tailor retention & communications by segment dynamics.
    """)

    # Length distribution
    col_len, col_txt = st.columns([1,2])
    length_dist = (
        df['length_cat']
        .value_counts(normalize=True).mul(100)
        .round(0)
        .rename_axis('Length')
        .reset_index(name='Pct')
    )
    with col_len:
        fig7 = px.pie(length_dist, names='Length', values='Pct', title="Review Length Distribution")
        st.plotly_chart(style_chart(fig7), use_container_width=True, height=250)
    with col_txt:
        st.markdown("""
        - **83%** short (<50 words), **15%** medium, **3%** long.  
        - Longer reviews → higher quality. Use targeted prompts to boost detail.
        """)

    # Payment method trend
    pay_trends = []
    for m in ['Direct Debit','Prepayment Meter']:
        # count reviews per month for this method
        cnt = (
            df[df['payment_method']==m]
            .groupby('Year-Month')
            .size()
            .reset_index(name='Count')
        )
        # bring in the total reviews that month
        cnt = cnt.merge(monthly, on='Year-Month')
        cnt['Pct'] = (cnt['Count'] / cnt['Reviews'] * 100).round(2)
        cnt['Method'] = m
        # keep only the columns we need
        pay_trends.append(cnt[['Year-Month','Pct','Method']])

    pay_df = pd.concat(pay_trends, ignore_index=True)

    if 'index' in pay_df.columns:
        pay_df = pay_df.rename(columns={'index':'Year-Month'})

    fig8 = px.line(
        pay_df, x='Year-Month', y='Pct', color='Method',
        title="% Reviews by Payment Method Over Time"
    )

    st.plotly_chart(style_chart(fig8), use_container_width=True)
    st.markdown("Shows billing-channel shifts—inform targeted billing communications.")

    # Fuel mix trend
    fuel_trends = []
    for cat in ['Gas','Electricity','Dual Fuel']:
        # 1. Count reviews per month for this fuel category
        cnt = (
            df[df['fuel_type']==cat]
            .groupby('Year-Month')
            .size()
            .reset_index(name='Count')  # now a DataFrame with Year-Month & Count
        )

        # 2. Merge in the total monthly reviews
        cnt = cnt.merge(monthly, on='Year-Month')

        # 3. Compute percentage share
        cnt['Pct'] = (cnt['Count'] / cnt['Reviews'] * 100).round(2)

        # 4. Tag the fuel type
        cnt['Fuel'] = cat

        # 5. Keep only the columns we need for plotting
        fuel_trends.append(cnt[['Year-Month','Pct','Fuel']])

    # 6. Concatenate all fuel categories
    fuel_df = pd.concat(fuel_trends, ignore_index=True)

    # 7. Now plot — fuel_df definitely has a Year-Month column
    fig9 = px.line(
        fuel_df,
        x='Year-Month',
        y='Pct',
        color='Fuel',
        title="% Reviews by Fuel Type Over Time"
    )
    st.plotly_chart(style_chart(fig9), use_container_width=True)
    st.markdown("Tailor service messaging by energy source: gas vs electric vs dual fuel.")

    # ─── Tariff mix trend (fixed vs variable) ───
    tariff_trends = []
    for cat in ['Fixed', 'Variable']:
        # 1. Count reviews per month
        cnt = (
            df[df['tariff_type'] == cat]
            .groupby('Year-Month')
            .size()
            .reset_index(name='Count')
        )
        # 2. Merge total-Reviews
        cnt = cnt.merge(monthly, on='Year-Month')
        # 3. Compute percent share
        cnt['Pct'] = (cnt['Count'] / cnt['Reviews'] * 100).round(2)
        # 4. Tag the tariff
        cnt['Tariff'] = cat
        tariff_trends.append(cnt[['Year-Month', 'Pct', 'Tariff']])

    tariff_df = pd.concat(tariff_trends, ignore_index=True)

    fig10 = px.line(
        tariff_df,
        x='Year-Month',
        y='Pct',
        color='Tariff',
        title="% Reviews by Tariff Type Over Time"
    )
    st.plotly_chart(style_chart(fig10), use_container_width=True)
    st.markdown("Track how customers on fixed vs variable plans shift over time—helps tailor rate communications.")

    # ─── Switching intent/outcome trend ───
    switch_trends = []
    for cat in ['Intent', 'Outcome']:
        cnt = (
            df[df['switching'] == cat]
            .groupby('Year-Month')
            .size()
            .reset_index(name='Count')
        )
        cnt = cnt.merge(monthly, on='Year-Month')
        cnt['Pct'] = (cnt['Count'] / cnt['Reviews'] * 100).round(2)
        cnt['Switch'] = cat
        switch_trends.append(cnt[['Year-Month', 'Pct', 'Switch']])

    switch_df = pd.concat(switch_trends, ignore_index=True)

    fig11 = px.line(
        switch_df,
        x='Year-Month',
        y='Pct',
        color='Switch',
        title="% Reviews Mentioning Switching Over Time"
    )
    st.plotly_chart(style_chart(fig11), use_container_width=True)
    st.markdown("Rising “Intent” or “Outcome” mentions flag churn risk—trigger retention offers accordingly.")

    # ─── Regulatory references trend ───
    reg_cnt = (
        df[df['reg_ref_flag'] == 1]
        .groupby('Year-Month')
        .size()
        .reset_index(name='Count')
    )
    reg_cnt = reg_cnt.merge(monthly, on='Year-Month')
    reg_cnt['Pct'] = (reg_cnt['Count'] / reg_cnt['Reviews'] * 100).round(2)

    fig12 = px.line(
        reg_cnt,
        x='Year-Month',
        y='Pct',
        title="% Reviews with Regulatory References Over Time"
    )
    st.plotly_chart(style_chart(fig12), use_container_width=True)
    st.markdown("Tracks formal contract/Ofgem mentions—peaks may indicate billing or compliance issues needing review.")

# -------------------------------------------------------------------
# Tab 4: Topics
# -------------------------------------------------------------------
with tabs[3]:
    st.header("🔍 Topic Relevance")
    st.markdown("""
    **Purpose:** Measure emphasis on six themes via:
    - Raw **keyword counts** (volume signal)  
    - **TF–IDF** weights (distinctiveness signal)  
    **Action:** Prioritize high‑volume issues or detect emerging spikes via TF–IDF.
    """)

    topic_keys = [
        ('billing_pricing','Billing'),
        ('service','Service'),
        ('switching_churn','Churn'),
        ('technical_meter','Technical'),
        ('app_website','Digital'),
        ('green','Green')
    ]
    stats = []
    for key,label in topic_keys:
        stats.append({
            'Topic': label,
            'Avg KW': df[f'kw_{key}'].mean().round(2),
            'Avg TF-IDF': df[f'tfidf_{key}'].mean().round(3)
        })
    tdf = pd.DataFrame(stats).melt(id_vars='Topic', var_name='Metric', value_name='Value')
    fig13 = px.bar(
        tdf, x='Topic', y='Value', color='Metric', barmode='group',
        title="Avg Keyword Count vs TF–IDF by Topic"
    )
    st.plotly_chart(style_chart(fig13), use_container_width=True)

    # Raw keyword trends
    kw_trends = df.groupby('Year-Month')[[f'kw_{k}' for k,_ in topic_keys]].mean().reset_index()
    fig14 = px.line(
        kw_trends, x='Year-Month',
        y=[f'kw_{k}' for k,_ in topic_keys],
        labels={'value':'Avg keyword count','variable':'Topic'},
        title="Raw Keyword Mentions Over Time"
    )
    st.plotly_chart(style_chart(fig14), use_container_width=True)
    st.markdown("High-volume themes—good for resource allocation but may include common words.")

    # TF–IDF trends
    tfidf_trends = df.groupby('Year-Month')[[f'tfidf_{k}' for k,_ in topic_keys]].mean().reset_index()
    fig15 = px.line(
        tfidf_trends, x='Year-Month',
        y=[f'tfidf_{k}' for k,_ in topic_keys],
        labels={'value':'Avg TF–IDF','variable':'Topic'},
        title="TF–IDF Topic Relevance Over Time"
    )
    st.plotly_chart(style_chart(fig15), use_container_width=True)
    st.markdown("TF–IDF highlights distinctive spikes—ideal for spotting new or urgent issues.")

# -------------------------------------------------------------------
# Tab 5: Quality Score
# -------------------------------------------------------------------
with tabs[4]:
    st.header("⭐ Quality Score")
    st.markdown("""
    **Purpose:** Composite measure of sentiment, depth, verbosity, and actionable flags.  
    **Action:** Focus on top‑quality reviews (>0.5) for case studies; share best practices.
    """)

    # Distribution
    fig16 = px.histogram(df, x='quality_score', nbins=50, title="Quality Score Distribution")
    st.plotly_chart(style_chart(fig16), use_container_width=True)
    st.markdown("Median ~0.24; top decile (>0.5) yields the richest, actionable feedback.")

    # Trend by provider
    q_trend = df.groupby(['Year-Month','Company'])['quality_score'].mean().reset_index()
    fig17 = go.Figure()
    for comp in q_trend['Company'].unique():
        d = q_trend[q_trend['Company']==comp]
        width = 4 if comp=='British Gas' else 1
        color = 'blue' if comp=='British Gas' else None
        fig17.add_trace(go.Scatter(
            x=d['Year-Month'], y=d['quality_score'], mode='lines',
            name=comp, line=dict(color=color, width=width)
        ))
    fig17.update_layout(title="Quality Score Over Time by Provider", yaxis_title="Quality Score")
    st.plotly_chart(style_chart(fig17), use_container_width=True)

    # Avg sentiment vs avg quality bubble chart
    summary = df.groupby('Company').agg({
        'sent_vader_n':'mean',
        'quality_score':'mean',
        'Company':'count'
    }).rename(columns={'Company':'Reviews'}).reset_index()
    fig18 = px.scatter(
        summary, x='sent_vader_n', y='quality_score',
        size='Reviews', color='Company',
        title="Avg Sentiment vs Avg Quality by Provider",
        labels={'sent_vader_n':'Avg VADER','quality_score':'Avg Quality','Reviews':'#Reviews'},
        size_max=60
    )
    fig18.update_traces(
        selector=lambda tr: tr.name=='British Gas',
        marker=dict(line=dict(width=2, color='blue'))
    )
    st.plotly_chart(style_chart(fig18), use_container_width=True)
    st.markdown("Bubble size = review volume. Upper‑right quadrant = best overall feedback.")

# -------------------------------------------------------------------
# Tab 6: Heat Pumps
# -------------------------------------------------------------------
with tabs[5]:
    st.header("🌡️ Heat Pumps Viability")
    st.markdown("""
    **Purpose:** Use Green/Eco topic chatter as a proxy for interest in sustainable tech.  
    **Action:** Launch heat‑pump campaigns when green mentions peak; partner with eco‑leaders.
    """)

    # Overall green trend
    green_ts = df.groupby('Year-Month')['kw_green'].mean().reset_index()
    fig19 = px.line(green_ts, x='Year-Month', y='kw_green', title="Avg Green Mentions Over Time")
    st.plotly_chart(style_chart(fig19), use_container_width=True)
    st.markdown("Rising green mentions show growing eco‑awareness; ideal timing for green product pushes.")

    # Green by provider
    green_cmp = df.groupby(['Year-Month','Company'])['kw_green'].mean().reset_index()
    fig20 = go.Figure()
    for comp in green_cmp['Company'].unique():
        d = green_cmp[green_cmp['Company']==comp]
        width = 4 if comp=='British Gas' else 1
        color = 'blue' if comp=='British Gas' else None
        fig20.add_trace(go.Scatter(
            x=d['Year-Month'], y=d['kw_green'], mode='lines',
            name=comp, line=dict(color=color, width=width)
        ))
    fig20.update_layout(title="Green Mentions by Provider", yaxis_title="Avg green keywords")
    st.plotly_chart(style_chart(fig20), use_container_width=True)
    st.markdown("British Gas leads eco conversation—prime pilot partner for green offerings.")

    # ─── Sentiment on Green Reviews Over Time ───
    # 1. Filter down to only green‑mention reviews, then compute monthly avg sentiment per provider
    green_trend = (
        df[df['kw_green'] > 0]
        .groupby(['Year-Month','Company'])['sent_vader_n']
        .mean()
        .reset_index()
    )

    # 2. Build a multi‑line chart
    fig21 = px.line(
        green_trend,
        x='Year-Month',
        y='sent_vader_n',
        color='Company',
        title="VADER Sentiment on Green‑Topic Reviews Over Time by Provider",
        labels={'sent_vader_n':'Avg VADER Sentiment (0–1)'}
    )

    # 3. Thicken British Gas line and colour it blue
    for trace in fig21.data:
        if trace.name == 'British Gas':
            trace.update(line=dict(color='blue', width=4))
        else:
            trace.update(line=dict(width=1))

    st.plotly_chart(style_chart(fig21), use_container_width=True)

    st.markdown("""
    **Interpretation:**  
    - Tracks how positivity on “green” issues evolves.  
    - Thick blue = British Gas; compare slopes to spot which providers are increasing eco‑interest.  
    - A rising trend signals growing eco‑awareness—ideal timing for sustainable product launches.
    """)
