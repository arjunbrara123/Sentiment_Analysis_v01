"""
Iman-Conover Method: Interactive Teaching Dashboard
A visual, step-by-step guide to understanding correlated random sampling
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Page config
st.set_page_config(
    page_title="Understanding Correlated Random Sampling",
    page_icon="🎲",
    layout="wide"
)

# Inject optional brand CSS ----------------------------------------------------
try:
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass  # run fine without custom styling

# Custom CSS for better visuals
st.markdown("""
<style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .medium-font {
        font-size:16px !important;
    }
    .highlight-box {
        padding: 20px;
        background-color: #f0f2f6;
        border-radius: 10px;
        margin: 10px 0;
    }
    .success-box {
        padding: 15px;
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        margin: 10px 0;
    }
    .info-box {
        padding: 15px;
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and Introduction
st.title("🎲 The Magic of Correlated Random Numbers")
st.markdown("### *Learn how to create realistic scenarios where things affect each other*")

# Initialize session state
if 'generated_data' not in st.session_state:
    st.session_state.generated_data = False
    st.session_state.X = None
    st.session_state.Y = None
    st.session_state.Z = None
    st.session_state.E = None
    st.session_state.ranks = None
    st.session_state.L = None

# Sidebar for global parameters
with st.sidebar:

    st.image("images/company_logo.png")  # top‑left logo
    st.header("🎛️ Control Panel")
    st.markdown("*Adjust these settings to see how they affect the results*")

    st.subheader("📊 Sample Size")
    n_trials = st.slider(
        "How many scenarios to generate?",
        min_value=20,
        max_value=500,
        value=100,
        step=10,
        help="More scenarios = smoother patterns"
    )

    st.subheader("🎲 Randomness")
    use_seed = st.checkbox("Use fixed random seed?", value=True,
                           help="Check this to get the same results each time")
    if use_seed:
        seed = st.number_input("Seed value", value=42, min_value=0)
    else:
        seed = None

    st.subheader("📦 Products")
    st.markdown("*We'll simulate costs for these products*")
    products = ["Boiler", "Drains", "Electrics"]

    # Correlation settings
    st.subheader("🔗 How Products Relate")
    st.markdown("*How strongly should costs move together?*")

    corr_boiler_drains = st.slider(
        "Boiler ↔ Drains correlation",
        min_value=-1.0,
        max_value=1.0,
        value=0.8,
        step=0.1,
        help="Positive = costs rise together, Negative = one rises when other falls"
    )

    corr_boiler_electrics = st.slider(
        "Boiler ↔ Electrics correlation",
        min_value=-1.0,
        max_value=1.0,
        value=-0.2,
        step=0.1
    )

    corr_drains_electrics = st.slider(
        "Drains ↔ Electrics correlation",
        min_value=-1.0,
        max_value=1.0,
        value=-0.1,
        step=0.1
    )

    # Build correlation matrix
    R_target = np.array([
        [1.0, corr_boiler_drains, corr_boiler_electrics],
        [corr_boiler_drains, 1.0, corr_drains_electrics],
        [corr_boiler_electrics, corr_drains_electrics, 1.0]
    ])

    if st.button("🚀 Generate New Data", type="primary"):
        st.session_state.generated_data = False

# Create tabs for each step
tabs = st.tabs([
    "📚 Overview",
    "🎯 Step 1: Independent Values",
    "🔍 Step 2: Check Independence",
    "🎨 Step 3: Design Relationships",
    "🧮 Step 4: Math Magic (L Matrix)",
    "🎲 Step 5: Random Scores",
    "🔗 Step 6: Linked Scores",
    "📊 Step 7: Ranking",
    "📦 Step 8: Sorting Values",
    "✨ Step 9: The Shuffle!",
    "🎉 Step 10: Final Results",
    "🎮 Playground"
])


# Helper function to generate data
def generate_all_data(n_trials, products, R_target, seed=None):
    if seed is not None:
        np.random.seed(seed)

    d = len(products)

    # Marginal parameters (lognormal)
    mu_log = np.array([4.6, 4.2, 3.8])
    sigma_log = np.array([0.50, 0.55, 0.60])

    # Step 1: Independent samples
    Z_raw = np.random.normal(size=(n_trials, d))
    X = np.exp(mu_log + sigma_log * Z_raw)

    # Check if R_target is valid (PSD)
    eigs = np.linalg.eigvalsh(R_target)
    if np.min(eigs) < -1e-10:
        # Project to nearest PSD
        eigvals, eigvecs = np.linalg.eigh(R_target)
        eigvals_clipped = np.clip(eigvals, 1e-10, None)
        R_psd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
        dvec = np.sqrt(np.diag(R_psd))
        Dinv = np.diag(1.0 / dvec)
        R_used = Dinv @ R_psd @ Dinv
        np.fill_diagonal(R_used, 1.0)
    else:
        R_used = R_target.copy()

    # Step 4: Factor L
    try:
        L = np.linalg.cholesky(R_used)
    except:
        eigvals, eigvecs = np.linalg.eigh(R_used)
        eigvals_clipped = np.clip(eigvals, 0.0, None)
        L = eigvecs @ np.diag(np.sqrt(eigvals_clipped))

    # Steps 5-6: Correlated scores
    E = np.random.normal(size=(n_trials, d))
    Z = E @ L.T

    # Step 7: Ranks
    ranks = np.zeros_like(Z, dtype=int)
    order_indices = {}
    for j in range(d):
        order = np.argsort(Z[:, j])
        order_indices[j] = order
        ranks[order, j] = np.arange(1, n_trials + 1)

    # Step 9: Shuffle
    Y = np.empty_like(X)
    for j in range(d):
        order = order_indices[j]
        sorted_vals = np.sort(X[:, j])
        Y[order, j] = sorted_vals

    return X, Y, Z, E, ranks, L, R_used


# Tab 0: Overview
with tabs[0]:
    st.header("🌟 Welcome to the World of Correlated Random Numbers!")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        <div class='highlight-box'>
        <h3>🤔 Why Do We Need This?</h3>

        Imagine you're planning for claim costs for 100 months. You need to estimate costs for:
        - 🔥 **Boilers** 
        - 🚿 **Drains**
        - ⚡ **Electrical systems**

        **The Challenge:** These costs aren't independent! When one system is expensive (maybe due to 
        cold weather, rainfall, or other linked factors), others tend to be expensive too.

        **The Solution:** The Iman-Conover method lets us create realistic scenarios where:
        1. Each product keeps its own cost pattern (some boilers have low claim rates, some are very high)
        2. BUT costs are related (high boiler claims often mean high plumbing & drain claims)
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='info-box'>
        <h3>🎯 Real-World Applications</h3>

        - **Insurance Risk:** Model related risks (fire → water damage)
        - **Investment Risk:** Simulate correlated stock movements
        - **Operational Risk (eg. Supply Chain):** Plan for related disruptions
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='success-box'>
        <h3>📖 How to Use This Tool</h3>

        1. **Adjust settings** in the sidebar
        2. **Click through tabs** to see each step
        3. **Play with values** to see effects
        4. **Use the Playground** to experiment

        Each tab shows one step of the process with clear visuals!
        </div>
        """, unsafe_allow_html=True)

        # Visual correlation explanation
        fig = go.Figure()

        # Positive correlation example
        x_pos = np.random.normal(0, 1, 50)
        y_pos = 0.8 * x_pos + np.random.normal(0, 0.5, 50)

        # Negative correlation example
        x_neg = np.random.normal(4, 1, 50)
        y_neg = -0.8 * x_neg + np.random.normal(4, 0.5, 50)

        fig.add_trace(go.Scatter(
            x=x_pos, y=y_pos,
            mode='markers',
            name='Positive Correlation',
            marker=dict(color='green', size=8),
            showlegend=True
        ))

        fig.add_trace(go.Scatter(
            x=x_neg, y=y_neg,
            mode='markers',
            name='Negative Correlation',
            marker=dict(color='red', size=8),
            showlegend=True
        ))

        fig.update_layout(
            title="What is Correlation?",
            xaxis_title="Product A Claims",
            yaxis_title="Product B Claims",
            height=300,
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)

# Generate data if needed
if not st.session_state.generated_data:
    X, Y, Z, E, ranks, L, R_used = generate_all_data(n_trials, products, R_target, seed)
    st.session_state.X = X
    st.session_state.Y = Y
    st.session_state.Z = Z
    st.session_state.E = E
    st.session_state.ranks = ranks
    st.session_state.L = L
    st.session_state.R_used = R_used
    st.session_state.generated_data = True

# Tab 1: Independent Values
with tabs[1]:
    st.header("Step 1: Generate Independent Random Values")

    st.markdown("""
    <div class='info-box'>
    <b>🎲 What's happening here?</b><br>
    We're rolling dice for each product separately. Each product has its own pattern 
    (some tend to be more expensive), but there's <b>no connection</b> between them yet.
    Think of it like three people independently rolling dice - their results don't affect each other.
    </div>
    """, unsafe_allow_html=True)

    # Show the data
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📊 First 10 Scenarios")
        df_display = pd.DataFrame(st.session_state.X[:10], columns=products)
        df_display.index = [f"Month {i + 1}" for i in range(10)]
        st.dataframe(df_display.style.format("{:.0f}"), use_container_width=True)

        st.markdown("""
        <div class='highlight-box'>
        <b>👀 Notice:</b> Each column (product) has its own range of values. 
        Boilers tend to be most expensive, Electrics least expensive.
        But there's no pattern between columns - high Boiler claims cost doesn't mean high Drains claims cost.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.subheader("📈 Distribution of Each Product")

        fig = make_subplots(rows=1, cols=3, subplot_titles=products)

        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

        for i, product in enumerate(products):
            fig.add_trace(
                go.Histogram(x=st.session_state.X[:, i],
                             nbinsx=20,
                             marker_color=colors[i],
                             showlegend=False),
                row=1, col=i + 1
            )

        fig.update_layout(height=300, title_text="Cost Distributions (Independent)")
        fig.update_xaxes(title_text="Cost ($)")
        fig.update_yaxes(title_text="Frequency", row=1, col=1)

        st.plotly_chart(fig, use_container_width=True)

    # Key insight
    st.markdown("""
    <div class='success-box'>
    <h3>💡 Key Insight</h3>
    Right now, knowing that boilers have high claims in a month tells us <b>nothing</b> about 
    drain or electrical claim rates. This is unrealistic - claim rates between products are related!
    </div>
    """, unsafe_allow_html=True)

# Tab 2: Check Independence
with tabs[2]:
    st.header("Step 2: Verify Independence")

    st.markdown("""
    <div class='info-box'>
    <b>🔍 What are we checking?</b><br>
    We're measuring if the products are related. A correlation of 0 means "completely independent" 
    - like strangers at a bus stop. We expect all correlations to be near 0 since we haven't 
    connected them yet.
    </div>
    """, unsafe_allow_html=True)

    # Calculate correlations
    df_X = pd.DataFrame(st.session_state.X, columns=products)
    corr_before = df_X.corr(method='spearman')

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📊 Correlation Heatmap (Before)")

        fig = go.Figure(data=go.Heatmap(
            z=corr_before.values,
            x=products,
            y=products,
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            text=np.round(corr_before.values, 2),
            texttemplate='%{text}',
            textfont={"size": 16},
            colorbar=dict(title="Correlation")
        ))

        fig.update_layout(
            title="How Related Are The Products?",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🎯 What Do These Numbers Mean?")

        st.markdown("""
        <div class='highlight-box'>
        <b>Reading the Correlation Scale:</b>

        - **1.0** = Perfect match (always move together) 🟦
        - **0.5** = Often move together 🟦
        - **0.0** = No relationship (independent) ⬜
        - **-0.5** = Often move opposite 🟥
        - **-1.0** = Perfect opposite (one up, other down) 🟥

        <br><b>Current Result:</b> All near 0 = Independent! ✅
        </div>
        """, unsafe_allow_html=True)

        # Scatter plot to visualize
        st.subheader("📈 Example: Boiler vs Drains")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=st.session_state.X[:, 0],
            y=st.session_state.X[:, 1],
            mode='markers',
            marker=dict(
                color=st.session_state.X[:, 2],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Electrics<br>Cost"),
                size=8
            ),
            text=[f"Month {i + 1}" for i in range(len(st.session_state.X))],
            hovertemplate="<b>%{text}</b><br>Boiler: $%{x:.0f}<br>Drains: $%{y:.0f}<extra></extra>"
        ))

        fig.update_layout(
            xaxis_title="Boiler Cost ($)",
            yaxis_title="Drains Cost ($)",
            height=350,
            title="No Pattern = Independent"
        )

        st.plotly_chart(fig, use_container_width=True)

# Tab 3: Design Relationships
with tabs[3]:
    st.header("Step 3: Design Your Target Relationships")

    st.markdown("""
    <div class='info-box'>
    <b>🎨 Your Design Studio</b><br>
    This is where you decide how products should relate to each other. You're the architect 
    of relationships! Use the sidebar sliders to adjust correlations and see the target pattern.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🎯 Your Target Correlation Matrix")

        # Check if matrix is valid
        eigs = np.linalg.eigvalsh(R_target)
        is_valid = np.min(eigs) >= -1e-10

        if not is_valid:
            st.warning("⚠️ Invalid correlation combination! Auto-adjusting to nearest valid matrix...")
            R_display = st.session_state.R_used
        else:
            R_display = R_target

        fig = go.Figure(data=go.Heatmap(
            z=R_display,
            x=products,
            y=products,
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            text=np.round(R_display, 2),
            texttemplate='%{text}',
            textfont={"size": 16},
            colorbar=dict(title="Target<br>Correlation")
        ))

        fig.update_layout(
            title="Relationships You Want",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🎭 Interpreting Your Design")

        # Interpret the correlations
        interpretations = []

        if corr_boiler_drains > 0.5:
            interpretations.append("🔥🚿 **Strong link**: When boilers are expensive, drains usually are too!")
        elif corr_boiler_drains > 0:
            interpretations.append("🔥🚿 **Mild link**: Boiler and drain costs tend to rise together")
        elif corr_boiler_drains < -0.5:
            interpretations.append("🔥🚿 **Strong opposite**: Expensive boilers mean cheap drains!")
        elif corr_boiler_drains < 0:
            interpretations.append("🔥🚿 **Mild opposite**: Higher boiler costs slightly reduce drain costs")
        else:
            interpretations.append("🔥🚿 **Independent**: Boiler and drain costs are unrelated")

        if corr_boiler_electrics > 0.5:
            interpretations.append("🔥⚡ **Strong link**: Boiler and electrical costs move together")
        elif corr_boiler_electrics < -0.5:
            interpretations.append("🔥⚡ **Strong opposite**: High boiler costs mean low electrical costs")
        elif abs(corr_boiler_electrics) < 0.2:
            interpretations.append("🔥⚡ **Weak/No link**: Boiler and electrical costs barely relate")

        for interp in interpretations:
            st.markdown(interp)

        st.markdown("""
        <div class='highlight-box'>
        <b>🏗️ Real-World Example:</b><br>
        When the weather is below zero and boiler claims rise (expensive), 
        the plumbing (drains) usually has frozen pips and spike as well. But maybe electrical 
        was updated separately, so it's independent or even cheaper (newer = more efficient).
        </div>
        """, unsafe_allow_html=True)

        # Matrix validity check
        if is_valid:
            st.success("✅ Your correlation design is mathematically valid!")
        else:
            st.info("📐 Adjusted to nearest valid correlation matrix")

# Tab 4: Math Magic (L Matrix)
with tabs[4]:
    st.header("Step 4: The Mathematical Magic (Creating the L Factor)")

    st.markdown("""
    <div class='info-box'>
    <b>🧙‍♂️ Behind the Curtain</b><br>
    This step involves some mathematical magic. Think of it like creating a "recipe" that will 
    help us mix independent ingredients to get the relationships we want. Don't worry about the 
    details - just know this creates the "linking instructions"!
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📐 The L Matrix (Linking Recipe)")

        # Display L matrix
        df_L = pd.DataFrame(st.session_state.L,
                            index=products,
                            columns=[f"Factor {i + 1}" for i in range(len(products))])

        fig = go.Figure(data=go.Heatmap(
            z=st.session_state.L,
            x=[f"Factor {i + 1}" for i in range(len(products))],
            y=products,
            colorscale='Viridis',
            text=np.round(st.session_state.L, 2),
            texttemplate='%{text}',
            textfont={"size": 14}
        ))

        fig.update_layout(
            title="The Magic Mixing Recipe",
            height=350
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class='highlight-box'>
        <b>🎨 What does this do?</b><br>
        Each row is like a recipe for one product:
        - Take some of Factor 1
        - Mix in some Factor 2
        - Add a dash of Factor 3

        This creates the correlation pattern we want!
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.subheader("🔄 Verification: Does our recipe work?")

        # Verify L @ L.T ≈ R
        reconstructed = st.session_state.L @ st.session_state.L.T

        fig = go.Figure(data=go.Heatmap(
            z=reconstructed,
            x=products,
            y=products,
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            text=np.round(reconstructed, 2),
            texttemplate='%{text}',
            textfont={"size": 14}
        ))

        fig.update_layout(
            title="Recipe Test (Should Match Target)",
            height=350
        )

        st.plotly_chart(fig, use_container_width=True)

        # Check how close we are
        diff = np.max(np.abs(reconstructed - st.session_state.R_used))
        if diff < 0.01:
            st.success(f"✅ Perfect recipe! Maximum difference: {diff:.6f}")
        else:
            st.warning(f"⚠️ Close but not perfect. Maximum difference: {diff:.6f}")

        st.markdown("""
        <div class='success-box'>
        <b>💡 Simple Analogy:</b><br>
        Imagine you want to create three paint colors that relate to each other 
        (some similar, some different). The L matrix is like your mixing guide - 
        how much of each base color to use for each final color.
        </div>
        """, unsafe_allow_html=True)

# Tab 5: Random Scores
with tabs[5]:
    st.header("Step 5: Generate Independent Random Scores")

    st.markdown("""
    <div class='info-box'>
    <b>🎲 Starting Fresh</b><br>
    We create a new set of random numbers (scores) - one for each product in each month. 
    These are like "ranking seeds" that we'll transform in the next step. Think of them as 
    initial lottery numbers before we link them together.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📊 Independent Scores (E)")

        # Show first few scores
        df_E = pd.DataFrame(st.session_state.E[:10], columns=products)
        df_E.index = [f"Month {i + 1}" for i in range(10)]

        st.dataframe(df_E.style.format("{:.2f}").background_gradient(cmap='coolwarm', axis=None),
                     use_container_width=True)

        st.markdown("""
        <div class='highlight-box'>
        <b>📏 What are these numbers?</b><br>
        - Standard normal scores (average = 0)
        - Completely independent (no relationships)
        - Will be transformed to create correlations
        - Think of them as "raw material" for relationships
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.subheader("📈 Score Distributions")

        fig = make_subplots(rows=1, cols=3, subplot_titles=products)

        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

        for i, product in enumerate(products):
            fig.add_trace(
                go.Histogram(x=st.session_state.E[:, i],
                             nbinsx=20,
                             marker_color=colors[i],
                             showlegend=False),
                row=1, col=i + 1
            )

        fig.update_layout(height=300, title_text="Independent Score Distributions")
        fig.update_xaxes(title_text="Score Value")
        fig.update_yaxes(title_text="Count", row=1, col=1)

        st.plotly_chart(fig, use_container_width=True)

        # Correlation check
        st.subheader("🔍 Independence Check")

        df_E_full = pd.DataFrame(st.session_state.E, columns=products)
        corr_E = df_E_full.corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr_E.values,
            x=products,
            y=products,
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            text=np.round(corr_E.values, 2),
            texttemplate='%{text}',
            textfont={"size": 14}
        ))

        fig.update_layout(
            title="Scores Are Independent (Near 0)",
            height=300
        )

        st.plotly_chart(fig, use_container_width=True)

# Tab 6: Linked Scores
with tabs[6]:
    st.header("Step 6: Transform to Correlated Scores")

    st.markdown("""
    <div class='info-box'>
    <b>🔗 The Transformation</b><br>
    Now we apply our "recipe" (L matrix) to mix the independent scores into correlated ones. 
    This is like taking three separate radio stations and tuning them so some play similar music!
    The formula: Z = E × L^T
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📊 Correlated Scores (Z)")

        # Show first few correlated scores
        df_Z = pd.DataFrame(st.session_state.Z[:10], columns=products)
        df_Z.index = [f"Month {i + 1}" for i in range(10)]

        st.dataframe(df_Z.style.format("{:.2f}").background_gradient(cmap='coolwarm', axis=None),
                     use_container_width=True)

        # Show the transformation visually
        st.subheader("🔄 Before → After Transformation")

        # Create scatter plot showing transformation
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["Before: Independent (E)", "After: Correlated (Z)"])

        # Before
        fig.add_trace(
            go.Scatter(x=st.session_state.E[:, 0], y=st.session_state.E[:, 1],
                       mode='markers', marker=dict(color='gray', size=5),
                       showlegend=False),
            row=1, col=1
        )

        # After
        fig.add_trace(
            go.Scatter(x=st.session_state.Z[:, 0], y=st.session_state.Z[:, 1],
                       mode='markers', marker=dict(color='purple', size=5),
                       showlegend=False),
            row=1, col=2
        )

        fig.update_xaxes(title_text=f"{products[0]} Score")
        fig.update_yaxes(title_text=f"{products[1]} Score", row=1, col=1)
        fig.update_layout(height=350)

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("✅ Correlation Achievement!")

        # Check correlation of Z
        df_Z_full = pd.DataFrame(st.session_state.Z, columns=products)
        corr_Z = df_Z_full.corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr_Z.values,
            x=products,
            y=products,
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            text=np.round(corr_Z.values, 2),
            texttemplate='%{text}',
            textfont={"size": 14}
        ))

        fig.update_layout(
            title="Achieved Correlations in Z",
            height=350
        )

        st.plotly_chart(fig, use_container_width=True)

        # Compare to target
        st.subheader("🎯 How close to target?")

        diff_matrix = corr_Z.values - st.session_state.R_used
        max_diff = np.max(np.abs(diff_matrix[np.triu_indices(3, k=1)]))

        if max_diff < 0.1:
            st.success(f"✅ Excellent! Maximum difference from target: {max_diff:.3f}")
        else:
            st.warning(f"⚠️ Good but not perfect. Maximum difference: {max_diff:.3f}")

        st.markdown("""
        <div class='success-box'>
        <b>🎉 What we've achieved:</b><br>
        The scores now have the correlation pattern we designed! 
        When Boiler scores are high, Drain scores tend to be high too 
        (if we set positive correlation). The "recipe" worked!
        </div>
        """, unsafe_allow_html=True)

# Tab 7: Ranking
with tabs[7]:
    st.header("Step 7: Convert Scores to Rankings")

    st.markdown("""
    <div class='info-box'>
    <b>🏆 Creating the Pecking Order</b><br>
    We rank each month for each product based on their scores. Months with lowest score 
    gets rank 1, highest gets rank N. These ranks will determine which actual cost values 
    each month gets. It's like creating a seating chart based on lottery numbers!
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📊 Score Rankings")

        # Show first few ranks
        df_ranks = pd.DataFrame(st.session_state.ranks[:15],
                                columns=[f"{p} Rank" for p in products])
        df_ranks.index = [f"Month {i + 1}" for i in range(15)]

        # Create a color-coded display
        st.dataframe(df_ranks.style.background_gradient(cmap='RdYlGn_r', axis=0),
                     use_container_width=True)

        st.markdown("""
        <div class='highlight-box'>
        <b>📖 How to read this:</b><br>
        - Rank 1 = Will get the LOWEST cost for that product
        - Higher rank = Will get progressively higher costs
        - Months with correlated high ranks will have high costs for multiple products
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.subheader("🔄 Rank Patterns")

        # Show rank correlation
        df_ranks_full = pd.DataFrame(st.session_state.ranks, columns=products)
        rank_corr = df_ranks_full.corr()

        fig = go.Figure(data=go.Heatmap(
            z=rank_corr.values,
            x=products,
            y=products,
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            text=np.round(rank_corr.values, 2),
            texttemplate='%{text}',
            textfont={"size": 14}
        ))

        fig.update_layout(
            title="Rank Correlations (Should Match Target)",
            height=350
        )

        st.plotly_chart(fig, use_container_width=True)

        # Visualize rank relationships
        st.subheader("👀 Rank Relationships")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=st.session_state.ranks[:, 0],
            y=st.session_state.ranks[:, 1],
            mode='markers',
            marker=dict(
                color=st.session_state.ranks[:, 2],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=f"{products[2]}<br>Rank"),
                size=8
            ),
            text=[f"Month {i + 1}" for i in range(len(st.session_state.ranks))],
            hovertemplate="<b>%{text}</b><br>%{xaxis.title.text}: %{x}<br>%{yaxis.title.text}: %{y}<extra></extra>"
        ))

        fig.update_layout(
            xaxis_title=f"{products[0]} Rank",
            yaxis_title=f"{products[1]} Rank",
            height=350,
            title="Months with Similar Ranks Across Products"
        )

        st.plotly_chart(fig, use_container_width=True)

# Tab 8: Sorting Values
with tabs[8]:
    st.header("Step 8: Sort Original Values")

    st.markdown("""
    <div class='info-box'>
    <b>📦 Preparing for the Shuffle</b><br>
    We take our original cost values (from Step 1) and sort them from lowest to highest 
    for each product. These sorted values will be distributed to months based on their 
    ranks. It's like arranging prizes from smallest to largest before a raffle!
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📊 Sorted Cost Values")

        # Create sorted values
        sorted_values = np.sort(st.session_state.X, axis=0)
        df_sorted = pd.DataFrame(sorted_values[:20], columns=products)
        df_sorted.index = [f"Rank {i + 1}" for i in range(20)]

        st.dataframe(df_sorted.style.format("{:.0f}").background_gradient(cmap='YlOrRd', axis=0),
                     use_container_width=True)

        st.markdown("""
        <div class='highlight-box'>
        <b>🎁 The Prize Pool:</b><br>
        - Rank 1 gets the smallest value (cheapest cost)
        - Rank 2 gets the second smallest
        - And so on...
        - Same values as Step 1, just sorted!
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.subheader("📈 Value Distribution")

        # Show the distribution remains the same
        fig = go.Figure()

        for i, product in enumerate(products):
            # Original values
            fig.add_trace(go.Box(
                y=st.session_state.X[:, i],
                name=f"{product} (Original)",
                marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'][i],
                boxmean='sd'
            ))

            # Sorted values (same distribution)
            fig.add_trace(go.Box(
                y=sorted_values[:, i],
                name=f"{product} (Sorted)",
                marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'][i],
                opacity=0.5,
                boxmean='sd'
            ))

        fig.update_layout(
            title="Same Values, Just Reordered",
            yaxis_title="Cost ($)",
            showlegend=True,
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class='success-box'>
        <b>💡 Key Point:</b><br>
        Sorting doesn't change the distribution! We still have the same 
        mix of cheap and expensive months for each product. We're just 
        organizing them for the shuffle.
        </div>
        """, unsafe_allow_html=True)

# Tab 9: The Shuffle
with tabs[9]:
    st.header("Step 9: The Magic Shuffle! ✨")

    st.markdown("""
    <div class='info-box'>
    <b>🎪 The Grand Finale!</b><br>
    This is where the magic happens! We assign the sorted cost values to months based on 
    their ranks. Months with rank 1 get the cheapest costs, rank 2 gets second cheapest, etc. 
    Since ranks are correlated, the final costs will be too!
    </div>
    """, unsafe_allow_html=True)

    # Animated explanation
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown("""
        <div class='highlight-box'>
        <h4>1️⃣ We Have Rankings</h4>
        Month A: Rank 5, 4, 6<br>
        Month B: Rank 1, 2, 1<br>
        Month C: Rank 8, 9, 7<br>
        (Correlated across products)
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='highlight-box'>
        <h4>2️⃣ We Have Sorted Costs</h4>
        Rank 1 → $50<br>
        Rank 2 → $75<br>
        Rank 3 → $90<br>
        ...(for each product)
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class='success-box'>
        <h4>3️⃣ Match Them Up!</h4>
        Month with rank 1<br>
        gets cost for rank 1<br><br>
        Result: Correlated costs!
        </div>
        """, unsafe_allow_html=True)

    # Show the actual shuffle result
    st.subheader("🎯 The Shuffle Result")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("**Before Shuffle (Independent)**")
        df_before = pd.DataFrame(st.session_state.X[:10], columns=products)
        df_before.index = [f"Month {i + 1}" for i in range(10)]
        st.dataframe(df_before.style.format("{:.0f}"), use_container_width=True)

    with col2:
        st.markdown("**After Shuffle (Correlated)**")
        df_after = pd.DataFrame(st.session_state.Y[:10], columns=products)
        df_after.index = [f"Month {i + 1}" for i in range(10)]
        st.dataframe(df_after.style.format("{:.0f}").background_gradient(cmap='YlOrRd', axis=1),
                     use_container_width=True)

    # Visualize the change
    st.subheader("📊 Visualizing the Transformation")

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Before: Independent", "After: Correlated"])

    # Before
    fig.add_trace(
        go.Scatter(x=st.session_state.X[:, 0], y=st.session_state.X[:, 1],
                   mode='markers', marker=dict(color='gray', size=8),
                   name="Before", showlegend=False),
        row=1, col=1
    )

    # After
    fig.add_trace(
        go.Scatter(x=st.session_state.Y[:, 0], y=st.session_state.Y[:, 1],
                   mode='markers',
                   marker=dict(color='purple', size=8),
                   name="After", showlegend=False),
        row=1, col=2
    )

    # Add correlation lines
    if corr_boiler_drains > 0.3:
        x_range = [st.session_state.Y[:, 0].min(), st.session_state.Y[:, 0].max()]
        z = np.polyfit(st.session_state.Y[:, 0], st.session_state.Y[:, 1], 1)
        p = np.poly1d(z)
        fig.add_trace(
            go.Scatter(x=x_range, y=p(x_range),
                       mode='lines', line=dict(color='red', dash='dash'),
                       showlegend=False),
            row=1, col=2
        )

    fig.update_xaxes(title_text=f"{products[0]} Cost ($)")
    fig.update_yaxes(title_text=f"{products[1]} Cost ($)", row=1, col=1)
    fig.update_layout(height=400, title_text="The Power of the Shuffle")

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class='success-box'>
    <h3>🎉 What Did We Achieve?</h3>

    - ✅ Each product keeps its original cost distribution
    - ✅ But now costs are correlated as designed!
    - ✅ Month with high ranks get high costs across correlated products
    - ✅ The marginal distributions are PERFECTLY preserved
    </div>
    """, unsafe_allow_html=True)

# Tab 10: Final Results
with tabs[10]:
    st.header("Step 10: Verify Final Results 🎊")

    st.markdown("""
    <div class='info-box'>
    <b>🔍 Quality Check Time!</b><br>
    Let's verify that our shuffle worked correctly. We should have:
    1. The same cost distributions as we started with (marginals preserved)
    2. The correlation pattern we designed (dependencies achieved)
    </div>
    """, unsafe_allow_html=True)

    # Calculate final correlations
    df_Y = pd.DataFrame(st.session_state.Y, columns=products)
    corr_after = df_Y.corr(method='spearman')

    # Main results
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.subheader("🎯 Target")
        fig = go.Figure(data=go.Heatmap(
            z=st.session_state.R_used,
            x=products, y=products,
            colorscale='RdBu', zmid=0,
            text=np.round(st.session_state.R_used, 2),
            texttemplate='%{text}',
            textfont={"size": 12}
        ))
        fig.update_layout(height=250, title="What We Wanted")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("✅ Achieved")
        fig = go.Figure(data=go.Heatmap(
            z=corr_after.values,
            x=products, y=products,
            colorscale='RdBu', zmid=0,
            text=np.round(corr_after.values, 2),
            texttemplate='%{text}',
            textfont={"size": 12}
        ))
        fig.update_layout(height=250, title="What We Got")
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        st.subheader("📊 Difference")
        diff = corr_after.values - st.session_state.R_used
        fig = go.Figure(data=go.Heatmap(
            z=diff,
            x=products, y=products,
            colorscale='RdBu', zmid=0,
            text=np.round(diff, 3),
            texttemplate='%{text}',
            textfont={"size": 12}
        ))
        fig.update_layout(height=250, title="How Close?")
        st.plotly_chart(fig, use_container_width=True)

    # Marginal preservation check
    st.subheader("📈 Marginal Distributions Preserved?")

    fig = make_subplots(rows=1, cols=3, subplot_titles=products)

    for i, product in enumerate(products):
        # Original
        fig.add_trace(
            go.Histogram(x=st.session_state.X[:, i], name="Original",
                         marker_color='gray', opacity=0.5, nbinsx=20),
            row=1, col=i + 1
        )
        # After shuffle
        fig.add_trace(
            go.Histogram(x=st.session_state.Y[:, i], name="After Shuffle",
                         marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'][i],
                         opacity=0.7, nbinsx=20),
            row=1, col=i + 1
        )

    fig.update_layout(height=300, showlegend=True,
                      title_text="Distributions Unchanged - Success!")
    fig.update_xaxes(title_text="Cost ($)")
    fig.update_yaxes(title_text="Count", row=1, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # Co-exceedance analysis
    st.subheader("🎯 Real-World Impact: Co-Exceedance Rates")

    col1, col2 = st.columns([1, 1])

    with col1:
        # Calculate co-exceedance
        threshold = 0.9  # 90th percentile

        results = []
        for i in range(3):
            for j in range(i + 1, 3):
                q_i = np.quantile(st.session_state.Y[:, i], threshold)
                q_j = np.quantile(st.session_state.Y[:, j], threshold)

                # Before
                exceed_before = np.mean((st.session_state.X[:, i] > q_i) &
                                        (st.session_state.X[:, j] > q_j))

                # After
                exceed_after = np.mean((st.session_state.Y[:, i] > q_i) &
                                       (st.session_state.Y[:, j] > q_j))

                results.append({
                    'Pair': f"{products[i]} & {products[j]}",
                    'Before': f"{exceed_before:.1%}",
                    'After': f"{exceed_after:.1%}",
                    'Change': f"{(exceed_after - exceed_before):.1%}"
                })

        df_results = pd.DataFrame(results)
        st.markdown("**Probability Both Products Exceed 90th Percentile**")
        st.dataframe(df_results, use_container_width=True)

        st.markdown("""
        <div class='highlight-box'>
        <b>💰 Why This Matters:</b><br>
        In risk management, we care about multiple bad things happening together. 
        The correlation increases (or decreases) the chance of joint extremes!
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Summary statistics
        st.markdown("**📊 Summary Statistics**")

        summary_before = pd.DataFrame({
            'Mean': st.session_state.X.mean(axis=0),
            'Std Dev': st.session_state.X.std(axis=0),
            'Min': st.session_state.X.min(axis=0),
            'Max': st.session_state.X.max(axis=0)
        }, index=products).round(0)

        summary_after = pd.DataFrame({
            'Mean': st.session_state.Y.mean(axis=0),
            'Std Dev': st.session_state.Y.std(axis=0),
            'Min': st.session_state.Y.min(axis=0),
            'Max': st.session_state.Y.max(axis=0)
        }, index=products).round(0)

        st.markdown("*Before Shuffle:*")
        st.dataframe(summary_before.T, use_container_width=True)

        st.markdown("*After Shuffle:*")
        st.dataframe(summary_after.T, use_container_width=True)

        st.success("✅ Statistics are identical! Marginals perfectly preserved!")

# Tab 11: Playground
with tabs[11]:
    st.header("🎮 Interactive Playground")

    st.markdown("""
    <div class='info-box'>
    <b>🧪 Experiment Time!</b><br>
    Try different correlation values and sample sizes to see how they affect the results. 
    Use the sidebar controls and click "Generate New Data" to explore!
    </div>
    """, unsafe_allow_html=True)

    # Quick scenarios
    st.subheader("💡 Try These Scenarios")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class='highlight-box'>
        <b>🏢 Various Month Scenario</b><br>
        Set all correlations to +0.8<br>
        (Everything fails together)
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='highlight-box'>
        <b>⚖️ Trade-off Scenario</b><br>
        Set Boiler-Drains: +0.9<br>
        Set others to -0.5<br>
        (Budget trade-offs)
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class='highlight-box'>
        <b>🎲 Independent Scenario</b><br>
        Set all correlations to 0<br>
        (No relationships)
        </div>
        """, unsafe_allow_html=True)

    # Interactive comparison
    st.subheader("📊 Live Comparison")

    col1, col2 = st.columns([1, 1])

    with col1:
        # Scatter plot
        fig = go.Figure()

        # Add before
        fig.add_trace(go.Scatter(
            x=st.session_state.X[:, 0],
            y=st.session_state.X[:, 1],
            mode='markers',
            name='Independent',
            marker=dict(color='gray', size=6, opacity=0.5)
        ))

        # Add after
        fig.add_trace(go.Scatter(
            x=st.session_state.Y[:, 0],
            y=st.session_state.Y[:, 1],
            mode='markers',
            name='Correlated',
            marker=dict(color='purple', size=6)
        ))

        fig.update_layout(
            title=f"Effect of {corr_boiler_drains:.1f} Correlation",
            xaxis_title=f"{products[0]} Cost ($)",
            yaxis_title=f"{products[1]} Cost ($)",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # 3D plot if desired
        fig = go.Figure()

        fig.add_trace(go.Scatter3d(
            x=st.session_state.Y[:, 0],
            y=st.session_state.Y[:, 1],
            z=st.session_state.Y[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=st.session_state.Y[:, 0] + st.session_state.Y[:, 1] + st.session_state.Y[:, 2],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Total<br>Cost")
            ),
            text=[f"B{i + 1}" for i in range(len(st.session_state.Y))],
            hovertemplate="<b>%{text}</b><br>Boiler: $%{x:.0f}<br>Drains: $%{y:.0f}<br>Electrics: $%{z:.0f}<extra></extra>"
        ))

        fig.update_layout(
            title="3D View of Correlated Costs",
            scene=dict(
                xaxis_title=products[0],
                yaxis_title=products[1],
                zaxis_title=products[2]
            ),
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    # Insights
    st.markdown("""
    <div class='success-box'>
    <h3>🎓 Key Takeaways</h3>

    1. **Correlation ≠ Causation**: We're modeling relationships, not causes
    2. **Marginals Stay Fixed**: Individual patterns never change
    3. **Rank-Based**: Works by reordering, not transforming values
    4. **Flexible**: Can model any valid correlation pattern
    5. **Practical**: Essential for realistic risk modeling

    <br><b>🚀 Next Steps:</b>
    - Try extreme correlations (±0.9) to see strong effects
    - Increase sample size for smoother patterns
    - Think about real scenarios where this matters!
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
<p>Built by the Risk & Actuarial team for teaching the Iman-Conover method</p>
<p>Remember: This preserves marginal distributions while imposing correlation structure!</p>
</div>
""", unsafe_allow_html=True)