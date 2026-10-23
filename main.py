"""
Streamlit Power Demo
--------------------
Run with:  streamlit run main.py

Shows how much you can build with very little code: interactive widgets,
live-updating charts (native Streamlit, Matplotlib, Plotly, Altair),
maps, caching, tabs, and a file-upload explorer.
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import altair as alt

# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------
st.set_page_config(page_title="Streamlit Power Demo", page_icon="📊", layout="wide")

st.title("📊 Streamlit Power Demo")
st.caption(
    "Every chart below is plain Python - no HTML, CSS or JavaScript. "
    "Change a widget in the sidebar and the whole app re-runs instantly."
)


# --------------------------------------------------------------------------
# Data helpers (cached so they don't recompute on every interaction)
# --------------------------------------------------------------------------
@st.cache_data
def make_sales_data(n_days: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    categories = ["Electronics", "Clothing", "Groceries", "Books"]
    rows = []
    for cat in categories:
        base = rng.integers(200, 600)
        trend = np.linspace(0, rng.integers(50, 250), n_days)
        noise = rng.normal(0, 40, n_days)
        season = 60 * np.sin(np.linspace(0, 6 * np.pi, n_days))
        sales = np.clip(base + trend + season + noise, 0, None)
        rows.append(pd.DataFrame({"date": dates, "category": cat, "sales": sales}))
    df = pd.concat(rows, ignore_index=True)
    df["month"] = df["date"].dt.to_period("M").astype(str)
    return df


@st.cache_data
def make_scatter_data(n_points: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    groups = rng.choice(["A", "B", "C"], size=n_points)
    centers = {"A": (0, 0), "B": (4, 4), "C": (-4, 3)}
    x = np.array([rng.normal(centers[g][0], 1.2) for g in groups])
    y = np.array([rng.normal(centers[g][1], 1.2) for g in groups])
    size = rng.uniform(5, 30, n_points)
    return pd.DataFrame({"x": x, "y": y, "group": groups, "size": size})


@st.cache_data
def make_map_data(n_points: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Points scattered around Rawalpindi / Islamabad
    return pd.DataFrame(
        {
            "lat": 33.60 + rng.normal(0, 0.05, n_points),
            "lon": 73.05 + rng.normal(0, 0.05, n_points),
        }
    )


# --------------------------------------------------------------------------
# Sidebar controls
# --------------------------------------------------------------------------
st.sidebar.header("⚙️ Controls")
n_days = st.sidebar.slider("Days of sales data", 30, 365, 180, step=15)
n_points = st.sidebar.slider("Scatter points", 50, 1000, 300, step=50)
seed = st.sidebar.number_input("Random seed", min_value=0, max_value=9999, value=42)
selected_cats = st.sidebar.multiselect(
    "Categories",
    ["Electronics", "Clothing", "Groceries", "Books"],
    default=["Electronics", "Clothing", "Groceries", "Books"],
)
st.sidebar.divider()
st.sidebar.info("Tip: tweak any control and watch every chart update.")

sales = make_sales_data(n_days, seed)
sales = sales[sales["category"].isin(selected_cats)]

if sales.empty:
    st.warning("Select at least one category in the sidebar.")
    st.stop()

# --------------------------------------------------------------------------
# KPI metrics
# --------------------------------------------------------------------------
total = sales["sales"].sum()
daily = sales.groupby("date")["sales"].sum()
best_cat = sales.groupby("category")["sales"].sum().idxmax()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total sales", f"${total:,.0f}")
c2.metric("Avg daily sales", f"${daily.mean():,.0f}")
c3.metric("Best category", best_cat)
c4.metric(
    "Last 7d vs prev 7d",
    f"${daily.tail(7).sum():,.0f}",
    f"{(daily.tail(7).sum() / max(daily.iloc[-14:-7].sum(), 1) - 1) * 100:+.1f}%",
)

st.divider()

# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------
tab_native, tab_mpl, tab_plotly, tab_altair, tab_map, tab_live, tab_upload = st.tabs(
    [
        "Native charts",
        "Matplotlib",
        "Plotly",
        "Altair",
        "Map",
        "Live simulation",
        "Upload your CSV",
    ]
)

# ---- 1. Native Streamlit charts -----------------------------------------
with tab_native:
    st.subheader("Built-in charts - one line each")
    pivot = sales.pivot_table(index="date", columns="category", values="sales")

    left, right = st.columns(2)
    with left:
        st.markdown("**Line chart**")
        st.line_chart(pivot)
    with right:
        st.markdown("**Area chart**")
        st.area_chart(pivot)

    st.markdown("**Bar chart (monthly totals)**")
    monthly = sales.pivot_table(
        index="month", columns="category", values="sales", aggfunc="sum"
    )
    st.bar_chart(monthly)

    with st.expander("Show the data table"):
        st.dataframe(sales.head(100), use_container_width=True)

# ---- 2. Matplotlib -------------------------------------------------------
with tab_mpl:
    st.subheader("Matplotlib works out of the box")
    left, right = st.columns(2)

    with left:
        fig, ax = plt.subplots(figsize=(6, 4))
        for cat, grp in sales.groupby("category"):
            ax.plot(grp["date"], grp["sales"], label=cat, alpha=0.8)
        ax.set_title("Daily sales by category")
        ax.set_xlabel("Date")
        ax.set_ylabel("Sales")
        ax.legend()
        fig.autofmt_xdate()
        st.pyplot(fig)

    with right:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        totals = sales.groupby("category")["sales"].sum()
        ax2.pie(totals, labels=totals.index, autopct="%1.1f%%", startangle=90)
        ax2.set_title("Share of total sales")
        st.pyplot(fig2)

    bins = st.slider("Histogram bins", 5, 60, 25)
    fig3, ax3 = plt.subplots(figsize=(10, 3.5))
    ax3.hist(sales["sales"], bins=bins, color="#4C78A8", edgecolor="white")
    ax3.set_title("Distribution of daily sales values")
    st.pyplot(fig3)

# ---- 3. Plotly -----------------------------------------------------------
with tab_plotly:
    st.subheader("Plotly - interactive (zoom, hover, pan)")
    scatter = make_scatter_data(n_points, seed)

    left, right = st.columns(2)
    with left:
        fig = px.scatter(
            scatter,
            x="x",
            y="y",
            color="group",
            size="size",
            title="Clustered scatter plot",
            opacity=0.75,
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.box(
            sales, x="category", y="sales", color="category", title="Sales spread"
        )
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        fig = px.line(
            sales.groupby(["date", "category"], as_index=False)["sales"].sum(),
            x="date",
            y="sales",
            color="category",
            title="Sales over time (use the range slider)",
        )
        fig.update_xaxes(rangeslider_visible=True)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        # 3D surface
        xs = np.linspace(-5, 5, 60)
        ys = np.linspace(-5, 5, 60)
        X, Y = np.meshgrid(xs, ys)
        Z = np.sin(np.sqrt(X**2 + Y**2)) * np.exp(-0.1 * np.sqrt(X**2 + Y**2))
        fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale="Viridis")])
        fig.update_layout(title="3D surface (drag to rotate)", margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    # Heatmap
    heat = sales.assign(weekday=sales["date"].dt.day_name()).pivot_table(
        index="category", columns="weekday", values="sales", aggfunc="mean"
    )
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heat = heat[[d for d in order if d in heat.columns]]
    fig = px.imshow(
        heat,
        text_auto=".0f",
        color_continuous_scale="Blues",
        title="Average sales by weekday",
        aspect="auto",
    )
    st.plotly_chart(fig, use_container_width=True)

# ---- 4. Altair -----------------------------------------------------------
with tab_altair:
    st.subheader("Altair - declarative, with linked selection")
    scatter = make_scatter_data(n_points, seed)

    brush = alt.selection_interval()
    points = (
        alt.Chart(scatter)
        .mark_circle(opacity=0.7)
        .encode(
            x="x:Q",
            y="y:Q",
            size="size:Q",
            color=alt.condition(brush, "group:N", alt.value("lightgray")),
            tooltip=["x", "y", "group", "size"],
        )
        .add_params(brush)
        .properties(height=300, title="Drag a box here...")
    )
    bars = (
        alt.Chart(scatter)
        .mark_bar()
        .encode(x="count():Q", y="group:N", color="group:N")
        .transform_filter(brush)
        .properties(height=300, title="...and this chart filters to your selection")
    )
    st.altair_chart(points | bars, use_container_width=True)

# ---- 5. Map --------------------------------------------------------------
with tab_map:
    st.subheader("Maps with a single call")
    n_map = st.slider("Number of points", 20, 500, 150)
    st.map(make_map_data(n_map, seed))

# ---- 6. Live simulation --------------------------------------------------
with tab_live:
    st.subheader("Random-walk simulation")
    st.write("Adjust the settings and press the button to see a fresh simulation.")

    col_a, col_b, col_c = st.columns(3)
    steps = col_a.slider("Steps", 100, 5000, 1000, step=100)
    walkers = col_b.slider("Walkers", 1, 20, 5)
    volatility = col_c.slider("Volatility", 0.1, 5.0, 1.0)

    if st.button("▶️ Run simulation", type="primary"):
        rng = np.random.default_rng()
        walks = rng.normal(0, volatility, size=(steps, walkers)).cumsum(axis=0)
        walk_df = pd.DataFrame(walks, columns=[f"Walker {i + 1}" for i in range(walkers)])
        st.line_chart(walk_df)
        st.success(
            f"Final spread between best and worst walker: "
            f"{walk_df.iloc[-1].max() - walk_df.iloc[-1].min():.2f}"
        )
    else:
        st.info("Press the button to run.")

# ---- 7. Upload your own CSV ----------------------------------------------
with tab_upload:
    st.subheader("Instant data explorer for any CSV")
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.write(f"**{df.shape[0]:,} rows x {df.shape[1]} columns**")
        st.dataframe(df.head(50), use_container_width=True)

        st.markdown("**Summary statistics**")
        st.dataframe(df.describe(include="all").T, use_container_width=True)

        numeric_cols = df.select_dtypes("number").columns.tolist()
        if len(numeric_cols) >= 1:
            col = st.selectbox("Pick a numeric column to plot", numeric_cols)
            st.plotly_chart(
                px.histogram(df, x=col, title=f"Distribution of {col}"),
                use_container_width=True,
            )
        if len(numeric_cols) >= 2:
            x_col = st.selectbox("X axis", numeric_cols, index=0)
            y_col = st.selectbox("Y axis", numeric_cols, index=1)
            st.plotly_chart(
                px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}"),
                use_container_width=True,
            )

        st.download_button(
            "Download summary as CSV",
            df.describe(include="all").T.to_csv().encode("utf-8"),
            file_name="summary.csv",
            mime="text/csv",
        )
    else:
        st.info("Upload any CSV to get instant tables and charts.")

st.divider()
st.caption("Built with Streamlit - run with `streamlit run main.py`")