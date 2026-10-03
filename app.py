"""
app.py — South Africa Crime Statistics Dashboard
Run with:  streamlit run app.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SA Crime Stats Dashboard",
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="expanded",
)

sns.set_style("whitegrid")
plt.rcParams.update({"figure.autolayout": True})

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH = "SouthAfricaCrimeStats_v2.csv"
MODEL_PATH = "crime_forecast_model.pkl"

YEAR_COLS = [
    "2005-2006", "2006-2007", "2007-2008", "2008-2009", "2009-2010",
    "2010-2011", "2011-2012", "2012-2013", "2013-2014", "2014-2015",
    "2015-2016",
]
FEATURES = ["YearStart", "CrimeCount", "Lag1", "Lag2"]
TARGET = "TargetNextYear"


# ---------------------------------------------------------------------------
# Data / model loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading crime data…")
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


@st.cache_resource(show_spinner="Loading model…")
def load_model(path: str):
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def build_long_df(df: pd.DataFrame) -> pd.DataFrame:
    df_long = df.melt(
        id_vars=["Province", "Station", "Category"],
        var_name="Year",
        value_name="CrimeCount",
    )
    df_long["YearStart"] = df_long["Year"].str[:4].astype(int)
    df_long = df_long.sort_values(
        ["Province", "Station", "Category", "YearStart"]
    ).reset_index(drop=True)
    return df_long


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🚔 SA Crime Dashboard")
st.sidebar.markdown("South Africa Recorded Crime Statistics (2005 – 2016)")

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "National Trends",
        "Provinces",
        "Categories",
        "Stations",
        "Predictions",
        "Model Performance",
    ],
)

st.sidebar.markdown("---")

if not os.path.exists(DATA_PATH):
    st.sidebar.error(
        f"⚠️ `{DATA_PATH}` not found.\n\n"
        "Place the CSV file in the same folder as `app.py`."
    )


# ---------------------------------------------------------------------------
# Load everything
# ---------------------------------------------------------------------------
df = load_data(DATA_PATH)
model = load_model(MODEL_PATH)

yearly_total = df[YEAR_COLS].sum()
df_long = build_long_df(df)


# ===========================================================================
# PAGE: Overview
# ===========================================================================
if page == "Overview":
    st.title("📊 South Africa Crime Statistics — Overview")
    st.caption(
        "An interactive dashboard exploring recorded crime across "
        "9 provinces, 1,143 police stations and 27 crime categories."
    )

    total_2005 = int(yearly_total["2005-2006"])
    total_2015 = int(yearly_total["2015-2016"])
    overall_change = (total_2015 - total_2005) / total_2005 * 100
    highest_year = yearly_total.idxmax()
    highest_val = int(yearly_total.max())
    lowest_year = yearly_total.idxmin()
    lowest_val = int(yearly_total.min())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Provinces", df["Province"].nunique())
    c3.metric("Stations", df["Station"].nunique())
    c4.metric("Categories", df["Category"].nunique())

    c1, c2, c3 = st.columns(3)
    c1.metric("Crime 2005/06", f"{total_2005:,}")
    c2.metric("Crime 2015/16", f"{total_2015:,}", f"{overall_change:+.2f}%")
    c3.metric(
        "Peak Year",
        highest_year,
        f"{highest_val:,}",
        delta_color="off",
    )

    st.markdown("---")
    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Summary Statistics (Yearly Totals)")
    st.dataframe(
        df[YEAR_COLS].describe().T.style.format("{:,.0f}"),
        use_container_width=True,
    )

    with st.expander("Show data types & missing values"):
        c1, c2 = st.columns(2)
        c1.write("**Data types**")
        c1.write(df.dtypes.astype(str))
        c2.write("**Missing values**")
        c2.write(df.isnull().sum())


# ===========================================================================
# PAGE: National Trends
# ===========================================================================
elif page == "National Trends":
    st.title("📈 National Crime Trends")

    col1, col2 = st.columns([3, 2])

    with col1:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(
            YEAR_COLS, yearly_total.values, marker="o",
            linewidth=2, color="#1f77b4",
        )
        ax.set_title("Total Recorded Crime by Financial Year")
        ax.set_xlabel("Financial Year")
        ax.set_ylabel("Total Recorded Crime")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("Yearly Summary")
        summary = pd.DataFrame({
            "Year": YEAR_COLS,
            "Total": yearly_total.values,
            "YoY Change": [np.nan] + list(yearly_total.diff().values[1:]),
            "% Change": [np.nan] + list((yearly_total.pct_change() * 100).values[1:]),
        })
        st.dataframe(
            summary.style.format({
                "Total": "{:,.0f}",
                "YoY Change": "{:+,.0f}",
                "% Change": "{:+.2f}%",
            }),
            use_container_width=True,
            height=420,
        )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        changes = yearly_total.diff()
        colors = ["#2ca02c" if v >= 0 else "#d62728" for v in changes.values[1:]]
        ax.bar(YEAR_COLS[1:], changes.values[1:], color=colors)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title("Year-over-Year Change")
        ax.set_xlabel("Financial Year")
        ax.set_ylabel("Change in Recorded Crime")
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        pct = yearly_total.pct_change() * 100
        colors = ["#2ca02c" if v >= 0 else "#d62728" for v in pct.values[1:]]
        ax.bar(YEAR_COLS[1:], pct.values[1:], color=colors)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title("Year-over-Year Percentage Change")
        ax.set_xlabel("Financial Year")
        ax.set_ylabel("% Change")
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")
    st.subheader("3-Year Rolling Average")

    rolling = yearly_total.rolling(window=3).mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(YEAR_COLS, yearly_total.values, marker="o", label="Actual")
    ax.plot(YEAR_COLS, rolling.values, marker="s", linewidth=2,
            label="3-Year Rolling Avg", color="orange")
    ax.set_xlabel("Financial Year")
    ax.set_ylabel("Recorded Crime")
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    st.pyplot(fig)
    plt.close(fig)


# ===========================================================================
# PAGE: Provinces
# ===========================================================================
elif page == "Provinces":
    st.title("🗺️ Crime by Province")

    province_yearly = df.groupby("Province")[YEAR_COLS].sum()

    fig, ax = plt.subplots(figsize=(11, 6))
    for province in province_yearly.index:
        ax.plot(
            YEAR_COLS, province_yearly.loc[province],
            marker="o", markersize=4, label=province,
        )
    ax.set_title("Recorded Crime Trends by Province")
    ax.set_xlabel("Financial Year")
    ax.set_ylabel("Total Recorded Crime")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")

    selected_year = st.selectbox(
        "Select financial year for ranking", YEAR_COLS, index=len(YEAR_COLS) - 1
    )

    province_rank = (
        df.groupby("Province")[selected_year].sum().sort_values(ascending=False)
    )
    province_pct = province_rank / province_rank.sum() * 100

    col1, col2 = st.columns([1, 1])

    with col1:
        fig, ax = plt.subplots(figsize=(7, 5))
        colors = sns.color_palette("viridis", len(province_rank))
        ax.barh(province_rank.index[::-1], province_rank.values[::-1], color=colors)
        ax.set_title(f"Total Crime by Province — {selected_year}")
        ax.set_xlabel("Recorded Crime")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("Ranking & Share")
        rank_df = pd.DataFrame({
            "Province": province_rank.index,
            "Crime": province_rank.values,
            "Share %": province_pct.values,
        })
        st.dataframe(
            rank_df.style.format({"Crime": "{:,.0f}", "Share %": "{:.2f}%"}),
            use_container_width=True,
            height=400,
        )

    st.markdown("---")
    st.subheader("Change: 2005/06 → 2015/16")
    change = (
        (province_yearly["2015-2016"] - province_yearly["2005-2006"])
        / province_yearly["2005-2006"] * 100
    ).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in change.values]
    ax.barh(change.index[::-1], change.values[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("% change")
    ax.set_title("Provincial Change in Recorded Crime (2005/06 → 2015/16)")
    st.pyplot(fig)
    plt.close(fig)


# ===========================================================================
# PAGE: Categories
# ===========================================================================
elif page == "Categories":
    st.title("🚨 Crime Categories")

    category_2015 = (
        df.groupby("Category")["2015-2016"].sum().sort_values(ascending=False)
    )

    top_n = st.slider("Number of top categories to display", 5, 27, 10, step=1)
    top_cats = category_2015.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_cats.index[::-1], top_cats.values[::-1],
            color=sns.color_palette("rocket", len(top_cats)))
    ax.set_xlabel("Recorded Crime")
    ax.set_title(f"Top {top_n} Crime Categories — 2015/2016")
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")
    st.subheader("Top 5 Categories Over Time")
    category_yearly = df.groupby("Category")[YEAR_COLS].sum()
    top5 = category_yearly.sum(axis=1).sort_values(ascending=False).head(5)

    fig, ax = plt.subplots(figsize=(11, 6))
    for cat in top5.index:
        ax.plot(YEAR_COLS, category_yearly.loc[cat],
                marker="o", linewidth=1.8, label=cat)
    ax.set_xlabel("Financial Year")
    ax.set_ylabel("Recorded Crime")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")
    st.subheader("Category Change (2005/06 → 2015/16)")
    cat_change = (
        (category_yearly["2015-2016"] - category_yearly["2005-2006"])
        / category_yearly["2005-2006"] * 100
    ).replace([np.inf, -np.inf], np.nan).dropna().sort_values(ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Largest increases**")
        st.dataframe(
            cat_change.head(10).to_frame("% change")
            .style.format("{:+.2f}%"),
            use_container_width=True,
        )
    with col2:
        st.markdown("**Largest decreases**")
        st.dataframe(
            cat_change.tail(10).sort_values().to_frame("% change")
            .style.format("{:+.2f}%"),
            use_container_width=True,
        )


# ===========================================================================
# PAGE: Stations
# ===========================================================================
elif page == "Stations":
    st.title("🏛️ Police Stations")

    selected_year = st.selectbox(
        "Financial year", YEAR_COLS, index=len(YEAR_COLS) - 1, key="station_year"
    )

    station_totals = (
        df.groupby("Station")[selected_year].sum().sort_values(ascending=False)
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 10 stations")
        st.dataframe(
            station_totals.head(10).to_frame("Recorded Crime")
            .style.format("{:,.0f}"),
            use_container_width=True,
        )
    with col2:
        st.subheader("Bottom 10 stations")
        st.dataframe(
            station_totals.tail(10).to_frame("Recorded Crime")
            .style.format("{:,.0f}"),
            use_container_width=True,
        )

    st.markdown("---")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(station_totals.head(15).index[::-1],
            station_totals.head(15).values[::-1],
            color=sns.color_palette("mako", 15))
    ax.set_title(f"Top 15 Stations — {selected_year}")
    ax.set_xlabel("Recorded Crime")
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")
    st.subheader("Historical Trend — Pick a Station")
    station_choice = st.selectbox("Station", sorted(df["Station"].unique()))
    station_df = df[df["Station"] == station_choice]
    station_yearly = station_df[YEAR_COLS].sum()

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(YEAR_COLS, station_yearly.values, marker="o", color="#2ca02c")
    ax.set_title(f"{station_choice} — Recorded Crime Over Time")
    ax.set_xlabel("Financial Year")
    ax.set_ylabel("Recorded Crime")
    ax.tick_params(axis="x", rotation=45)
    st.pyplot(fig)
    plt.close(fig)


# ===========================================================================
# PAGE: Predictions
# ===========================================================================
elif page == "Predictions":
    st.title("🔮 Crime Forecast")
    st.markdown(
        "Predict the **next financial year's recorded crime** for a given "
        "police station / category using the trained Random Forest model."
    )

    if model is None:
        st.error(
            f"Model file `{MODEL_PATH}` not found. "
            "Please place the trained `.pkl` file next to `app.py`."
        )
    else:
        st.success(f"Model loaded: `{MODEL_PATH}`")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            year_start = st.number_input(
                "Current Year Start", min_value=2005, max_value=2030,
                value=2014, step=1,
            )
        with col2:
            current = st.number_input(
                "Current Year Crime (CrimeCount)",
                min_value=0, value=1000, step=10,
            )
        with col3:
            lag1 = st.number_input(
                "Previous Year (Lag1)", min_value=0, value=950, step=10,
            )
        with col4:
            lag2 = st.number_input(
                "Two Years Ago (Lag2)", min_value=0, value=900, step=10,
            )

        if st.button("Predict", type="primary"):
            X = pd.DataFrame(
                [[year_start, current, lag1, lag2]], columns=FEATURES
            )
            pred = model.predict(X)[0]

            st.markdown("### Result")
            c1, c2 = st.columns([1, 2])
            with c1:
                st.metric(
                    f"Predicted crime for {year_start + 1}/{year_start + 2}",
                    f"{pred:,.0f}",
                    f"{(pred - current) / max(current, 1) * 100:+.2f}% vs current",
                )
            with c2:
                chart_data = pd.DataFrame({
                    "Year": ["Lag2", "Lag1", "Current", "Predicted"],
                    "Crime": [lag2, lag1, current, pred],
                })
                fig, ax = plt.subplots(figsize=(7, 3.5))
                colors = ["#aec7e8", "#aec7e8", "#1f77b4", "#ff7f0e"]
                ax.bar(chart_data["Year"], chart_data["Crime"], color=colors)
                ax.set_ylabel("Recorded Crime")
                st.pyplot(fig)
                plt.close(fig)


# ===========================================================================
# PAGE: Model Performance
# ===========================================================================
elif page == "Model Performance":
    st.title("🧪 Model Performance")

    st.markdown(
        """
        **Model:** `RandomForestRegressor` (tuned)  
        `max_depth=10`, `max_features='sqrt'`, `min_samples_leaf=2`,
        `n_estimators=100`, `random_state=42`

        **Features:** `YearStart`, `CrimeCount`, `Lag1`, `Lag2`  
        **Target:** Next year's `CrimeCount`
        """
    )

    perf = pd.DataFrame({
        "Dataset": ["Train", "Validation", "Test"],
        "MAE":  [6.041038, 13.168526, 12.670979],
        "RMSE": [34.229162, 38.231810, 35.385503],
        "R²":   [0.965792, 0.957300, 0.961078],
    })

    st.subheader("Reported Metrics")
    st.dataframe(
        perf.style.format({
            "MAE": "{:.3f}", "RMSE": "{:.3f}", "R²": "{:.4f}"
        }),
        use_container_width=True,
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        x = np.arange(len(perf))
        w = 0.35
        ax.bar(x - w/2, perf["MAE"], w, label="MAE", color="#1f77b4")
        ax.bar(x + w/2, perf["RMSE"], w, label="RMSE", color="#ff7f0e")
        ax.set_xticks(x)
        ax.set_xticklabels(perf["Dataset"])
        ax.set_ylabel("Error")
        ax.set_title("MAE & RMSE by Dataset")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(perf["Dataset"], perf["R²"], color=["#2ca02c", "#9467bd", "#d62728"])
        ax.set_ylim(0.9, 1.0)
        ax.set_ylabel("R²")
        ax.set_title("R² by Dataset")
        for i, v in enumerate(perf["R²"]):
            ax.text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=9)
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")
    st.subheader("Comparison with Baseline")
    st.dataframe(
        pd.DataFrame({
            "Model": ["Persistence Baseline", "Tuned Random Forest"],
            "MAE":   [12.501183, 12.670979],
            "RMSE":  [35.361596, 35.385503],
            "R²":    [0.961131, 0.961078],
        }).style.format({
            "MAE": "{:.3f}", "RMSE": "{:.3f}", "R²": "{:.4f}"
        }),
        use_container_width=True,
    )

    st.info(
        "The tuned Random Forest performs close to the simple persistence "
        "baseline, which reflects the very strong autocorrelation of "
        "recorded crime counts at the station × category level. However, "
        "the model provides lift on validation data and remains highly "
        "interpretable through its lag features."
    )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.caption(
    "Data source: South African Police Service (SAPS) — Recorded Crime "
    "Statistics. Dashboard built with Streamlit."
)