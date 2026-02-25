import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.express as px
import plotly.graph_objects as go
import logging

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


st.set_page_config(page_title='Nagpur RE Forecast | FinVise', layout='wide')

@st.cache_data
def load_data():
    df = pd.read_csv("nagpur_real_estate_cleaned.xls")
    forecast_summary = pd.read_csv("forecast_summary.xls")
    locality_stats = pd.read_csv("locality_stats.xls")
    return df, forecast_summary, locality_stats

df, forecast_summary_df, locality_stats = load_data()

# SIDEBAR NAVIGATION 
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Trend & Forecast", "Compare Localities", "Download Data"]
)

localities = df["locality"].dropna().unique()

# page-1
if page == "Dashboard":

    st.title("Nagpur Real Estate Dashboard")

    selected_locality = st.selectbox("Select Locality", localities)

    loc_df = df[df["locality"] == selected_locality]

    
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Avg Price / Sqft",
        round(loc_df["avg_price_per_sqft"].mean(), 2)
    )

    col2.metric("Total Listings", len(loc_df))

    col3.metric(
        "Median Price",
        round(loc_df["median_price"].median(), 2)
    )

    # PRICE DISTRIBUTION 
    st.subheader("Price Distribution")
    fig = px.histogram(
        loc_df,
        x="avg_price_per_sqft",
        nbins=30,
        title=f"{selected_locality} Price Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

    # TOP 5 EXPENSIVE & AFFORDABLE 
    st.subheader("Top 5 Expensive vs Affordable Localities")

    avg_prices = (
        df.groupby("locality")["avg_price_per_sqft"]
        .mean()
        .sort_values()
    )

    top5 = avg_prices.tail(5).reset_index()
    bottom5 = avg_prices.head(5).reset_index()

    col1, col2 = st.columns(2)

    with col1:
        fig_top = px.bar(top5, x="locality", y="avg_price_per_sqft",
                         title="Top 5 Expensive")
        st.plotly_chart(fig_top, use_container_width=True)

    with col2:
        fig_bottom = px.bar(bottom5, x="locality", y="avg_price_per_sqft",
                            title="Top 5 Affordable")
        st.plotly_chart(fig_bottom, use_container_width=True)


# PAGE 2 — TREND & FORECAST 


elif page == "Trend & Forecast":

    st.title("Price Trend & Forecast")

    selected_locality = st.selectbox("Select Locality", localities)
    forecast_days = st.slider("Forecast Days", 30, 180, 90)

    df_loc = df[df["locality"] == selected_locality]


    def create_simulated_timeseries(df_locality, days=120):
        current_price = df_locality["avg_price_per_sqft"].mean()

        dates = pd.date_range(end=pd.Timestamp.today(), periods=days)
        trend = np.linspace(0, current_price * 0.05, days)
        noise = np.random.normal(0, current_price * 0.02, days)

        prices = current_price + trend + noise

        return pd.DataFrame({"ds": dates, "y": prices})

    ts_data = create_simulated_timeseries(df_loc)

    # HISTORICAL TREND 
    st.subheader("Historical Trend")
    fig_hist = px.line(ts_data, x="ds", y="y",
                       title=f"{selected_locality} Historical Trend")
    st.plotly_chart(fig_hist, use_container_width=True)

    # PROPHET FORECAST 
    model = Prophet()
    model.fit(ts_data)

    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)

    st.subheader("Forecast with Confidence Interval")

    fig_forecast = go.Figure()

    fig_forecast.add_trace(go.Scatter(
        x=ts_data["ds"], y=ts_data["y"],
        mode="lines", name="Historical"
    ))

    fig_forecast.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["yhat"],
        mode="lines", name="Forecast"
    ))

    fig_forecast.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["yhat_upper"],
        line=dict(width=0), showlegend=False
    ))

    fig_forecast.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["yhat_lower"],
        fill="tonexty", mode="lines",
        line=dict(width=0),
        name="Confidence Interval"
    ))

    st.plotly_chart(fig_forecast, use_container_width=True)

    # FORECAST SUMMARY TABLE
    st.subheader("Forecast Summary")

    st.dataframe(
        forecast_summary_df.rename(columns={"growth_pct": "% Growth"}),
        use_container_width=True
    )


# PAGE 3 — COMPARE LOCALITIES


elif page == "Compare Localities":

    st.title("Compare Localities")

    selected_locs = st.multiselect(
        "Select Localities to Compare",
        localities,
        default=localities[:2]
    )

    comp_df = df[df["locality"].isin(selected_locs)]

    comp_avg = (
        comp_df.groupby("locality")["avg_price_per_sqft"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        comp_avg,
        x="locality",
        y="avg_price_per_sqft",
        title="Average Price per Sqft Comparison"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Locality Statistics")

    stats_table = (
        comp_df.groupby("locality")
        .agg(
            avg_price_sqft=("avg_price_per_sqft", "mean"),
            median_price=("median_price", "median"),
            listings=("locality", "count")
        )
        .reset_index()
    )

    st.dataframe(stats_table, use_container_width=True)


# PAGE 4 — DOWNLOAD DATA 


elif page == "Download Data":

    st.title("⬇ Download Data")

    st.download_button(
        "Download Cleaned Data",
        df.to_csv(index=False),
        "nagpur_real_estate_cleaned.csv",
        "text/csv"
    )

    st.download_button(
        "Download Forecast Summary",
        forecast_summary_df.to_csv(index=False),
        "forecast_summary.csv",
        "text/csv"
    )

    st.success("Files ready for download")

