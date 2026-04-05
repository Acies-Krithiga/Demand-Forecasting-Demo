"""Outlier detection page for Demand Forecasting Dashboard"""
import streamlit as st
from .config import PROJECT_ROOT, OUTPUTS_PATH, INPUTS_PATH, SCRIPTS_PATH
import pandas as pd
import plotly.graph_objects as go

def page_outlier():

    # 1️⃣ Load Data
    file_path = (OUTPUTS_PATH / "units_sold_corrected.csv").resolve()
    try:
        df = pd.read_csv(file_path, encoding="utf-8-sig")
    except FileNotFoundError:
        st.error(f"❌ File not found at: {file_path}")
        st.stop()

    # Validate columns
    required_cols = ["date", "store_id", "item_id", "units_sold", "units_sold_corrected"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    # Clean types
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["units_sold"] = pd.to_numeric(df["units_sold"], errors="coerce")
    df["units_sold_corrected"] = pd.to_numeric(df["units_sold_corrected"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    # 2️⃣ Summary Metrics
    total_points = len(df)
    corrected_points = (df["units_sold"] != df["units_sold_corrected"]).sum()
    pct_corrected = 100 * corrected_points / total_points if total_points else 0
    mean_actual = df["units_sold"].mean()
    mean_corrected = df["units_sold_corrected"].mean()
    mean_shift = 100 * (mean_corrected - mean_actual) / mean_actual if mean_actual else 0



    st.markdown("### 🧭 Correction Summary Metrics")
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-card"><div class='metric-title'>Total Records</div><div class='metric-value'>{total_points:,}</div></div>
        <div class="metric-card"><div class='metric-title'>Corrected Values</div><div class='metric-value'>{corrected_points:,}</div></div>
        <div class="metric-card"><div class='metric-title'>% Data Corrected</div><div class='metric-value'>{pct_corrected:.2f}%</div></div>
        <div class="metric-card"><div class='metric-title'>Mean Shift (%)</div><div class='metric-value'>{mean_shift:.2f}</div></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # 3️⃣ Period Filter (below metrics)
    st.markdown("### 🔍 Period Filter")
    filter_mode = st.selectbox("Select View Type", ["Full Range", "Month", "Quarter", "Year", "Custom Range"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.to_period("Q")
    df_filtered = df.copy()

    if filter_mode == "Month":
        y, m = st.columns(2)
        with y: selected_year = st.selectbox("Year", sorted(df["year"].unique(), reverse=True))
        with m: selected_month = st.selectbox("Month", range(1, 13))
        df_filtered = df[(df["year"] == selected_year) & (df["month"] == selected_month)]
    elif filter_mode == "Quarter":
        y, q = st.columns(2)
        with y: selected_year = st.selectbox("Year", sorted(df["year"].unique(), reverse=True))
        with q:
            quarters = df.loc[df["year"] == selected_year, "quarter"].unique()
            selected_quarter = st.selectbox("Quarter", sorted(quarters))
        df_filtered = df[df["quarter"] == selected_quarter]
    elif filter_mode == "Year":
        selected_year = st.selectbox("Year", sorted(df["year"].unique(), reverse=True))
        df_filtered = df[df["year"] == selected_year]
    elif filter_mode == "Custom Range":
        min_d, max_d = df["date"].min(), df["date"].max()
        start_d, end_d = st.date_input("Select Range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
        df_filtered = df[(df["date"] >= pd.to_datetime(start_d)) & (df["date"] <= pd.to_datetime(end_d))]

    # 4️⃣ Chart 1 — Aggregated Actual vs Corrected
    st.markdown("## 1️⃣ Aggregated Actual vs Corrected Demand")
    agg_df = df_filtered.groupby("date", as_index=False)[["units_sold", "units_sold_corrected"]].sum()
    fig_agg = go.Figure()
    fig_agg.add_trace(go.Scatter(x=agg_df["date"], y=agg_df["units_sold"], mode="lines", name="Actual", line=dict(color="steelblue", width=2.5)))
    fig_agg.add_trace(go.Scatter(x=agg_df["date"], y=agg_df["units_sold_corrected"], mode="lines", name="Corrected", line=dict(color="mediumseagreen", width=2.5)))
    fig_agg.update_layout(title=f"Aggregated Actual vs Corrected Demand ({filter_mode})", xaxis_title="Date", yaxis_title="Units Sold",
                          template="plotly_white", legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig_agg, use_container_width=True)

    var_actual = df_filtered["units_sold"].var()
    var_corrected = df_filtered["units_sold_corrected"].var()
    reduction = 100 * (var_actual - var_corrected) / var_actual if var_actual else 0
    st.metric("Overall Variance Reduction (%)", f"{reduction:.2f}")

    # 5️⃣ Chart 2 — Store × Item Level
    st.markdown("---")
    st.markdown("## 2️⃣ Store × Item Level Comparison")
    corrected_groups = df_filtered.groupby(["store_id", "item_id"]).apply(lambda g: (g["units_sold"] != g["units_sold_corrected"]).any())
    corrected_groups = corrected_groups[corrected_groups].index.tolist()

    if not corrected_groups:
        st.warning("No store × item combinations with corrections found.")
        st.stop()

    selected_pair = st.selectbox("Select Store × Item", corrected_groups, format_func=lambda x: f"Store {x[0]} | Item {x[1]}")
    store, item = selected_pair
    f = df_filtered[(df_filtered["store_id"] == store) & (df_filtered["item_id"] == item)]

    fig_detail = go.Figure()
    fig_detail.add_trace(go.Scatter(x=f["date"], y=f["units_sold"], mode="lines+markers", name="Actual", line=dict(color="steelblue", width=2.5)))
    fig_detail.add_trace(go.Scatter(x=f["date"], y=f["units_sold_corrected"], mode="lines+markers", name="Corrected", line=dict(color="mediumseagreen", width=2.5)))
    fig_detail.update_layout(title=f"Store {store} | Item {item} — Actual vs Corrected", xaxis_title="Date", yaxis_title="Units Sold", template="plotly_white", legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig_detail, use_container_width=True)

    var_a, var_c = f["units_sold"].var(), f["units_sold_corrected"].var()
    red_local = 100 * (var_a - var_c) / var_a if var_a else 0
    st.metric(f"Variance Reduction (Store {store}, Item {item})", f"{red_local:.2f}")

    # 6️⃣ Data Table + Download
    st.markdown("---")
    st.markdown("### 📘 View or Download Processed Data")
    st.dataframe(df.head(20))
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Full Corrected Dataset", data=csv, file_name="units_sold_corrected_output.csv", mime="text/csv")

