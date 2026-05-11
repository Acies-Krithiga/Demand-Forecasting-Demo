"""Data upload page for Demand Forecasting Dashboard"""
import os
import sys
import subprocess

import pandas as pd
import streamlit as st

from .config import (
    INPUTS_PATH,
    SCRIPTS_PATH,
    SALES_FACT_DRIVE_URL,
    download_csv_from_url,
    is_valid_sales_fact_df,
)


def page_data_upload():
    """Data upload page"""
    expected_files = {
        "sales_fact": {"label": "Sales Fact File (.csv)", "mandatory": True, "description": "Mandatory base data for sales history."},
        "price_fact": {"label": "Price Fact File (.csv)", "mandatory": False, "description": "Historical pricing data."},
        "calendar_dim": {"label": "Calendar Dimension (.csv)", "mandatory": False, "description": "Date and calendar hierarchy details."},
        "external_data": {"label": "External Data (.csv)", "mandatory": False, "description": "Supplemental data like weather or macroeconomics."},
        "product_dim": {"label": "Product Dimension (.csv)", "mandatory": False, "description": "Product hierarchy and attributes."},
        "location_dim": {"label": "Location Dimension (.csv)", "mandatory": False, "description": "Store or regional information."},
        "promotion_fact": {"label": "Promotion Fact File (.csv)", "mandatory": False, "description": "Promotion details and flags."},
    }

    total_files = len(expected_files)
    mandatory_file_key = "sales_fact"
    INPUTS_PATH.mkdir(parents=True, exist_ok=True)

    def cache_sales_fact(file_path):
        """Cache the parsed sales file in session state for this Streamlit session."""
        try:
            df = pd.read_csv(file_path)
            if is_valid_sales_fact_df(df):
                st.session_state["sales_fact_df"] = df
            else:
                st.session_state.pop("sales_fact_df", None)
        except Exception:
            st.session_state.pop("sales_fact_df", None)

    def save_csv_from_url(csv_url: str, file_path):
        """Download a CSV from a URL and save it to disk."""
        df = download_csv_from_url(csv_url, file_path)
        if file_path.name == f"{mandatory_file_key}.csv":
            st.session_state["sales_fact_df"] = df

    def check_file_status():
        status_list = []
        ready_count = 0

        for key, props in expected_files.items():
            file_path = INPUTS_PATH / f"{key}.csv"
            exists = file_path.exists() and file_path.stat().st_size > 0

            if exists:
                status_icon = "✅ Uploaded"
                ready_count += 1
            elif props["mandatory"]:
                status_icon = "❌ Mandatory Missing"
            else:
                status_icon = "⚠️ Optional Missing"

            status_list.append({
                "File Name": f"{key}.csv",
                "Description": props["label"],
                "Mandatory": "Yes" if props["mandatory"] else "No",
                "Status": status_icon,
            })

        return pd.DataFrame(status_list), ready_count

    def get_sales_fact_df() -> pd.DataFrame | None:
        """Return a valid sales_fact dataframe from session state or disk."""
        cached_df = st.session_state.get("sales_fact_df")
        if cached_df is not None and is_valid_sales_fact_df(cached_df):
            return cached_df.copy()

        sales_fact_path = INPUTS_PATH / f"{mandatory_file_key}.csv"
        if sales_fact_path.exists() and sales_fact_path.stat().st_size > 0:
            try:
                df = pd.read_csv(sales_fact_path)
                if is_valid_sales_fact_df(df):
                    st.session_state["sales_fact_df"] = df
                    return df.copy()
            except Exception:
                pass
        return None

    st.header(f"📂 Data File Upload ({total_files} File Types)")

    file_options = {props["label"]: key for key, props in expected_files.items()}
    selected_label = st.selectbox(
        "1. Select the file type to upload:",
        options=list(file_options.keys()),
        key="file_select",
    )
    selected_key = file_options[selected_label]
    st.caption(f"**Target File:** `{selected_key}.csv` ({expected_files[selected_key]['description']})")

    uploaded_file = st.file_uploader(
        f"2. Upload '{selected_key}.csv' (CSV file only)",
        type=["csv"],
        key="file_uploader",
        label_visibility="visible",
    )

    if selected_key == mandatory_file_key:
        auto_sales_fact_path = INPUTS_PATH / f"{mandatory_file_key}.csv"
        if not auto_sales_fact_path.exists() or auto_sales_fact_path.stat().st_size == 0:
            try:
                save_csv_from_url(SALES_FACT_DRIVE_URL, auto_sales_fact_path)
                if auto_sales_fact_path.exists() and auto_sales_fact_path.stat().st_size > 0:
                    st.success("✅ Sales fact loaded automatically from Google Drive.")
                else:
                    st.error("❌ Automatic Google Drive import created an empty sales fact file.")
            except Exception as e:
                st.warning(f"Automatic Google Drive import failed: {e}")
        else:
            st.caption("Sales fact is already available locally. You can upload a new CSV to replace it.")

        if auto_sales_fact_path.exists() and auto_sales_fact_path.stat().st_size > 0:
            cache_sales_fact(auto_sales_fact_path)

        st.caption("You can still upload a CSV file here to replace the sales fact data.")

    if uploaded_file is not None:
        file_path = INPUTS_PATH / f"{selected_key}.csv"
        try:
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            if selected_key == mandatory_file_key:
                cache_sales_fact(file_path)

            if file_path.exists() and file_path.stat().st_size > 0:
                st.success(f"✅ **Success:** `{selected_key}.csv` saved to disk.")
            else:
                st.error(f"❌ Save Failed: File `{selected_key}.csv` is empty or could not be saved correctly.")
        except Exception as e:
            st.exception(f"Error saving file: {e}")

    sales_fact_df = get_sales_fact_df()
    if sales_fact_df is not None:
        st.markdown("---")
        st.subheader("Preview: `df_sales.head()`")
        st.dataframe(sales_fact_df.head(), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.header("📊 Data Ingestion Status")

    status_df, ready_count = check_file_status()
    mandatory_ready = status_df[status_df["File Name"] == f"{mandatory_file_key}.csv"]["Status"].iloc[0] == "✅ Uploaded"

    st.dataframe(status_df, hide_index=True, use_container_width=True)

    progress_percent = int((ready_count / total_files) * 100)
    st.subheader(f"Overall Progress: {ready_count}/{total_files} Files Ready")
    st.progress(progress_percent)

    if mandatory_ready:
        st.success("🎉 Mandatory file (`sales_fact.csv`) is uploaded! You can now run the pipeline.")
    elif ready_count > 0:
        st.warning("⚠️ The mandatory file (`sales_fact.csv`) is still missing. Please upload it to proceed.")
    else:
        st.warning("Please upload the mandatory file (`sales_fact.csv`) to start.")

    st.markdown("---")
    st.header("⚙️ Execute Forecasting Pipeline")

    main_script_path = SCRIPTS_PATH / "main.py"

    if mandatory_ready:
        if st.button("🚀 Run main.py"):
            with st.spinner("Running main.py... please wait"):
                try:
                    if not main_script_path.exists():
                        st.error(f"❌ `main.py` not found at: `{main_script_path}`")
                    else:
                        result = subprocess.run(
                            [sys.executable, str(main_script_path)],
                            capture_output=True,
                            text=True,
                            cwd=str(SCRIPTS_PATH),
                            env={
                                **os.environ,
                                "STAT_MAX_RULES": os.getenv("STAT_MAX_RULES", "25"),
                                "BASELINE_MAX_COMBINATIONS": os.getenv("BASELINE_MAX_COMBINATIONS", "50"),
                                "ML_MAX_COMBINATIONS": os.getenv("ML_MAX_COMBINATIONS", "10"),
                            },
                        )

                        if result.returncode == 0:
                            st.success("✅ `main.py` executed successfully!")
                            st.text_area("Console Output", result.stdout, height=200)
                        else:
                            st.error("❌ `main.py` encountered an error. Check the Error Log below.")
                            st.text_area("Error Log (stderr)", result.stderr, height=200)
                            if result.stdout:
                                st.text_area("Console Output (stdout)", result.stdout, height=200)
                except Exception as e:
                    st.exception(f"Error while running `main.py`: {e}")
    else:
        st.warning("⚠️ The mandatory file (`sales_fact.csv`) must be uploaded before running the pipeline.")

    st.markdown("---")
    st.caption(f"💾 Files are stored in: `{INPUTS_PATH}`")
