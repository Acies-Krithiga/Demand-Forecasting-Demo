"""Data upload page for Demand Forecasting Dashboard"""
import os
import sys
from io import BytesIO
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen
import pandas as pd
import streamlit as st
import subprocess
from .config import PROJECT_ROOT, OUTPUTS_PATH, INPUTS_PATH, SCRIPTS_PATH


def page_data_upload():
    """Data upload page"""
    # File definitions
    expected_files = {
        "sales_fact": {"label": "Sales Fact File (.csv)", "mandatory": True, "description": "Mandatory base data for sales history."},
        "price_fact": {"label": "Price Fact File (.csv)", "mandatory": False, "description": "Historical pricing data."},
        "calendar_dim": {"label": "Calendar Dimension (.csv)", "mandatory": False, "description": "Date and calendar hierarchy details."},
        "external_data": {"label": "External Data (.csv)", "mandatory": False, "description": "Supplemental data like weather or macroeconomics."},
        "product_dim": {"label": "Product Dimension (.csv)", "mandatory": False, "description": "Product hierarchy and attributes."},
        "location_dim": {"label": "Location Dimension (.csv)", "mandatory": False, "description": "Store or regional information."},
        "promotion_fact": {"label": "Promotion Fact File (.csv)", "mandatory": False, "description": "Promotion details and flags."},
    }
    TOTAL_FILES = len(expected_files)
    MANDATORY_FILE_KEY = "sales_fact"
    SALES_FACT_DRIVE_URL = "https://drive.google.com/file/d/1JkkqVIrFQ1pw5WAs7GRJ5zpAPNZf_Z3Z/view?usp=sharing"

    # Create inputs directory if it doesn't exist
    INPUTS_PATH.mkdir(parents=True, exist_ok=True)

    def get_google_drive_direct_url(shared_url: str) -> str:
        """Convert a Google Drive share link into a direct download URL."""
        parsed = urlparse(shared_url.strip())
        query_params = parse_qs(parsed.query)

        if "id" in query_params:
            file_id = query_params["id"][0]
        else:
            file_id = None
            path_parts = [part for part in parsed.path.split("/") if part]
            if "d" in path_parts:
                d_index = path_parts.index("d")
                if d_index + 1 < len(path_parts):
                    file_id = path_parts[d_index + 1]

        if not file_id and "/file/d/" in parsed.path:
            try:
                file_id = parsed.path.split("/file/d/")[1].split("/")[0]
            except Exception:
                file_id = None

        if not file_id:
            return shared_url

        return f"https://drive.google.com/uc?export=download&id={file_id}"

    def save_csv_from_url(csv_url: str, file_path):
        """Download a CSV from a URL and save it to disk."""
        direct_url = get_google_drive_direct_url(csv_url)
        request = Request(direct_url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request) as response:
            content = response.read()

        if not content:
            raise ValueError("The link returned no content.")

        # Validate that the content is readable as CSV before saving.
        pd.read_csv(BytesIO(content))

        with open(file_path, "wb") as f:
            f.write(content)

    # Function to check file status
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
                "Status": status_icon
            })
        
        return pd.DataFrame(status_list), ready_count

    st.header(f"📂 Data File Upload ({TOTAL_FILES} File Types)")

    file_options = {props["label"]: key for key, props in expected_files.items()}
    selected_label = st.selectbox(
        "1. Select the file type to upload:",
        options=list(file_options.keys()),
        key="file_select"
    )
    selected_key = file_options[selected_label]
    st.caption(f"**Target File:** `{selected_key}.csv` ({expected_files[selected_key]['description']})")

    uploaded_file = st.file_uploader(
        f"2. Upload '{selected_key}.csv' (CSV file only)",
        type=["csv"],
        key="file_uploader",
        label_visibility="visible"
    )

    if selected_key == MANDATORY_FILE_KEY:
        auto_sales_fact_path = INPUTS_PATH / f"{MANDATORY_FILE_KEY}.csv"
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

    if selected_key == MANDATORY_FILE_KEY:
        st.caption("You can still upload a CSV file here to replace the sales fact data.")

    if uploaded_file is not None:
        file_path = INPUTS_PATH / f"{selected_key}.csv"
        try:
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            if file_path.exists() and file_path.stat().st_size > 0:
                st.success(f"✅ **Success:** `{selected_key}.csv` saved to disk.")
            else:
                st.error(f"❌ Save Failed: File `{selected_key}.csv` is empty or could not be saved correctly.")
        except Exception as e:
            st.exception(f"Error saving file: {e}")

    st.markdown("---")
    st.header("📊 Data Ingestion Status")

    status_df, ready_count = check_file_status()
    mandatory_ready = status_df[status_df["File Name"] == f"{MANDATORY_FILE_KEY}.csv"]["Status"].iloc[0] == "✅ Uploaded"

    st.dataframe(status_df, hide_index=True, use_container_width=True)

    progress_percent = int((ready_count / TOTAL_FILES) * 100)
    st.subheader(f"Overall Progress: {ready_count}/{TOTAL_FILES} Files Ready")
    st.progress(progress_percent)

    if mandatory_ready:
        st.success("🎉 Mandatory file (`sales_fact.csv`) is uploaded! You can now run the pipeline.")
    elif ready_count > 0:
        st.warning("⚠️ The mandatory file (`sales_fact.csv`) is still missing. Please upload it to proceed.")
    else:
        st.warning("Please upload the mandatory file (`sales_fact.csv`) to start.")

    st.markdown("---")
    st.header("⚙️ Execute Forecasting Pipeline")

    # Connect to scripts/main.py
    main_script_path = SCRIPTS_PATH / "main.py"

    if mandatory_ready:
        if st.button("🚀 Run main.py"):
            with st.spinner("Running main.py... please wait ⏳"):
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

