"""Shared configuration and utilities for all pages"""
from pathlib import Path
from io import BytesIO
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen
import pandas as pd
import streamlit as st


def get_project_root():
    """Get project root directory from the location of this module."""
    return Path(__file__).resolve().parents[2]


# Initialize paths
PROJECT_ROOT = get_project_root()
OUTPUTS_PATH = PROJECT_ROOT / "data" / "outputs"
INPUTS_PATH = PROJECT_ROOT / "data" / "inputs"
SCRIPTS_PATH = PROJECT_ROOT / "scripts"
SALES_FACT_DRIVE_URL = "https://drive.google.com/file/d/1JkkqVIrFQ1pw5WAs7GRJ5zpAPNZf_Z3Z/view?usp=sharing"
REQUIRED_SALES_FACT_COLUMNS = {"date", "store_id", "item_id", "cat_id", "units_sold"}


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


def download_csv_from_url(csv_url: str, file_path: Path) -> pd.DataFrame:
    """Download a CSV from a URL and save it to disk."""
    direct_url = get_google_drive_direct_url(csv_url)
    request = Request(direct_url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request) as response:
        content = response.read()

    if not content:
        raise ValueError("The link returned no content.")

    df = pd.read_csv(BytesIO(content))
    missing = REQUIRED_SALES_FACT_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Downloaded CSV is missing required columns: {sorted(missing)}. "
            f"Found columns: {list(df.columns)[:10]}"
        )
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False)
    return df


def is_valid_sales_fact_df(df: pd.DataFrame) -> bool:
    """Return True when the dataframe looks like the expected sales_fact.csv."""
    if df is None or df.empty:
        return False
    return REQUIRED_SALES_FACT_COLUMNS.issubset(set(df.columns))


def load_csv_file(file_path: Path) -> pd.DataFrame:
    """Load a CSV file and return as DataFrame"""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading {file_path.name}: {e}")
        return pd.DataFrame()


def get_default_id_column(df: pd.DataFrame) -> str | None:
    """Get default ID column from dataframe"""
    # Prefer common ID column names
    for col in ['store_id', 'Store', 'store', 'id', 'ID', 'item_id', 'Item', 'SKU']:
        if col in df.columns:
            return col
    # Fallback to first object column
    for col in df.columns:
        if df[col].dtype == 'object':
            return col
    return None


def filter_by_id(df: pd.DataFrame, id_col: str | None, selected_id):
    """Filter dataframe by selected ID value"""
    if df is None or df.empty:
        return df
    if not id_col or id_col not in df.columns:
        return df
    if selected_id in (None, 'All'):
        return df
    try:
        return df[df[id_col].astype(str) == str(selected_id)]
    except Exception:
        return df[df[id_col] == selected_id]


def render_filter_controls(df: pd.DataFrame, prefix: str = ""):
    """Render filter controls for ID column and value"""
    if df is None or df.empty:
        return None, 'All'
    
    # Use prefix to store separate filter states for each component
    id_col_key = f'{prefix}_id_col' if prefix else 'id_col'
    selected_id_key = f'{prefix}_selected_id' if prefix else 'selected_id'
    
    # Initialize ID column in session state
    if id_col_key not in st.session_state or st.session_state[id_col_key] not in df.columns:
        st.session_state[id_col_key] = get_default_id_column(df)
    
    # Get current ID column
    current_id_col = st.session_state.get(id_col_key, get_default_id_column(df))
    if current_id_col not in df.columns:
        current_id_col = get_default_id_column(df)
        st.session_state[id_col_key] = current_id_col
    
    # Container for filters
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            # ID column selector
            id_col = st.selectbox(
                'Filter by Column',
                options=df.columns.tolist(),
                index=df.columns.get_loc(current_id_col) if current_id_col in df.columns else 0,
                key=f'{prefix}_filter_col'
            )
            
            # Update session state with selected column
            if st.session_state.get(id_col_key) != id_col:
                st.session_state[id_col_key] = id_col
                # Reset selected value when column changes
                st.session_state[selected_id_key] = 'All'
        
        with col2:
            # ID value selector based on selected column
            id_values = ['All']
            try:
                unique_values = sorted(df[id_col].dropna().unique().tolist())
                id_values.extend([str(v) for v in unique_values])
            except Exception:
                pass
            
            # Get current selection, but reset if column changed
            current_selected = st.session_state.get(selected_id_key, 'All')
            # If current selection not in new list, reset to 'All'
            if current_selected not in id_values:
                current_selected = 'All'
                st.session_state[selected_id_key] = 'All'
            
            default_index = id_values.index(current_selected) if current_selected in id_values else 0
            
            selected_id = st.selectbox(
                'Select Value',
                options=id_values,
                index=default_index,
                key=f'{prefix}_filter_value'
            )
            st.session_state[selected_id_key] = selected_id
    
    return id_col, selected_id

