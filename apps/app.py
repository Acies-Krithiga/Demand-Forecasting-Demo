import streamlit as st
import warnings

warnings.filterwarnings('ignore')

# Import page functions from page_modules module
from page_modules import (
    page_home,
    page_data_upload,
    page_eda,
    page_segmentation_and_rules,
    page_outlier,
    page_forecasting
)


def _inject_global_styles():
    """Add global UI styling for a more polished look."""
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Source+Sans+3:wght@400;600;700&display=swap');

            :root {
                --bg-1: #f6f7fb;
                --bg-2: #eef2f8;
                --card: #ffffff;
                --ink: #0f172a;
                --muted: #5b6b82;
                --accent: #0ea5a4;
                --accent-2: #2563eb;
                --border: #e5e7eb;
                --shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
            }

            html, body, [class*="css"] {
                font-family: 'Source Sans 3', system-ui, -apple-system, Segoe UI, sans-serif;
                color: var(--ink);
                font-size: 0.94rem;
            }

            .stApp {
                background: radial-gradient(1200px 400px at 10% -10%, #dff6f4 0%, transparent 60%),
                            linear-gradient(180deg, var(--bg-1), var(--bg-2));
            }

            section.main > div {
                padding-top: 1.5rem;
                padding-bottom: 3rem;
            }

            .block-container {
                background: rgba(255, 255, 255, 0.75);
                border: 1px solid var(--border);
                border-radius: 20px;
                padding: 2.0rem 1.8rem 3rem 1.8rem;
                box-shadow: var(--shadow);
                overflow: visible;
            }

            h1 {
                font-family: 'Space Grotesk', 'Source Sans 3', sans-serif;
                letter-spacing: 0.2px;
                font-size: 1.6rem;
            }

            h2, h3 {
                font-family: 'Source Sans 3', system-ui, -apple-system, Segoe UI, sans-serif;
                letter-spacing: 0.1px;
            }

            h2 {
                font-size: 0.98rem;
            }

            h3 {
                font-size: 0.92rem;
            }

            hr {
                border-color: #e2e8f0;
            }

            .stSidebar {
                background: linear-gradient(180deg, #0f172a 0%, #0b1b33 100%);
            }

            .stSidebar [data-testid="stMarkdownContainer"] {
                color: #e5e7eb;
            }

            .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar h4 {
                color: #ffffff;
                font-family: 'Space Grotesk', 'Source Sans 3', sans-serif;
            }

            .stSidebar .stRadio > div {
                background: rgba(255, 255, 255, 0.06);
                border-radius: 12px;
                padding: 0.35rem 0.5rem;
            }

            .stSidebar .stRadio label {
                color: #e5e7eb;
                font-weight: 600;
            }

            .stMetric {
                background: var(--card);
                border: 1px solid var(--border);
                border-radius: 14px;
                padding: 14px 16px;
                box-shadow: var(--shadow);
            }

            .stDataFrame, .stTable {
                background: var(--card);
                border-radius: 12px;
                border: 1px solid var(--border);
                box-shadow: var(--shadow);
            }

            [data-testid="stPlotlyChart"] {
                border-radius: 12px;
                border: 1px solid var(--border);
                box-shadow: var(--shadow);
                background: var(--card);
                padding: 8px;
            }

            [data-testid="stExpander"] {
                border-radius: 14px;
                border: 1px solid var(--border);
                box-shadow: var(--shadow);
                background: var(--card);
            }

            [data-testid="stExpander"] summary {
                font-weight: 600;
            }

            [data-testid="stTabs"] button {
                font-weight: 600;
                letter-spacing: 0.2px;
            }

            [data-testid="stTabs"] [aria-selected="true"] {
                color: var(--ink) !important;
            }

            .stButton > button {
                background: linear-gradient(120deg, var(--accent), var(--accent-2));
                color: #ffffff;
                border: none;
                border-radius: 12px;
                padding: 0.5rem 1rem;
                font-weight: 600;
                box-shadow: var(--shadow);
            }

            .stButton > button:hover {
                filter: brightness(0.98);
            }

            input, textarea {
                border-radius: 10px !important;
                border: 1px solid var(--border) !important;
            }

            .stSelectbox, .stMultiSelect, .stTextInput, .stNumberInput, .stDateInput {
                background: transparent;
            }

            .page-hero {
                background: linear-gradient(120deg, #0ea5a4 0%, #2563eb 100%);
                border-radius: 18px;
                padding: 12px 18px;
                color: #ffffff;
                box-shadow: var(--shadow);
                margin: 0.6rem -1.8rem 18px -1.8rem;
                width: calc(100% + 3.6rem);
                display: block;
            }

            .page-hero .hero-title {
                font-family: 'Source Sans 3', system-ui, -apple-system, Segoe UI, sans-serif;
                font-size: 1.15rem;
                font-weight: 600;
                letter-spacing: 0.2px;
                color: #ffffff;
                margin-top: 6px;
            }

            .page-hero .hero-subtitle {
                margin-top: 6px;
                font-size: 0.85rem;
                color: rgba(255, 255, 255, 0.85);
            }

            .pill {
                display: inline-block;
                background: rgba(255, 255, 255, 0.2);
                border: 1px solid rgba(255, 255, 255, 0.25);
                color: #ffffff;
                padding: 2px 10px;
                border-radius: 999px;
                font-size: 0.7rem;
                margin-right: 6px;
            }

            .section-card {
                background: var(--card);
                border-radius: 14px;
                border: 1px solid var(--border);
                padding: 14px 18px;
                box-shadow: var(--shadow);
            }

            .stHomeCard {
                background-color: var(--card);
                padding: 22px;
                border-radius: 14px;
                border: 1px solid var(--border);
                border-left: 6px solid var(--accent);
                box-shadow: var(--shadow);
                height: 100%;
                margin-bottom: 20px;
                transition: transform 0.2s ease;
            }

            .stHomeCard:hover {
                transform: translateY(-3px);
            }

            .stHomeCard h3 {
                color: var(--accent);
                font-size: 1.05rem;
                margin-top: 0;
                margin-bottom: 10px;
            }

            .nav-step {
                background: #f8fbff;
                padding: 18px;
                border-radius: 12px;
                margin-top: 15px;
                border: 1px dashed #bfd3ff;
                box-shadow: 0 4px 12px rgba(15, 23, 42, 0.05);
            }

            .nav-step h3 {
                color: #10b981;
                font-weight: 600;
                font-size: 1rem;
                margin-bottom: 6px;
            }

            .metric-container {
                display: flex;
                justify-content: space-between;
                gap: 18px;
                margin: 20px 0 32px 0;
            }

            .metric-card {
                flex: 1;
                padding: 20px 22px;
                border-radius: 16px;
                color: var(--ink);
                background: linear-gradient(135deg, #e0f2fe, #e0e7ff);
                box-shadow: var(--shadow);
                text-align: center;
                transition: transform 0.2s ease, box-shadow 0.2s ease;
            }

            .metric-card:hover {
                transform: translateY(-3px);
                box-shadow: 0 10px 24px rgba(15, 23, 42, 0.12);
            }

            .metric-title {
                font-size: 0.88rem;
                font-weight: 600;
                opacity: 0.75;
                margin-bottom: 8px;
            }

            .metric-value {
                font-size: 1.4rem;
                font-weight: 700;
                color: #1e3a8a;
            }

            .eda-metric-card {
                background: var(--card);
                border-radius: 12px;
                border: 1px solid var(--border);
                box-shadow: var(--shadow);
                padding: 14px;
                text-align: center;
                height: 100%;
                display: flex;
                flex-direction: column;
                justify-content: center;
            }

            .eda-metric-title {
                font-size: 0.88rem;
                color: var(--muted);
                margin-bottom: 6px;
                font-weight: 600;
            }

            .eda-metric-value {
                font-size: 1.4rem;
                font-weight: 700;
                color: var(--ink);
            }
        </style>
        """,
        unsafe_allow_html=True
    )


def _render_page_header(title_text: str, subtitle_text: str):
    st.markdown(
        f"""
        <div class="page-hero">
            <div class="pill">Forecasting Studio</div>
            <div class="pill">Analytics Workspace</div>
            <div class="hero-title">{title_text}</div>
            <div class="hero-subtitle">{subtitle_text}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def main():
    """Main Streamlit app"""
    st.set_page_config(
        page_title='Demand Forecasting Dashboard',
        page_icon=':bar_chart:',
        layout='wide',
        initial_sidebar_state='expanded'
    )

    _inject_global_styles()

    # Sidebar navigation
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 0.5rem 0 1rem 0;">
                <h2 style="margin-bottom: 0.2rem;">Forecasting Studio</h2>
                <div style="color:#cbd5f5;font-size:0.85rem;">Plan, analyze, and act on demand signals</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        page = st.radio(
            'Navigation',
            ['Home', 'Data Upload', 'EDA', 'Segmentation', 'Outlier', 'Forecasting'],
            index=0
        )
        st.divider()
        st.markdown("""
        **Quick Guide**
        - Home: Overview and workflow
        - Data Upload: Load CSV inputs and run pipeline
        - EDA: Inspect data quality and trends
        - Segmentation: Review segment labels and rules
        - Outlier: Outlier detection (in progress)
        - Forecasting: Compare models and results
        """)

    page_titles = {
        'Home': 'Intelligent Demand Forecasting & SCM Platform',
        'Data Upload': 'Demand Data Ingestion',
        'EDA': 'Demand Forecasting EDA',
        'Segmentation': 'Segmentation',
        'Outlier': 'Outlier Detection & Correction',
        'Forecasting': 'Forecasting'
    }
    page_subtitle = "Unified workspace for uploads, segmentation, and forecasting outputs."

    _render_page_header(page_titles.get(page, "Demand Forecasting"), page_subtitle)

    # Render selected page
    if page == 'Home':
        page_home()
    elif page == 'Data Upload':
        page_data_upload()
    elif page == 'EDA':
        page_eda()
    elif page == 'Segmentation':
        page_segmentation_and_rules()
    elif page == 'Outlier':
        page_outlier()
    elif page == 'Forecasting':
        page_forecasting()


if __name__ == '__main__':
    main()
