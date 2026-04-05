"""Home page for Demand Forecasting Dashboard"""
import streamlit as st
from .config import PROJECT_ROOT, OUTPUTS_PATH, INPUTS_PATH, SCRIPTS_PATH


def page_home():
    """Home page with introduction and navigation"""
    st.header("What is Demand Forecasting?")
    st.markdown("""
    Demand forecasting is the analytical process of estimating **future customer demand** for a product or service. 
    It forms the bedrock of **Supply Chain Management (SCM)**, enabling strategic decisions to be made across the 
    organization. By leveraging historical sales, market trends, and related factors, businesses can optimize 
    operations, ensuring profitability and customer satisfaction.
    """)

    st.subheader("Core Business Objectives")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="stHomeCard">
            <h3>Inventory Optimization</h3>
            <p>Predicting demand accurately helps you maintain the perfect stock level, preventing costly <b>stockouts</b> (lost sales) and equally expensive <b>overstocking</b> (carrying costs).</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="stHomeCard">
            <h3>Financial Planning</h3>
            <p>Reliable forecasts are essential for solid financial health. They inform budgeting, optimize pricing strategies, and ensure stable cash flow projections across quarters.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="stHomeCard">
            <h3>Production Scheduling</h3>
            <p>By knowing what will be needed, you can enable efficient manufacturing schedules, guaranteeing that material resources meet future market needs.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.header("Forecasting Workflow")
    st.info("""
    To begin your analysis, please proceed to the **Data Upload** page in the sidebar to upload the required files.
    """)

    nav_col1, nav_col2, nav_col3 = st.columns(3)

    with nav_col1:
        st.markdown("""
        <div class="nav-step">
            <h3>1. Data Ingestion</h3>
            <p>Navigate to the Data Upload page to securely <b>upload all required CSV files</b> (e.g., Sales, Price, Calendar, etc.) into the <code>/data/inputs</code> directory.</p>
        </div>
        """, unsafe_allow_html=True)

    with nav_col2:
        st.markdown("""
        <div class="nav-step">
            <h3>2. Exploratory Analysis</h3>
            <p>The EDA page allows you to review data quality, check for missing values, and visualize key trends and patterns in your historical dataset.</p>
        </div>
        """, unsafe_allow_html=True)

    with nav_col3:
        st.markdown("""
        <div class="nav-step">
            <h3>3. Forecasting & Results</h3>
            <p>On the Forecasting page, you can configure the model, generate the demand predictions, and analyze the resulting forecast accuracy.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Developed by Acies Global for Supply Chain Excellence.")

