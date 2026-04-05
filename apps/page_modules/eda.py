"""EDA Dashboard page for Demand Forecasting Dashboard"""
import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
from .config import PROJECT_ROOT, OUTPUTS_PATH, INPUTS_PATH, SCRIPTS_PATH


def page_eda():
    """EDA Dashboard page"""
    # EDA outputs directory
    eda_output_dir = OUTPUTS_PATH / "EDA"
    
    # Required files for EDA
    required_files = {
        'df_sales': eda_output_dir / "df_sales.csv",
        'overall_metrics': eda_output_dir / "overall_metrics.json",
        'store_breakdown_metrics': eda_output_dir / "store_breakdown_metrics.csv",
        'category_daily_sales': eda_output_dir / "category_daily_sales.csv",
        'category_sales_treemap': eda_output_dir / "category_sales_treemap.csv",
        'sku_count_by_cat': eda_output_dir / "sku_count_by_cat.json"
    }
    
    # Check if all required files exist
    missing_files = [name for name, path in required_files.items() if not path.exists()]
    
    if missing_files:
        st.error("⚠️ **EDA outputs not available!**")
        st.info(f"""
        EDA outputs are missing. Please run the pipeline first:
        
        1. Go to the **Data Upload** page
        2. Upload the required CSV files (at minimum `sales_fact.csv`)
        3. Click the **🚀 Run main.py** button
        
        Missing files: {', '.join(missing_files)}
        """)
        st.stop()
    
    # Load EDA outputs
    try:
        import json
        
        # Load dataframes
        df_sales = pd.read_csv(required_files['df_sales'])
        df_sales['date'] = pd.to_datetime(df_sales['date'])
        store_breakdown_metrics = pd.read_csv(required_files['store_breakdown_metrics'])
        category_daily_sales = pd.read_csv(required_files['category_daily_sales'])
        category_daily_sales['date'] = pd.to_datetime(category_daily_sales['date'])
        category_sales_treemap = pd.read_csv(required_files['category_sales_treemap'])
        
        # Load JSON files
        with open(required_files['overall_metrics'], 'r') as f:
            overall_metrics = json.load(f)
        
        with open(required_files['sku_count_by_cat'], 'r') as f:
            sku_count_by_cat = json.load(f)
            
    except Exception as e:
        st.error(f"⚠️ **Error loading EDA outputs:** {e}")
        st.exception(e)
        st.info("Please run main.py to regenerate EDA outputs.")
        st.stop()

    # Display metrics
    st.header("1. Overall Network Summary")
    st.markdown(f"**Data Coverage Time Frame:** **{overall_metrics['Time Frame']}**")

    col_top_1, col_top_2, col_top_3 = st.columns(3)
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    def display_metric_card(title, value, format_spec='{:.0f}'):
        """Displays a custom Streamlit metric card with uniform formatting."""
        if isinstance(value, (int, float, np.number)) and not np.isnan(value):
            display_value = format_spec.format(value)
        elif isinstance(value, str):
            display_value = value
        else:
            display_value = "N/A"

        st.markdown(
            f"""
            <div class="eda-metric-card">
                <div class="eda-metric-title">{title}</div>
                <div class="eda-metric-value">{display_value}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_top_1:
        display_metric_card("Total Units Sold", overall_metrics['Total Units Sold'], '{:,.0f}')
    with col_top_2:
        display_metric_card("Avg Sales Per Day (Total)", overall_metrics['Average Sales Per Day (Total)'], '{:.1f}')
    with col_top_3:
        display_metric_card("Average Sales Per Store", overall_metrics['Average Sales Per Store'], '{:,.1f}')

    st.markdown("<br>", unsafe_allow_html=True)
    col_bottom_1, col_bottom_2, col_bottom_3 = st.columns(3)

    with col_bottom_1:
        display_metric_card("Number of Stores", overall_metrics['Number of Stores'])
    with col_bottom_2:
        display_metric_card("Total Product Categories", overall_metrics['Total Product Categories'])
    with col_bottom_3:
        display_metric_card("Total SKUs", overall_metrics['Total SKUs'], '{:,.0f}')

    st.subheader("Total Quantity Sold by Store")
    bar_data = store_breakdown_metrics[store_breakdown_metrics['store_id'] != 'Overall'].sort_values('total_quantity_sold', ascending=False)

    fig_bar = px.bar(
        bar_data,
        x='store_id',
        y='total_quantity_sold',
        title="Total Units Sold by Store (All Products)",
        labels={'store_id': 'Store ID', 'total_quantity_sold': 'Total Units Sold'},
        color='store_id'
    )
    fig_bar.update_layout(showlegend=False, height=350, margin=dict(t=50, b=10, l=10, r=10))
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.header("2. Store Performance & Breakdown")
    
    all_stores_options = ['Overall'] + sorted(df_sales['store_id'].unique().tolist())
    selected_store = st.selectbox(
        "Select Store for Detailed Metrics",
        options=all_stores_options,
        index=0
    )

    if selected_store != 'Overall':
        st.subheader(f"Metrics for **{selected_store}** (All Products)")
        store_data = store_breakdown_metrics[store_breakdown_metrics['store_id'] == selected_store].iloc[0]
        metric_cols = st.columns(4)
        
        with metric_cols[0]:
            display_metric_card("Avg Sales Per Day", store_data['avg_sales_per_day'], '{:.1f}')
        with metric_cols[1]:
            display_metric_card("Total Quantity Sold", store_data['total_quantity_sold'], '{:,.0f}')
        with metric_cols[2]:
            display_metric_card("Consistency (CV)", store_data['Sales Consistency (CV)'] * 100, '{:.1f}%')
        with metric_cols[3]:
            display_metric_card("Avg Transaction Size", store_data['Average Transaction Size (Units)'], '{:.2f}')
    else:
        st.info("Select a specific store above to see detailed store metrics.")

    st.markdown("---")
    deep_dive_store = selected_store
    st.header(f"3 Product Category Sales Treemap for **{deep_dive_store}**")

    if deep_dive_store == 'Overall':
        st.info("Please select a specific store in Section 2 to view the Category Sales Treemap.")
        treemap_data = pd.DataFrame()
    else:
        treemap_data = df_sales[df_sales['store_id'] == deep_dive_store].groupby('cat_id')['units_sold'].sum().reset_index()
        treemap_data.rename(columns={'units_sold': 'Total Units Sold'}, inplace=True)

    if not treemap_data.empty:
        fig_treemap = px.treemap(
            treemap_data,
            path=['cat_id'],
            values='Total Units Sold',
            title=f"Category Sales Breakdown in {deep_dive_store} (Units)",
            color='Total Units Sold',
            color_continuous_scale='Sunsetdark'
        )
        fig_treemap.update_layout(margin=dict(t=50, b=10, l=10, r=10))
        st.plotly_chart(fig_treemap, use_container_width=True)

    available_cats_in_store = sorted(df_sales[df_sales['store_id'] == deep_dive_store]['cat_id'].unique().tolist())
    if available_cats_in_store:
        selected_categories = st.selectbox(
            "Select Product Category for Deep Dive",
            options=available_cats_in_store,
            key='selected_sku_category'
        )

        if selected_categories:
            category_store_data = df_sales[
                (df_sales['store_id'] == deep_dive_store) & 
                (df_sales['cat_id'] == selected_categories)
            ]
            category_daily_agg = category_store_data.groupby('date')['units_sold'].sum().reset_index()

            total_units_sold = category_store_data['units_sold'].sum()
            category_skus = category_store_data['item_id'].nunique()

            store_min_date = category_store_data['date'].min()
            store_max_date = category_store_data['date'].max()
            all_store_dates = pd.date_range(start=store_min_date, end=store_max_date, freq='D')
            total_days_in_range = len(all_store_dates)
            
            dates_with_sales = category_daily_agg[category_daily_agg['units_sold'] > 0]['date'].nunique()
            category_zero_sales_days = total_days_in_range - dates_with_sales
            avg_units_sold_per_day = total_units_sold / total_days_in_range if total_days_in_range > 0 else 0

            metric_cols = st.columns(4)
            with metric_cols[0]:
                display_metric_card(f"No. of SKUs", category_skus)
            with metric_cols[1]:
                display_metric_card("Total Units Sold", total_units_sold, '{:,.0f}')
            with metric_cols[2]:
                display_metric_card("Avg Units Sold Per Day", avg_units_sold_per_day, '{:.2f}')
            with metric_cols[3]:
                display_metric_card("Zero Sales Days", category_zero_sales_days, '{:.0f}')

            filtered_daily_sales = category_daily_sales[
                category_daily_sales['cat_id'] == selected_categories
            ]
            
            weekly_sales = filtered_daily_sales.copy()
            weekly_sales['date_str'] = weekly_sales['date'].dt.strftime('%Y-%m-%d')
            weekly_sales_agg = weekly_sales.groupby(['date_str', 'cat_id'])['units_sold'].sum().reset_index()
            weekly_sales_agg['date_for_chart'] = pd.to_datetime(weekly_sales_agg['date_str'])
            
            fig_cat_trend = px.line(
                weekly_sales_agg,
                x='date_for_chart',
                y='units_sold',
                color='cat_id',
                title='Weekly Sales Trend Over Time (Network Aggregate)',
                labels={'date_for_chart': 'Date (Weekly Aggregate)', 'units_sold': 'Total Units Sold (Weekly)', 'cat_id': 'Category'},
                markers=True
            )
            fig_cat_trend.update_layout(height=400, margin=dict(t=50, b=10, l=10, r=10))
            st.plotly_chart(fig_cat_trend, use_container_width=True)

            st.subheader("Year Over Year Sales Seasonality by Month")
            monthly_sales = filtered_daily_sales.copy()
            monthly_sales['year'] = monthly_sales['date'].dt.year.astype(str)
            monthly_sales['month_name'] = monthly_sales['date'].dt.strftime('%b')

            monthly_sales_agg = monthly_sales.groupby(['year', 'month_name', 'cat_id'])['units_sold'].sum().reset_index()
            monthly_sales_agg['month_name'] = pd.Categorical(
                monthly_sales_agg['month_name'], 
                categories=month_order, 
                ordered=True
            )
            monthly_sales_agg = monthly_sales_agg.sort_values('month_name')

            fig_yoy = px.line(
                monthly_sales_agg,
                x='month_name',
                y='units_sold',
                color='year',
                line_group='year',
                facet_col='cat_id',
                facet_col_wrap=3,
                title='Year Over Year Sales Seasonality by Month',
                labels={'month_name': 'Month', 'units_sold': 'Total Units Sold', 'year': 'Year', 'cat_id': 'Category'},
                markers=True
            )
            fig_yoy.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
            fig_yoy.update_layout(height=400, margin=dict(t=50, b=10, l=10, r=10))
            st.plotly_chart(fig_yoy, use_container_width=True)

