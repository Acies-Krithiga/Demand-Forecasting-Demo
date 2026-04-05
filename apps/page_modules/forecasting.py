
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from .config import PROJECT_ROOT, OUTPUTS_PATH


def page_forecasting():
    """Page for displaying Forecasting results (Baseline, Statistical, ML)"""
    def show_missing_outputs(tab_name, missing_files):
        st.warning(f"{tab_name} outputs are not available yet.")
        st.info(
            "Run the pipeline from Data Upload page.\n\n"
            f"Missing files: {', '.join(missing_files)}"
        )
    
    # Tabs for different forecasting types
    tab1, tab2, tab3, tab4 = st.tabs(["Baseline Forecasting", "Statistical Forecasting", "ML Forecasting", "Probability Forecasting"])

    with tab1:
        st.subheader("Baseline Forecasting")
        base_fcst_path = (OUTPUTS_PATH / "base_forecast_df.csv").resolve()
        base_mape_path = (OUTPUTS_PATH / "mape_baseline_df.csv").resolve()

        base_fcst_df = None
        base_mape_df = None

        missing_baseline = []
        if base_fcst_path.exists():
            try:
                base_fcst_df = pd.read_csv(base_fcst_path)
                # Parse date column if present
                if 'date' in base_fcst_df.columns:
                    try:
                        base_fcst_df['date'] = pd.to_datetime(base_fcst_df['date'])
                    except Exception:
                        pass
            except Exception as e:
                st.error(f" Error loading baseline forecast file `{base_fcst_path}`: {e}")
        else:
            missing_baseline.append("base_forecast_df.csv")

        if base_mape_path.exists():
            try:
                base_mape_df = pd.read_csv(base_mape_path)
            except Exception as e:
                st.error(f" Error loading baseline MAPE file `{base_mape_path}`: {e}")
        else:
            missing_baseline.append("mape_baseline_df.csv")

        if missing_baseline:
            show_missing_outputs("Baseline Forecasting", missing_baseline)

        if base_fcst_df is not None and not base_fcst_df.empty:
            st.divider()
            st.markdown("**Validation Forecast (Actual vs Predictions)**")

            # Identify model columns (everything except core columns)
            core_cols = {'date', 'actual', 'store_id', 'item_id'}
            model_cols = [c for c in base_fcst_df.columns if c not in core_cols]

            # Filters
            filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 2])
            with filter_col1:
                stores = sorted(base_fcst_df['store_id'].dropna().unique().tolist()) if 'store_id' in base_fcst_df.columns else []
                selected_store = st.selectbox('Store', options=stores, index=0 if stores else None, key='baseline_store')
            with filter_col2:
                if selected_store and 'item_id' in base_fcst_df.columns:
                    items = sorted(base_fcst_df.loc[base_fcst_df['store_id'] == selected_store, 'item_id'].dropna().unique().tolist())
                else:
                    items = []
                selected_item = st.selectbox('Item', options=items, index=0 if items else None, key='baseline_item')
            with filter_col3:
                if model_cols:
                    selected_models = st.multiselect('Models to display', options=model_cols, default=model_cols, key='baseline_models')
                else:
                    selected_models = []

            # Show available models for this store-item
            if selected_store and selected_item:
                st.caption("Available models in data for the selected store-item:")
                with st.container(border=True):
                    st.write(", ".join(model_cols) if model_cols else "No model columns found")

            # Filter dataframe for selection
            if selected_store and selected_item:
                sel_df = base_fcst_df.copy()
                sel_df = sel_df[(sel_df.get('store_id') == selected_store) & (sel_df.get('item_id') == selected_item)]

                # Display line chart if date is present (Validation Period Only)
                validation_df = sel_df[sel_df['actual'].notna()].copy() if 'actual' in sel_df.columns else sel_df.copy()
                
                if not validation_df.empty:
                    display_cols = ['actual'] + selected_models if selected_models else ['actual']
                    show_cols = [c for c in display_cols if c in validation_df.columns]
                    if 'date' in validation_df.columns and show_cols:
                        chart_df = validation_df[['date'] + show_cols].set_index('date').sort_index()
                        st.line_chart(chart_df, use_container_width=True)

                    # Data table
                    with st.expander(" Show validation data table", expanded=False):
                        st.dataframe(
                            validation_df[['date', 'actual'] + [c for c in selected_models if c in validation_df.columns]] if 'date' in validation_df.columns else validation_df[['actual'] + [c for c in selected_models if c in validation_df.columns]],
                            use_container_width=True,
                            hide_index=True,
                            height=350
                        )
                else:
                    st.warning(" No validation rows found for the selected store-item.")
            
            # NEW SECTION: Historical Actual vs Future Forecasts
            st.divider()
            st.markdown("**Historical Actual & Future Forecasts**")
            
            if selected_store and selected_item:
                # Load future forecast data from base_future_df.csv
                base_future_path = (OUTPUTS_PATH / "base_future_df.csv").resolve()
                base_future_df = None
                if base_future_path.exists():
                    try:
                        base_future_df = pd.read_csv(base_future_path)
                        if 'date' in base_future_df.columns:
                            base_future_df['date'] = pd.to_datetime(base_future_df['date'])
                    except Exception as e:
                        st.warning(f"Could not load future forecast data: {e}")
                
                # Load historical sales data
                sales_fact_path = (PROJECT_ROOT / "data" / "inputs" / "sales_fact.csv").resolve()
                sales_df = None
                if sales_fact_path.exists():
                    try:
                        sales_df = pd.read_csv(sales_fact_path)
                        if 'date' in sales_df.columns:
                            sales_df['date'] = pd.to_datetime(sales_df['date'])
                    except Exception as e:
                        st.warning(f"Could not load historical sales data: {e}")
                
                if base_future_df is not None and not base_future_df.empty:
                    # Filter future forecast data for selected store-item
                    forecast_df = base_future_df[
                        (base_future_df.get('store_id') == selected_store) & 
                        (base_future_df.get('item_id') == selected_item)
                    ].copy()
                    forecast_df = forecast_df.sort_values('date').reset_index(drop=True)
                    
                    # Get historical actuals from sales_fact.csv
                    hist_data_df = None
                    if sales_df is not None and 'units_sold' in sales_df.columns:
                        hist_sales = sales_df[
                            (sales_df['store_id'] == selected_store) & 
                            (sales_df['item_id'] == selected_item)
                        ].copy()
                        if not hist_sales.empty:
                            hist_sales = hist_sales.sort_values('date')
                            hist_data_df = hist_sales[['date', 'units_sold']].copy()
                    
                    # Create visualization: Historical Actual vs Future Forecasts
                    if (hist_data_df is not None and not hist_data_df.empty) and not forecast_df.empty:
                        fig = go.Figure()
                        
                        # Plot historical actuals
                        fig.add_trace(go.Scatter(
                            x=hist_data_df['date'],
                            y=hist_data_df['units_sold'],
                            mode='lines',
                            name='Historical Actual',
                            line=dict(color='#1f77b4', width=2),
                            hovertemplate='<b>Historical Actual</b><br>' +
                                          'Date: %{x}<br>' +
                                          'Units Sold: %{y}<br>' +
                                          '<extra></extra>'
                        ))
                        
                        # Plot future forecasts for each selected model (different colors)
                        color_palette = px.colors.qualitative.Set2
                        for idx, model in enumerate(selected_models):
                            if model in forecast_df.columns:
                                model_fcst = forecast_df[['date', model]].dropna()
                                if not model_fcst.empty:
                                    fig.add_trace(go.Scatter(
                                        x=model_fcst['date'],
                                        y=model_fcst[model],
                                        mode='lines',
                                        name=f'{model} (Future Forecast)',
                                        line=dict(color=color_palette[idx % len(color_palette)], width=2),
                                        hovertemplate=f'<b>{model} (Future Forecast)</b><br>' +
                                                      'Date: %{x}<br>' +
                                                      'Forecast: %{y}<br>' +
                                                      '<extra></extra>'
                                    ))
                        
                        # Add vertical line to separate history and forecast
                        if not forecast_df.empty:
                            cutoff_date = forecast_df['date'].min()
                            # Convert pandas Timestamp to Python datetime for plotly compatibility
                            if isinstance(cutoff_date, pd.Timestamp):
                                cutoff_date = cutoff_date.to_pydatetime()
                            # Use add_shape for the vertical line
                            fig.add_shape(
                                type="line",
                                x0=cutoff_date,
                                x1=cutoff_date,
                                y0=0,
                                y1=1,
                                yref="paper",
                                line=dict(color="gray", width=1, dash="dash")
                            )
                            # Add annotation text separately
                            fig.add_annotation(
                                x=cutoff_date,
                                y=1.02,
                                yref="paper",
                                text="Forecast Start",
                                showarrow=False,
                                font=dict(color="gray", size=10)
                            )
                        
                        # Update layout
                        fig.update_layout(
                            title=f'Historical Actual & Future Forecasts: {selected_store} - {selected_item}',
                            xaxis_title='Date',
                            yaxis_title='Units Sold',
                            height=500,
                            hovermode='x unified',
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            )
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    elif hist_data_df is None or hist_data_df.empty:
                        st.info("Historical sales data not available for this store-item combination.")
                    elif forecast_df.empty:
                        st.info("No future forecasts available for this store-item combination.")
                else:
                    st.warning(f"Future forecast file not found at: `{base_future_path}` or file is empty.")

            # MAPE section
            st.divider()
            st.markdown("**MAPE on Validation (by model)**")
            if base_mape_df is not None and not base_mape_df.empty and selected_store and selected_item:
                mape_row = base_mape_df[(base_mape_df.get('store_id') == selected_store) & (base_mape_df.get('item_id') == selected_item)]
                if not mape_row.empty:
                    # Try to map known baseline model names
                    ma_mape = None
                    sn_mape = None
                    if 'ma_mape' in mape_row.columns:
                        ma_mape = float(mape_row['ma_mape'].iloc[0])
                    if 'sn_mape' in mape_row.columns:
                        sn_mape = float(mape_row['sn_mape'].iloc[0])

                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("Moving Average MAPE", f"{ma_mape:.2f}%" if ma_mape is not None else "-")
                    with c2:
                        st.metric("Weighted Snaive MAPE", f"{sn_mape:.2f}%" if sn_mape is not None else "-")

                    with st.expander(" Show MAPE row", expanded=False):
                        st.dataframe(mape_row, use_container_width=True, hide_index=True)
                else:
                    st.warning(" No MAPE entry found for the selected store-item.")
            else:
                st.info("Select a store and item to view MAPE.")
        else:
            st.info("Baseline forecasting data not available.")

    # ----------------------------
    # Statistical Forecasting Tab
    # ----------------------------
    with tab2:
        st.subheader("Statistical Forecasting")
        
        # Load files
        stat_fcst_path = (OUTPUTS_PATH / "stat_forecasting.csv").resolve()
        future_stat_path = (OUTPUTS_PATH / "future_statforecast.csv").resolve()
        best_fit_path = (OUTPUTS_PATH / "best_fit_df.csv").resolve()
        
        stat_fcst_df = None
        future_stat_df = None
        best_fit_df = None
        
        missing_stat = []
        if stat_fcst_path.exists():
            try:
                stat_fcst_df = pd.read_csv(stat_fcst_path)
                if 'date' in stat_fcst_df.columns:
                    stat_fcst_df['date'] = pd.to_datetime(stat_fcst_df['date'])
            except Exception as e:
                st.error(f"Error loading statistical forecast file: {e}")
        else:
            missing_stat.append("stat_forecasting.csv")
        
        if future_stat_path.exists():
            try:
                future_stat_df = pd.read_csv(future_stat_path)
                if 'date' in future_stat_df.columns:
                    future_stat_df['date'] = pd.to_datetime(future_stat_df['date'])
            except Exception as e:
                st.error(f"Error loading future statistical forecast file: {e}")
        else:
            missing_stat.append("future_statforecast.csv")
        
        if best_fit_path.exists():
            try:
                best_fit_df = pd.read_csv(best_fit_path)
            except Exception as e:
                st.error(f"Error loading best fit file: {e}")
        else:
            missing_stat.append("best_fit_df.csv")

        if missing_stat:
            show_missing_outputs("Statistical Forecasting", missing_stat)
        
        # Get all unique stores and items from best_fit_df
        all_stores = []
        all_items = []
        if best_fit_df is not None and not best_fit_df.empty:
            if 'store_id' in best_fit_df.columns:
                all_stores = sorted(best_fit_df['store_id'].dropna().unique().tolist())
            if 'item_id' in best_fit_df.columns:
                all_items = sorted(best_fit_df['item_id'].dropna().unique().tolist())
        
        # Display Best Fit Data
        st.divider()
        st.markdown("**Best Fit Algorithm Results**")
        
        # Initialize filter variables (will be defined in filter section)
        selected_table_store = 'All'
        selected_table_item = 'All'
        
        if best_fit_df is not None and not best_fit_df.empty:
            # ========== OVERVIEW METRICS SECTION (Shown when "All" selected) ==========
            # This will show overview metrics before filters are applied
            st.subheader(" Statistical Forecasting Overview")
            
            # Overview metrics row
            overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)
            with overview_col1:
                st.metric("Total Store-Items", len(best_fit_df))
            with overview_col2:
                avg_mape = best_fit_df['best_mape'].mean() if 'best_mape' in best_fit_df.columns else 0
                st.metric("Avg MAPE", f"{avg_mape:.2f}%")
            with overview_col3:
                min_mape = best_fit_df['best_mape'].min() if 'best_mape' in best_fit_df.columns else 0
                st.metric("Best MAPE", f"{min_mape:.2f}%")
            with overview_col4:
                max_mape = best_fit_df['best_mape'].max() if 'best_mape' in best_fit_df.columns else 0
                st.metric("Worst MAPE", f"{max_mape:.2f}%")
            
            st.divider()
            
            # ========== HISTORICAL ACTUAL & FUTURE FORECASTS SECTION (MOVED TO TOP) ==========
            st.markdown("**Historical Actual & Future Forecasts**")
            
            # Filter section for Graph
            with st.expander("📈 Select for Chart", expanded=False):
                graph_filter_col1, graph_filter_col2 = st.columns(2)
                with graph_filter_col1:
                    graph_store_options = ['All'] + all_stores if all_stores else ['All']
                    selected_graph_store = st.selectbox(
                        'Store', 
                        options=graph_store_options, 
                        index=0,
                        key='stat_graph_store'
                    )
                with graph_filter_col2:
                    if selected_graph_store == 'All':
                        graph_item_options = ['All']
                    else:
                        filtered_items = sorted(best_fit_df.loc[best_fit_df['store_id'] == selected_graph_store, 'item_id'].dropna().unique().tolist()) if best_fit_df is not None and not best_fit_df.empty and 'item_id' in best_fit_df.columns else []
                        graph_item_options = ['All'] + filtered_items
                    
                    selected_graph_item = st.selectbox(
                        'Item', 
                        options=graph_item_options, 
                        index=0,
                        key='stat_graph_item'
                    )
            
            # Only show chart when a specific store-item is selected (not "All")
            if selected_graph_store != 'All' and selected_graph_item != 'All':
                # Load historical sales data
                sales_fact_path = (PROJECT_ROOT / "data" / "inputs" / "sales_fact.csv").resolve()
                sales_df = None
                if sales_fact_path.exists():
                    try:
                        sales_df = pd.read_csv(sales_fact_path)
                        if 'date' in sales_df.columns:
                            sales_df['date'] = pd.to_datetime(sales_df['date'])
                    except Exception as e:
                        st.warning(f"Could not load historical sales data: {e}")
                
                if future_stat_df is not None and not future_stat_df.empty:
                    # Get best fit algorithm for this store-item
                    best_algorithm = None
                    if best_fit_df is not None and not best_fit_df.empty:
                        best_fit_row = best_fit_df[
                            (best_fit_df['store_id'] == selected_graph_store) & 
                            (best_fit_df['item_id'] == selected_graph_item)
                        ]
                        if not best_fit_row.empty:
                            best_algorithm = best_fit_row['best_fit_algorithm'].iloc[0]
                    
                    # Filter future forecasts - use best fit algorithm if available, otherwise show all
                    if best_algorithm:
                        forecast_df = future_stat_df[
                            (future_stat_df['store_id'] == selected_graph_store) & 
                            (future_stat_df['item_id'] == selected_graph_item) &
                            (future_stat_df['algorithm'] == best_algorithm)
                        ].copy()
                    else:
                        # If no best fit, show all algorithms
                        forecast_df = future_stat_df[
                            (future_stat_df['store_id'] == selected_graph_store) & 
                            (future_stat_df['item_id'] == selected_graph_item)
                        ].copy()
                    
                    forecast_df = forecast_df.sort_values('date').reset_index(drop=True)
                    
                    # Get historical actuals from sales_fact.csv
                    hist_data_df = None
                    if sales_df is not None and 'units_sold' in sales_df.columns:
                        hist_sales = sales_df[
                            (sales_df['store_id'] == selected_graph_store) & 
                            (sales_df['item_id'] == selected_graph_item)
                        ].copy()
                        if not hist_sales.empty:
                            hist_sales = hist_sales.sort_values('date')
                            hist_data_df = hist_sales[['date', 'units_sold']].copy()
                    
                    # Create visualization: Historical Actual vs Future Forecasts
                    if (hist_data_df is not None and not hist_data_df.empty) and not forecast_df.empty:
                        fig = go.Figure()
                        
                        # Plot historical actuals
                        fig.add_trace(go.Scatter(
                            x=hist_data_df['date'],
                            y=hist_data_df['units_sold'],
                            mode='lines',
                            name='Historical Actual',
                            line=dict(color='#1f77b4', width=2),
                            hovertemplate='<b>Historical Actual</b><br>' +
                                          'Date: %{x}<br>' +
                                          'Units Sold: %{y}<br>' +
                                          '<extra></extra>'
                        ))
                        
                        # Plot future forecasts
                        if best_algorithm:
                            # Single best fit algorithm
                            forecast_subset = forecast_df[['date', 'forecast']].dropna()
                            if not forecast_subset.empty:
                                fig.add_trace(go.Scatter(
                                    x=forecast_subset['date'],
                                    y=forecast_subset['forecast'],
                                    mode='lines',
                                    name=f'{best_algorithm} (Future Forecast)',
                                    line=dict(color='#ff7f0e', width=2),
                                    hovertemplate=f'<b>{best_algorithm} (Future Forecast)</b><br>' +
                                                  'Date: %{x}<br>' +
                                                  'Forecast: %{y}<br>' +
                                                  '<extra></extra>'
                                ))
                        else:
                            # Multiple algorithms - show each with different color
                            color_palette = px.colors.qualitative.Set2
                            algorithms = forecast_df['algorithm'].unique()
                            for idx, algo in enumerate(algorithms):
                                algo_forecast = forecast_df[forecast_df['algorithm'] == algo][['date', 'forecast']].dropna()
                                if not algo_forecast.empty:
                                    fig.add_trace(go.Scatter(
                                        x=algo_forecast['date'],
                                        y=algo_forecast['forecast'],
                                        mode='lines',
                                        name=f'{algo} (Future Forecast)',
                                        line=dict(color=color_palette[idx % len(color_palette)], width=2),
                                        hovertemplate=f'<b>{algo} (Future Forecast)</b><br>' +
                                                      'Date: %{x}<br>' +
                                                      'Forecast: %{y}<br>' +
                                                      '<extra></extra>'
                                    ))
                        
                        # Add vertical line to separate history and forecast
                        if not forecast_df.empty:
                            cutoff_date = forecast_df['date'].min()
                            # Convert pandas Timestamp to Python datetime for plotly compatibility
                            if isinstance(cutoff_date, pd.Timestamp):
                                cutoff_date = cutoff_date.to_pydatetime()
                            # Use add_shape for the vertical line
                            fig.add_shape(
                                type="line",
                                x0=cutoff_date,
                                x1=cutoff_date,
                                y0=0,
                                y1=1,
                                yref="paper",
                                line=dict(color="gray", width=1, dash="dash")
                            )
                            # Add annotation text separately
                            fig.add_annotation(
                                x=cutoff_date,
                                y=1.02,
                                yref="paper",
                                text="Forecast Start",
                                showarrow=False,
                                font=dict(color="gray", size=10)
                            )
                        
                        # Update layout
                        fig.update_layout(
                            title=f'Historical Actual & Future Forecasts: {selected_graph_store} - {selected_graph_item}' + 
                                  (f' (Best Fit: {best_algorithm})' if best_algorithm else ''),
                            xaxis_title='Date',
                            yaxis_title='Units Sold',
                            height=500,
                            hovermode='x unified',
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            )
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    elif hist_data_df is None or hist_data_df.empty:
                        st.info("Historical sales data not available for this store-item combination.")
                    elif forecast_df.empty:
                        st.info("No future forecasts available for this store-item combination.")
                else:
                    st.warning(f"Future forecast file not found or is empty.")
            else:
                st.info("Please select a specific Store and Item to view the forecast chart.")
            
            st.divider()
            
            # ========== FILTER SECTION FOR TABLE (MOVED RIGHT ABOVE TABLE) ==========
            st.subheader(" Detailed Results Table")
            with st.expander("📊 Filter Data", expanded=True):
                table_filter_col1, table_filter_col2 = st.columns(2)
                with table_filter_col1:
                    table_store_options = ['All'] + all_stores if all_stores else ['All']
                    selected_table_store = st.selectbox(
                        'Store', 
                        options=table_store_options, 
                        index=0,
                        key='stat_table_store'
                    )
                with table_filter_col2:
                    if selected_table_store == 'All':
                        table_item_options = ['All'] + all_items if all_items else ['All']
                    else:
                        filtered_items = sorted(best_fit_df.loc[best_fit_df['store_id'] == selected_table_store, 'item_id'].dropna().unique().tolist()) if best_fit_df is not None and not best_fit_df.empty and 'item_id' in best_fit_df.columns else []
                        table_item_options = ['All'] + filtered_items
                    
                    selected_table_item = st.selectbox(
                        'Item', 
                        options=table_item_options, 
                        index=0,
                        key='stat_table_item'
                    )
            
            # Apply filters to display_df
            display_df = best_fit_df.copy()
            if selected_table_store != 'All':
                display_df = display_df[display_df['store_id'] == selected_table_store]
            if selected_table_item != 'All':
                display_df = display_df[display_df['item_id'] == selected_table_item]
            
            # Show metrics only when a specific store-item is selected
            if selected_table_store != 'All' and selected_table_item != 'All' and not display_df.empty:
                # Show metrics for selected store-item
                best_algorithm = display_df['best_fit_algorithm'].iloc[0]
                best_mape = display_df['best_mape'].iloc[0]
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Best Fit Algorithm", best_algorithm)
                with col2:
                    st.metric("Best MAPE", f"{best_mape:.2f}%")
                st.divider()
            
            # Display the filtered table
            st.dataframe(
                display_df,
                use_container_width=True,
                height=300,
                hide_index=True
            )
            
            # Download button
            st.download_button(
                label='📥 Download Best Fit CSV',
                data=best_fit_df.to_csv(index=False).encode('utf-8'),
                file_name='best_fit.csv',
                mime='text/csv',
                use_container_width=True
            )
            
            # ========== TOP PERFORMERS CHART (MOVED TO BOTTOM) ==========
            if (selected_table_store == 'All' or selected_table_item == 'All') and 'store_id' in best_fit_df.columns and 'best_mape' in best_fit_df.columns:
                st.divider()
                st.subheader(" Top 10 Store-Items by Best MAPE")
                top_performers = best_fit_df.nsmallest(10, 'best_mape')[['store_id', 'item_id', 'best_mape', 'best_fit_algorithm']].copy()
                top_performers['Store-Item'] = top_performers['store_id'].astype(str) + ' - ' + top_performers['item_id'].astype(str)
                
                fig_top = px.bar(
                    top_performers,
                    x='Store-Item',
                    y='best_mape',
                    title='Top 10 Best Performing Store-Items (Lowest MAPE)',
                    labels={'best_mape': 'MAPE (%)', 'Store-Item': 'Store - Item'},
                    color='best_mape',
                    color_continuous_scale='Greens_r'
                )
                fig_top.update_layout(
                    height=320,
                    showlegend=False,
                    margin=dict(l=80, r=20, t=50, b=80),
                    title=dict(font=dict(size=13), pad=dict(b=15)),
                    xaxis_title=dict(font=dict(size=11)),
                    yaxis_title=dict(font=dict(size=11)),
                    paper_bgcolor='white',
                    plot_bgcolor='white',
                    xaxis=dict(gridcolor='#E8E8E8', gridwidth=1, tickangle=-45),
                    yaxis=dict(gridcolor='#E8E8E8', gridwidth=1)
                )
                st.plotly_chart(fig_top, use_container_width=True)

    # ----------------------
    # ML Forecasting Tab
    # ----------------------
    with tab3:     # Load files
        best_fit_ml_path = (OUTPUTS_PATH / "best_fit_ml_df.csv").resolve()
        future_ml_path = (OUTPUTS_PATH / "ml_bestfit_predictions_future.csv").resolve()
        
        best_fit_ml_df = None
        future_ml_df = None
        
        missing_ml = []
        if best_fit_ml_path.exists():
            try:
                best_fit_ml_df = pd.read_csv(best_fit_ml_path)
            except Exception as e:
                st.error(f"Error loading best fit ML file: {e}")
        else:
            missing_ml.append("best_fit_ml_df.csv")
        
        if future_ml_path.exists():
            try:
                future_ml_df = pd.read_csv(future_ml_path)
                # Parse date column
                if 'date' in future_ml_df.columns:
                    future_ml_df['date'] = pd.to_datetime(future_ml_df['date'])
            except Exception as e:
                st.error(f"Error loading future ML forecast file: {e}")
        else:
            missing_ml.append("ml_bestfit_predictions_future.csv")
        
        # Get all unique stores and items from best_fit_ml_df
        all_stores_ml = []
        all_items_ml = []
        if best_fit_ml_df is not None and not best_fit_ml_df.empty:
            if 'store_id' in best_fit_ml_df.columns:
                all_stores_ml = sorted(best_fit_ml_df['store_id'].dropna().unique().tolist())
            if 'item_id' in best_fit_ml_df.columns:
                all_items_ml = sorted(best_fit_ml_df['item_id'].dropna().unique().tolist())
        
        # Load Feature Importance data
        feature_importance_path = (OUTPUTS_PATH / "feature_importance.csv").resolve()
        feature_importance_df = None
        
        if feature_importance_path.exists():
            try:
                feature_importance_df = pd.read_csv(feature_importance_path)
                # Sort by importance descending
                feature_importance_df = feature_importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
            except Exception as e:
                st.warning(f"Could not load feature importance file: {e}")
        else:
            missing_ml.append("feature_importance.csv")

        if missing_ml:
            show_missing_outputs("ML Forecasting", missing_ml)
        elif feature_importance_df is None:
            st.info("Feature importance file not found. Run feature selection to generate it.")
        
        # Initialize filter variables (will be defined in filter section)
        selected_ml_table_store = 'All'
        selected_ml_table_item = 'All'
        selected_ml_graph_store = 'All'
        selected_ml_graph_item = 'All'
        
        # ========== FEATURE IMPORTANCE SECTION ==========
        if feature_importance_df is not None and not feature_importance_df.empty:
            # Filter out zero importance features for display
            non_zero_features = feature_importance_df[feature_importance_df['importance'] > 0].copy()
            
            if not non_zero_features.empty:
                # Feature Importance Bar Chart
                st.subheader(" Feature Importance")
                
                # Show top 15 features or all if less than 15
                top_n = min(15, len(non_zero_features))
                top_features = non_zero_features.head(top_n)
                
                fig_feat = px.bar(
                    top_features,
                    x='importance',
                    y='feature',
                    orientation='h',
                    title=f'Top {top_n} Most Important Features',
                    labels={'importance': 'Importance Score', 'feature': 'Feature Name'},
                    color_discrete_sequence=['#2E86AB']
                )
                fig_feat.update_layout(
                    height=400,
                    showlegend=False,
                    margin=dict(l=150, r=20, t=50, b=50),
                    title=dict(font=dict(size=14), pad=dict(b=15)),
                    xaxis_title=dict(font=dict(size=12)),
                    yaxis_title=dict(font=dict(size=12)),
                    paper_bgcolor='white',
                    plot_bgcolor='white',
                    xaxis=dict(gridcolor='#E8E8E8', gridwidth=1),
                    yaxis=dict(gridcolor='#E8E8E8', gridwidth=1),
                    yaxis_categoryorder='total ascending'
                )
                st.plotly_chart(fig_feat, use_container_width=True)
                
                st.divider()
                
                # Feature Importance Table
                with st.expander(" View All Features", expanded=False):
                    # Format importance for better readability
                    display_df = feature_importance_df.copy()
                    display_df['importance_score'] = display_df['importance'].apply(lambda x: f"{x:.6f}")
                    # Calculate percentage of total importance
                    total_importance = feature_importance_df['importance'].sum()
                    display_df['importance_%'] = (display_df['importance'] / total_importance * 100).apply(lambda x: f"{x:.2f}%")
                    
                    st.dataframe(
                        display_df[['feature', 'importance_score', 'importance_%']],
                        use_container_width=True,
                        height=400,
                        hide_index=True
                    )
                    
                    # Download button for feature importance
                    st.download_button(
                        label='📥 Download Feature Importance CSV',
                        data=feature_importance_df.to_csv(index=False).encode('utf-8'),
                        file_name='feature_importance.csv',
                        mime='text/csv',
                        use_container_width=True
                    )
                
                st.divider()
        
        if best_fit_ml_df is not None and not best_fit_ml_df.empty:
            # ========== OVERVIEW METRICS SECTION ==========
            st.subheader(" ML Forecasting Overview")
            
            # Overview metrics row
            overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)
            with overview_col1:
                st.metric("Total Store-Items", len(best_fit_ml_df))
            with overview_col2:
                avg_mape = best_fit_ml_df['best_mape'].mean() if 'best_mape' in best_fit_ml_df.columns else 0
                st.metric("Avg MAPE", f"{avg_mape:.2f}%")
            with overview_col3:
                min_mape = best_fit_ml_df['best_mape'].min() if 'best_mape' in best_fit_ml_df.columns else 0
                st.metric("Best MAPE", f"{min_mape:.2f}%")
            with overview_col4:
                max_mape = best_fit_ml_df['best_mape'].max() if 'best_mape' in best_fit_ml_df.columns else 0
                st.metric("Worst MAPE", f"{max_mape:.2f}%")
            
            st.divider()
            
            # ========== HISTORICAL ACTUAL & FUTURE FORECASTS SECTION (MOVED TO TOP) ==========
            st.markdown("**Historical Actual & Future Forecasts**")
            
            # Filter section for Graph
            with st.expander(" Select for Chart", expanded=False):
                ml_graph_filter_col1, ml_graph_filter_col2 = st.columns(2)
                with ml_graph_filter_col1:
                    ml_graph_store_options = ['All'] + all_stores_ml if all_stores_ml else ['All']
                    selected_ml_graph_store = st.selectbox(
                        'Store', 
                        options=ml_graph_store_options, 
                        index=0,
                        key='ml_graph_store'
                    )
                with ml_graph_filter_col2:
                    if selected_ml_graph_store == 'All':
                        ml_graph_item_options = ['All']
                    else:
                        filtered_items_ml_graph = sorted(best_fit_ml_df.loc[best_fit_ml_df['store_id'] == selected_ml_graph_store, 'item_id'].dropna().unique().tolist()) if best_fit_ml_df is not None and not best_fit_ml_df.empty and 'item_id' in best_fit_ml_df.columns else []
                        ml_graph_item_options = ['All'] + filtered_items_ml_graph
                    
                    selected_ml_graph_item = st.selectbox(
                        'Item', 
                        options=ml_graph_item_options, 
                        index=0,
                        key='ml_graph_item'
                    )
            
            # Only show chart when a specific store-item is selected (not "All")
            if selected_ml_graph_store != 'All' and selected_ml_graph_item != 'All':
                # Load historical sales data from sales_fact.csv
                sales_fact_path = (PROJECT_ROOT / "data" / "inputs" / "sales_fact.csv").resolve()
                sales_df = None
                if sales_fact_path.exists():
                    try:
                        sales_df = pd.read_csv(sales_fact_path)
                        if 'date' in sales_df.columns:
                            sales_df['date'] = pd.to_datetime(sales_df['date'])
                    except Exception as e:
                        st.warning(f"Could not load historical sales data: {e}")
                
                if future_ml_df is not None and not future_ml_df.empty:
                    # Get best fit algorithm for this store-item
                    best_algorithm_ml = None
                    if best_fit_ml_df is not None and not best_fit_ml_df.empty:
                        best_fit_ml_row = best_fit_ml_df[
                            (best_fit_ml_df['store_id'] == selected_ml_graph_store) & 
                            (best_fit_ml_df['item_id'] == selected_ml_graph_item)
                        ]
                        if not best_fit_ml_row.empty:
                            best_algorithm_ml = best_fit_ml_row['best_fit_algorithm'].iloc[0]
                    
                    # Date column name is 'date'
                    date_col = 'date'
                    
                    # Filter future forecasts - use best fit algorithm if available, otherwise show all
                    if best_algorithm_ml:
                        forecast_ml_df = future_ml_df[
                            (future_ml_df['store_id'] == selected_ml_graph_store) & 
                            (future_ml_df['item_id'] == selected_ml_graph_item) &
                            (future_ml_df['algorithm'] == best_algorithm_ml)
                        ].copy()
                    else:
                        # If no best fit, show all algorithms
                        forecast_ml_df = future_ml_df[
                            (future_ml_df['store_id'] == selected_ml_graph_store) & 
                            (future_ml_df['item_id'] == selected_ml_graph_item)
                        ].copy()
                    
                    # Ensure date column exists and is datetime
                    if not forecast_ml_df.empty and date_col in forecast_ml_df.columns:
                        if not pd.api.types.is_datetime64_any_dtype(forecast_ml_df[date_col]):
                            forecast_ml_df[date_col] = pd.to_datetime(forecast_ml_df[date_col])
                        forecast_ml_df = forecast_ml_df.sort_values(date_col).reset_index(drop=True)
                    
                    # Get historical actuals from sales_fact.csv, excluding last 366 days
                    hist_data_df = None
                    if sales_df is not None and 'units_sold' in sales_df.columns:
                        hist_sales = sales_df[
                            (sales_df['store_id'] == selected_ml_graph_store) & 
                            (sales_df['item_id'] == selected_ml_graph_item)
                        ].copy()
                        if not hist_sales.empty:
                            hist_sales = hist_sales.sort_values('date')
                            # Exclude last 366 days to avoid overlap with forecast period
                            if len(hist_sales) > 366:
                                hist_sales = hist_sales.iloc[:-366]
                            hist_data_df = hist_sales[['date', 'units_sold']].copy()
                    
                    # Create visualization: Historical Actual vs Future Forecasts
                    if (hist_data_df is not None and not hist_data_df.empty) and not forecast_ml_df.empty:
                        fig = go.Figure()
                        
                        # Plot historical actuals
                        fig.add_trace(go.Scatter(
                            x=hist_data_df['date'],
                            y=hist_data_df['units_sold'],
                            mode='lines',
                            name='Historical Actual',
                            line=dict(color='#1f77b4', width=2),
                            hovertemplate='<b>Historical Actual</b><br>' +
                                          'Date: %{x}<br>' +
                                          'Units Sold: %{y}<br>' +
                                          '<extra></extra>'
                        ))
                        
                        # Plot future forecasts
                        if best_algorithm_ml:
                            # Single best fit algorithm
                            forecast_subset = forecast_ml_df[[date_col, 'prediction']].dropna()
                            if not forecast_subset.empty:
                                # Use lines+markers mode to ensure single points are visible
                                mode = 'lines+markers' if len(forecast_subset) == 1 else 'lines'
                                fig.add_trace(go.Scatter(
                                    x=forecast_subset[date_col],
                                    y=forecast_subset['prediction'],
                                    mode=mode,
                                    name=f'{best_algorithm_ml} (Future Forecast)',
                                    line=dict(color='#ff7f0e', width=2),
                                    marker=dict(size=8, color='#ff7f0e') if len(forecast_subset) == 1 else dict(size=6, color='#ff7f0e'),
                                    hovertemplate=f'<b>{best_algorithm_ml} (Future Forecast)</b><br>' +
                                                  'Date: %{x}<br>' +
                                                  'Forecast: %{y}<br>' +
                                                  '<extra></extra>'
                                ))
                        else:
                            # Multiple algorithms - show each with different color
                            color_palette = px.colors.qualitative.Set2
                            algorithms = forecast_ml_df['algorithm'].unique()
                            for idx, algo in enumerate(algorithms):
                                algo_forecast = forecast_ml_df[forecast_ml_df['algorithm'] == algo][[date_col, 'prediction']].dropna()
                                if not algo_forecast.empty:
                                    # Use lines+markers mode to ensure single points are visible
                                    mode = 'lines+markers' if len(algo_forecast) == 1 else 'lines'
                                    fig.add_trace(go.Scatter(
                                        x=algo_forecast[date_col],
                                        y=algo_forecast['prediction'],
                                        mode=mode,
                                        name=f'{algo} (Future Forecast)',
                                        line=dict(color=color_palette[idx % len(color_palette)], width=2),
                                        marker=dict(size=8, color=color_palette[idx % len(color_palette)]) if len(algo_forecast) == 1 else None,
                                        hovertemplate=f'<b>{algo} (Future Forecast)</b><br>' +
                                                      'Date: %{x}<br>' +
                                                      'Forecast: %{y}<br>' +
                                                      '<extra></extra>'
                                    ))
                        
                        # Add vertical line to separate history and forecast
                        if not forecast_ml_df.empty:
                            cutoff_date = forecast_ml_df[date_col].min()
                            # Convert pandas Timestamp to Python datetime for plotly compatibility
                            if isinstance(cutoff_date, pd.Timestamp):
                                cutoff_date = cutoff_date.to_pydatetime()
                            # Use add_shape for the vertical line
                            fig.add_shape(
                                type="line",
                                x0=cutoff_date,
                                x1=cutoff_date,
                                y0=0,
                                y1=1,
                                yref="paper",
                                line=dict(color="gray", width=1, dash="dash")
                            )
                            # Add annotation text separately
                            fig.add_annotation(
                                x=cutoff_date,
                                y=1.02,
                                yref="paper",
                                text="Forecast Start",
                                showarrow=False,
                                font=dict(color="gray", size=10)
                            )
                        
                        # Update layout with proper date range
                        all_dates = list(hist_data_df['date']) if hist_data_df is not None and not hist_data_df.empty else []
                        if not forecast_ml_df.empty and date_col in forecast_ml_df.columns:
                            all_dates.extend(forecast_ml_df[date_col].tolist())
                        
                        xaxis_range = None
                        if all_dates:
                            min_date = min(all_dates)
                            max_date = max(all_dates)
                            # Extend range slightly for better visibility
                            date_range = max_date - min_date
                            xaxis_range = [min_date - date_range * 0.02, max_date + date_range * 0.02]
                        
                        fig.update_layout(
                            title=f'Historical Actual & Future Forecasts: {selected_ml_graph_store} - {selected_ml_graph_item}' + 
                                  (f' (Best Fit: {best_algorithm_ml})' if best_algorithm_ml else ''),
                            xaxis_title='Date',
                            yaxis_title='Units Sold',
                            height=500,
                            hovermode='x unified',
                            xaxis=dict(range=xaxis_range) if xaxis_range else {},
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            )
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    elif hist_data_df is None or hist_data_df.empty:
                        st.info("Historical sales data not available for this store-item combination.")
                    elif forecast_ml_df.empty:
                        st.info("No future forecasts available for this store-item combination.")
                else:
                    st.warning(f"Future ML forecast file not found or is empty.")
            else:
                st.info("Please select a specific Store and Item to view the forecast chart.")
            
            st.divider()
            
            # ========== FILTER SECTION FOR TABLE (MOVED RIGHT ABOVE TABLE) ==========
            st.subheader(" Detailed Results Table")
            with st.expander(" Filter Data", expanded=True):
                ml_table_filter_col1, ml_table_filter_col2 = st.columns(2)
                with ml_table_filter_col1:
                    ml_table_store_options = ['All'] + all_stores_ml if all_stores_ml else ['All']
                    selected_ml_table_store = st.selectbox(
                        'Store', 
                        options=ml_table_store_options, 
                        index=0,
                        key='ml_table_store'
                    )
                with ml_table_filter_col2:
                    if selected_ml_table_store == 'All':
                        ml_table_item_options = ['All'] + all_items_ml if all_items_ml else ['All']
                    else:
                        filtered_items_ml = sorted(best_fit_ml_df.loc[best_fit_ml_df['store_id'] == selected_ml_table_store, 'item_id'].dropna().unique().tolist()) if best_fit_ml_df is not None and not best_fit_ml_df.empty and 'item_id' in best_fit_ml_df.columns else []
                        ml_table_item_options = ['All'] + filtered_items_ml
                    
                    selected_ml_table_item = st.selectbox(
                        'Item', 
                        options=ml_table_item_options, 
                        index=0,
                        key='ml_table_item'
                    )
            
            # Apply filters to display_ml_df
            display_ml_df = best_fit_ml_df.copy()
            if selected_ml_table_store != 'All':
                display_ml_df = display_ml_df[display_ml_df['store_id'] == selected_ml_table_store]
            if selected_ml_table_item != 'All':
                display_ml_df = display_ml_df[display_ml_df['item_id'] == selected_ml_table_item]
            
            # Show metrics only when a specific store-item is selected
            if selected_ml_table_store != 'All' and selected_ml_table_item != 'All' and not display_ml_df.empty:
                # Show metrics for selected store-item
                best_algorithm_ml = display_ml_df['best_fit_algorithm'].iloc[0]
                best_mape_ml = display_ml_df['best_mape'].iloc[0]
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Best Fit Algorithm", best_algorithm_ml)
                with col2:
                    st.metric("Best MAPE", f"{best_mape_ml:.2f}%")
                st.divider()
            
            # Display the best fit ML data table (reduced height)
            st.dataframe(
                display_ml_df,
                use_container_width=True,
                height=300,
                hide_index=True
            )
            
            # Download button
            st.download_button(
                label=' Download Best Fit ML CSV',
                data=best_fit_ml_df.to_csv(index=False).encode('utf-8'),
                file_name='best_fit_ml.csv',
                mime='text/csv',
                use_container_width=True
            )
            
            # ========== TOP PERFORMERS CHART (MOVED TO BOTTOM) ==========
            if (selected_ml_table_store == 'All' or selected_ml_table_item == 'All') and 'store_id' in best_fit_ml_df.columns and 'best_mape' in best_fit_ml_df.columns:
                st.divider()
                st.subheader(" Top 10 Store-Items by Best MAPE")
                top_performers = best_fit_ml_df.nsmallest(10, 'best_mape')[['store_id', 'item_id', 'best_mape', 'best_fit_algorithm']].copy()
                top_performers['Store-Item'] = top_performers['store_id'].astype(str) + ' - ' + top_performers['item_id'].astype(str)
                
                fig_top = px.bar(
                    top_performers,
                    x='Store-Item',
                    y='best_mape',
                    title='Top 10 Best Performing Store-Items (Lowest MAPE)',
                    labels={'best_mape': 'MAPE (%)', 'Store-Item': 'Store - Item'},
                    color='best_mape',
                    color_continuous_scale='Greens_r'
                )
                fig_top.update_layout(
                    height=320,
                    showlegend=False,
                    margin=dict(l=80, r=20, t=50, b=80),
                    title=dict(font=dict(size=13), pad=dict(b=15)),
                    xaxis_title=dict(font=dict(size=11)),
                    yaxis_title=dict(font=dict(size=11)),
                    paper_bgcolor='white',
                    plot_bgcolor='white',
                    xaxis=dict(gridcolor='#E8E8E8', gridwidth=1, tickangle=-45),
                    yaxis=dict(gridcolor='#E8E8E8', gridwidth=1)
                )
                st.plotly_chart(fig_top, use_container_width=True)
        else:
            st.warning("Best fit ML data not available. Please ensure the best fit ML file exists.")

    # ----------------------
    # Probability Forecasting Tab
    # ----------------------
    with tab4:
        st.subheader("Probability Forecasting")
        
        # Load probability forecast file
        prob_fcst_path = (OUTPUTS_PATH / "future_probability_forecasts.csv").resolve()
        prob_fcst_df = None
        
        missing_prob = []
        if prob_fcst_path.exists():
            try:
                prob_fcst_df = pd.read_csv(prob_fcst_path)
                if 'date' in prob_fcst_df.columns:
                    prob_fcst_df['date'] = pd.to_datetime(prob_fcst_df['date'])
            except Exception as e:
                st.error(f"Error loading probability forecast file: {e}")
        else:
            missing_prob.append("future_probability_forecasts.csv")

        if missing_prob:
            show_missing_outputs("Probability Forecasting", missing_prob)
        
        if prob_fcst_df is not None and not prob_fcst_df.empty:
            # Get unique stores and items (no algorithm filter)
            all_stores_prob = []
            all_items_prob = []
            
            if 'store_id' in prob_fcst_df.columns:
                all_stores_prob = sorted(prob_fcst_df['store_id'].dropna().unique().tolist())
            if 'item_id' in prob_fcst_df.columns:
                all_items_prob = sorted(prob_fcst_df['item_id'].dropna().unique().tolist())
            
            # Filters - Store and Item only
            with st.expander("📊 Select Store & Item", expanded=True):
                prob_filter_col1, prob_filter_col2 = st.columns(2)
                with prob_filter_col1:
                    prob_store_options = ['All'] + all_stores_prob if all_stores_prob else ['All']
                    selected_prob_store = st.selectbox(
                        'Store',
                        options=prob_store_options,
                        index=0,
                        key='prob_store'
                    )
                with prob_filter_col2:
                    if selected_prob_store == 'All':
                        prob_item_options = ['All'] + all_items_prob if all_items_prob else ['All']
                    else:
                        filtered_items_prob = sorted(
                            prob_fcst_df.loc[prob_fcst_df['store_id'] == selected_prob_store, 'item_id']
                            .dropna().unique().tolist()
                        ) if 'item_id' in prob_fcst_df.columns else []
                        prob_item_options = ['All'] + filtered_items_prob
                    
                    selected_prob_item = st.selectbox(
                        'Item',
                        options=prob_item_options,
                        index=0,
                        key='prob_item'
                    )
            
            # Show visualization when specific store-item is selected
            if (selected_prob_store != 'All' and selected_prob_item != 'All' and prob_fcst_df is not None):
                
                # Filter probability forecast data (use first algorithm if multiple exist, or aggregate)
                forecast_prob_df = prob_fcst_df[
                    (prob_fcst_df['store_id'] == selected_prob_store) &
                    (prob_fcst_df['item_id'] == selected_prob_item)
                ].copy()
                
                # If multiple algorithms, use the first one or aggregate
                if 'algorithm' in forecast_prob_df.columns and forecast_prob_df['algorithm'].nunique() > 1:
                    # Use first algorithm or take mean across algorithms
                    first_algo = forecast_prob_df['algorithm'].iloc[0]
                    forecast_prob_df = forecast_prob_df[forecast_prob_df['algorithm'] == first_algo].copy()
                
                if not forecast_prob_df.empty:
                    forecast_prob_df = forecast_prob_df.sort_values('date').reset_index(drop=True)
                    
                    # Ensure date column is datetime
                    if not pd.api.types.is_datetime64_any_dtype(forecast_prob_df['date']):
                        forecast_prob_df['date'] = pd.to_datetime(forecast_prob_df['date'])
                    
                    # Calculate threshold range and default value
                    threshold_default = None
                    threshold_min = None
                    threshold_max = None
                    if 'point_forecast' in forecast_prob_df.columns:
                        max_val = forecast_prob_df[['q95', 'point_forecast']].max().max() if 'q95' in forecast_prob_df.columns else forecast_prob_df['point_forecast'].max()
                        min_val = forecast_prob_df[['q05', 'point_forecast']].min().min() if 'q05' in forecast_prob_df.columns else forecast_prob_df['point_forecast'].min()
                        threshold_default = float(forecast_prob_df['point_forecast'].mean())
                        threshold_min = float(min_val)
                        threshold_max = float(max_val * 1.2)
                    
                    # Initialize threshold in session state if not exists
                    if 'prob_threshold_value' not in st.session_state:
                        st.session_state.prob_threshold_value = threshold_default
                    
                    # ========== MAIN ROW: Fan Chart (Full Width) ==========
                    st.markdown("### 📈 Probability Forecast (Fan Chart)")
                    
                    # Create fan chart with red and blue color scheme
                    fig = go.Figure()
                    
                    forecast_dates = forecast_prob_df['date'].tolist()
                    
                    # Fan chart: Add bands with red and blue color scheme
                    # Outer band: q05 to q95 (90% CI) - Light red/pink
                    if 'q05' in forecast_prob_df.columns and 'q95' in forecast_prob_df.columns:
                        q05_values = forecast_prob_df['q05'].tolist()
                        q95_values = forecast_prob_df['q95'].tolist()
                        fig.add_trace(go.Scatter(
                            x=forecast_dates + forecast_dates[::-1],
                            y=q95_values + q05_values[::-1],
                            fill='toself',
                            fillcolor='rgba(239, 68, 68, 0.12)',  # Light red, α≈0.12
                            line=dict(color='rgba(239, 68, 68, 0.2)', width=1),
                            hoverinfo="skip",
                            showlegend=True,
                            name='90% CI (q05-q95)',
                            legendgroup="bands"
                        ))
                    
                    # Inner band: q25 to q75 (50% IQR) - Medium blue
                    if 'q25' in forecast_prob_df.columns and 'q75' in forecast_prob_df.columns:
                        q25_values = forecast_prob_df['q25'].tolist()
                        q75_values = forecast_prob_df['q75'].tolist()
                        fig.add_trace(go.Scatter(
                            x=forecast_dates + forecast_dates[::-1],
                            y=q75_values + q25_values[::-1],
                            fill='toself',
                            fillcolor='rgba(59, 130, 246, 0.35)',  # Medium blue, α≈0.35
                            line=dict(color='rgba(59, 130, 246, 0.4)', width=1),
                            hoverinfo="skip",
                            showlegend=True,
                            name='50% IQR (q25-q75)',
                            legendgroup="bands"
                        ))
                        
                    # Plot median (q50) as solid blue line - only if different from point forecast
                    if 'q50' in forecast_prob_df.columns and 'point_forecast' in forecast_prob_df.columns:
                        # Check if median differs from point forecast
                        median_diff = (forecast_prob_df['q50'] != forecast_prob_df['point_forecast']).any()
                        if median_diff:
                            fig.add_trace(go.Scatter(
                                x=forecast_dates,
                                y=forecast_prob_df['q50'],
                                mode='lines',
                                name='Median (q50)',
                                line=dict(color='#3B82F6', width=3, dash='solid'),  # Blue
                                hovertemplate='<b>Median (q50)</b><br>' +
                                              'Date: %{x|%Y-%m-%d}<br>' +
                                              'Value: %{y:.2f}<br>' +
                                              '<extra></extra>'
                            ))
                    
                    # Plot point forecast as dashed red line (on top) - ALWAYS show
                    if 'point_forecast' in forecast_prob_df.columns:
                        fig.add_trace(go.Scatter(
                            x=forecast_dates,
                            y=forecast_prob_df['point_forecast'],
                            mode='lines',
                            name='Point Forecast',
                            line=dict(color='#DC2626', width=3, dash='dash'),  # Red
                            hovertemplate='<b>Point Forecast</b><br>' +
                                          'Date: %{x|%Y-%m-%d}<br>' +
                                          'Forecast: %{y:.2f}<br>' +
                                          '<extra></extra>'
                        ))
                    
                    # Add threshold line if set (only if within reasonable range)
                    threshold = st.session_state.get('prob_threshold_value', threshold_default)
                    if threshold is not None:
                        y_max = forecast_prob_df['q95'].max() if 'q95' in forecast_prob_df.columns else forecast_prob_df['point_forecast'].max()
                        if threshold <= y_max * 1.1:  # Only show if not too far out of range
                            fig.add_hline(
                                y=threshold,
                                line_dash="dot",
                                line_color="#EF4444",  # Red-500
                                line_width=2.5,
                                annotation_text=f"Threshold: {threshold:.0f}",
                                annotation_position="right",
                                annotation_font_size=11,
                                annotation_font_color="#EF4444"
                            )
                    
                    # Update layout - improved styling
                    fig.update_layout(
                        title=dict(
                            text=f'Probability Forecast: {selected_prob_store} - {selected_prob_item}',
                            font=dict(size=16, color='#1F2937')
                        ),
                        xaxis_title='Date',
                        yaxis_title='Units Sold',
                        height=600,
                        hovermode='x unified',
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=-0.12,
                            xanchor="center",
                            x=0.5,
                            font=dict(size=11),
                            bgcolor='rgba(255,255,255,0.8)',
                            bordercolor='rgba(0,0,0,0.1)',
                            borderwidth=1
                        ),
                        paper_bgcolor='white',
                        plot_bgcolor='#FAFAFA',
                        margin=dict(l=60, r=40, t=70, b=90),
                        xaxis=dict(
                            showgrid=True, 
                            gridcolor='rgba(200, 200, 200, 0.3)',
                            gridwidth=1,
                            showline=True,
                            linecolor='rgba(200, 200, 200, 0.5)'
                        ),
                        yaxis=dict(
                            showgrid=True, 
                            gridcolor='rgba(200, 200, 200, 0.3)',
                            gridwidth=1,
                            showline=True,
                            linecolor='rgba(200, 200, 200, 0.5)'
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
                    
                    st.divider()
                    
                    # ========== BOTTOM ROW: Distribution & Quantile Table Side by Side ==========
                    st.markdown("### 📊 Detailed View for Selected Date")
                    
                    # Date selector in its own row above the distribution and table
                    selected_date = st.selectbox(
                        "Select Date for Detailed View",
                        options=forecast_prob_df['date'].tolist(),
                        format_func=lambda x: x.strftime('%Y-%m-%d'),
                        key='prob_selected_date',
                        help="Choose a date to see its probability distribution and quantile values"
                    )
                    
                    st.markdown("")  # Small spacing
                    
                    # Distribution and Quantile table side by side
                    dist_col1, dist_col2 = st.columns(2)
                    
                    with dist_col1:
                        st.markdown("#### 📊 Probability Distribution")
                        
                        if selected_date:
                            selected_row = forecast_prob_df[forecast_prob_df['date'] == selected_date].iloc[0]
                            
                            # Probability density visualization (histogram approximation)
                            
                            # Create a simple histogram from quantiles
                            quantile_values = []
                            quantile_labels = []
                            
                            for q_col in ['q05', 'q10', 'q25', 'q50', 'q75', 'q90', 'q95']:
                                if q_col in selected_row:
                                    quantile_values.append(selected_row[q_col])
                                    quantile_labels.append(q_col)
                            
                            if quantile_values:
                                # Create histogram-like visualization with gradient colors
                                fig_dist = go.Figure()
                                
                                # Create bars between quantiles with gradient colors (red to green)
                                # Colors from low (red) to high (green) representing risk
                                colors = [
                                    'rgba(239, 68, 68, 0.7)',   # q05-q10: Red (low values)
                                    'rgba(245, 158, 11, 0.7)',  # q10-q25: Orange
                                    'rgba(251, 191, 36, 0.7)',  # q25-q50: Yellow
                                    'rgba(34, 197, 94, 0.7)',  # q50-q75: Green (median range)
                                    'rgba(251, 191, 36, 0.7)',  # q75-q90: Yellow
                                    'rgba(245, 158, 11, 0.7)',  # q90-q95: Orange
                                ]
                                
                                for i in range(len(quantile_values) - 1):
                                    fig_dist.add_trace(go.Bar(
                                        x=[(quantile_values[i] + quantile_values[i+1]) / 2],
                                        y=[1],
                                        width=quantile_values[i+1] - quantile_values[i],
                                        name=f'{quantile_labels[i]}-{quantile_labels[i+1]}',
                                        marker_color=colors[i] if i < len(colors) else colors[-1],
                                        opacity=0.7,
                                        showlegend=False
                                    ))
                                
                                # Add vertical lines for key quantiles with better styling
                                quantile_line_colors = {
                                    'q05': '#DC2626',  # Red
                                    'q50': '#1E3A8A',  # Dark blue
                                    'q95': '#059669'   # Green
                                }
                                
                                for q_val, q_label in zip(quantile_values, quantile_labels):
                                    if q_label in ['q05', 'q50', 'q95']:
                                        fig_dist.add_vline(
                                            x=q_val,
                                            line_dash="dash",
                                            line_color=quantile_line_colors.get(q_label, 'black'),
                                            line_width=2,
                                            annotation_text=q_label,
                                            annotation_position="top",
                                            annotation_font_size=11,
                                            annotation_font_color=quantile_line_colors.get(q_label, 'black')
                                        )
                                
                                # Compact height - balanced appearance
                                chart_height = 280
                                
                                # Calculate tight x-axis range
                                x_min = quantile_values[0] * 0.98
                                x_max = quantile_values[-1] * 1.02
                                
                                fig_dist.update_layout(
                                    title=dict(
                                        text=f"Distribution for {selected_date.strftime('%Y-%m-%d')}",
                                        font=dict(size=13, color='#1F2937')
                                    ),
                                    xaxis_title="Units Sold",
                                    yaxis_title="Density",
                                    height=chart_height,
                                    showlegend=False,
                                    paper_bgcolor='white',
                                    plot_bgcolor='#FAFAFA',
                                    margin=dict(l=50, r=30, t=50, b=15),  # Reduced bottom margin only
                                    xaxis=dict(
                                        showgrid=True, 
                                        gridcolor='rgba(200, 200, 200, 0.3)',
                                        range=[x_min, x_max],
                                        fixedrange=True
                                    ),
                                    yaxis=dict(
                                        showgrid=True, 
                                        gridcolor='rgba(200, 200, 200, 0.3)',
                                        range=[0, 1.05],
                                        fixedrange=True
                                    )
                                )
                                
                                st.plotly_chart(
                                    fig_dist, 
                                    use_container_width=True, 
                                    config={'displayModeBar': False}
                                )
                                
                                # Move probability section here to fill the gap below chart
                                threshold = st.session_state.get('prob_threshold_value', threshold_default)
                                if threshold is not None and 'q05' in selected_row and 'q95' in selected_row:
                                    date_q05 = selected_row['q05']
                                    date_q95 = selected_row['q95']
                                    
                                    if threshold <= date_q05:
                                        prob_above = 0.95
                                    elif threshold >= date_q95:
                                        prob_above = 0.05
                                    else:
                                        prob_above = 1 - (threshold - date_q05) / (date_q95 - date_q05) * 0.9 - 0.05
                                        prob_above = max(0.05, min(0.95, prob_above))
                                    
                                    # Color code risk
                                    if prob_above >= 0.7:
                                        risk_color = "🟢"
                                        risk_text = "Low Risk"
                                        bar_color = "#10B981"
                                    elif prob_above >= 0.3:
                                        risk_color = "🟡"
                                        risk_text = "Medium Risk"
                                        bar_color = "#F59E0B"
                                    else:
                                        risk_color = "🔴"
                                        risk_text = "High Risk"
                                        bar_color = "#EF4444"
                                    
                                    # Probability section below chart - fills the gap
                                    st.markdown("<br>", unsafe_allow_html=True)
                                    st.markdown(f"**P(Demand > {threshold:.0f})**")
                                    
                                    # Compact layout for probability display
                                    prob_display_col1, prob_display_col2 = st.columns([3, 2])
                                    with prob_display_col1:
                                        st.markdown(f"<div style='font-size: 18px; font-weight: bold;'>{prob_above*100:.1f}%</div>", unsafe_allow_html=True)
                                    with prob_display_col2:
                                        st.markdown(f"<div style='font-size: 14px; padding-top: 4px;'>{risk_color} {risk_text}</div>", unsafe_allow_html=True)
                                    
                                    # Progress bar
                                    st.markdown(
                                        f'<div style="background-color: {bar_color}; height: 32px; width: 100%; border-radius: 6px; margin-top: 10px; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 13px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">{prob_above*100:.1f}%</div>',
                                        unsafe_allow_html=True
                                    )
                                
                    with dist_col2:
                        st.markdown("#### 📋 Quantile Values")
                        
                        # Threshold slider moved here (in quantile section)
                        if threshold_default is not None and threshold_min is not None and threshold_max is not None:
                            st.markdown("**🎚️ Demand Threshold**")
                            threshold = st.slider(
                                "Set threshold value",
                                min_value=threshold_min,
                                max_value=threshold_max,
                                value=st.session_state.get('prob_threshold_value', threshold_default),
                                step=1.0,
                                key='prob_threshold_slider',
                                help="Adjust threshold to see probability of demand exceeding this value"
                            )
                            st.session_state.prob_threshold_value = threshold
                            st.markdown("---")
                        
                        if selected_date:
                            selected_row = forecast_prob_df[forecast_prob_df['date'] == selected_date].iloc[0]
                            
                            # Show quantile values table
                            quantile_values = []
                            quantile_labels = []
                            
                            for q_col in ['q05', 'q10', 'q25', 'q50', 'q75', 'q90', 'q95']:
                                if q_col in selected_row:
                                    quantile_values.append(selected_row[q_col])
                                    quantile_labels.append(q_col)
                            
                            if quantile_values:
                                quantile_display = pd.DataFrame({
                                    'Quantile': quantile_labels,
                                    'Value': quantile_values  # Keep as numbers for proper formatting
                                })
                                
                                # Calculate exact height based on number of rows (no extra rows)
                                num_rows = len(quantile_display)
                                # Compact: header (40px) + row height (32px each) + minimal padding
                                table_height = 40 + (num_rows * 32) + 10
                                
                                # Styled table with exact height - no extra rows
                                st.dataframe(
                                    quantile_display, 
                                    use_container_width=True, 
                                    hide_index=True, 
                                    height=table_height,  # Exact height, no extra rows
                                    column_config={
                                        "Quantile": st.column_config.TextColumn(
                                            "Quantile",
                                            width="small",
                                            help="Quantile level (e.g., q05 = 5th percentile)"
                                        ),
                                        "Value": st.column_config.NumberColumn(
                                            "Value",
                                            width="medium",
                                            format="%.2f",
                                            help="Forecast value for this quantile"
                                        )
                                    }
                                )
                    
                    st.divider()
                    
                    # ========== BOTTOM ROW: Quantile Forecast Table ==========
                    st.markdown("### 📋 Complete Quantile Forecast Table")
                    
                    # Prepare display dataframe
                    display_cols = ['date', 'point_forecast', 'q05', 'q25', 'q50', 'q75', 'q95']
                    available_cols = [col for col in display_cols if col in forecast_prob_df.columns]
                    display_df = forecast_prob_df[available_cols].copy()
                    
                    # Format numeric columns
                    numeric_cols = display_df.select_dtypes(include=[np.number]).columns
                    for col in numeric_cols:
                        display_df[col] = display_df[col].round(2)
                    
                    # Format date column for better display (keep as datetime for DateColumn)
                    if 'date' in display_df.columns:
                        if not pd.api.types.is_datetime64_any_dtype(display_df['date']):
                            display_df['date'] = pd.to_datetime(display_df['date'])
                    
                    # Build column config dynamically
                    column_config = {}
                    if 'date' in display_df.columns:
                        column_config["date"] = st.column_config.DateColumn(
                            "Date",
                            width="small",
                            format="YYYY-MM-DD"
                        )
                    if 'point_forecast' in display_df.columns:
                        column_config["point_forecast"] = st.column_config.NumberColumn(
                            "Point Forecast",
                            width="medium",
                            format="%.2f"
                        )
                    for q_col in ['q05', 'q25', 'q50', 'q75', 'q95']:
                        if q_col in display_df.columns:
                            help_text = {
                                'q05': '5th percentile',
                                'q25': '25th percentile',
                                'q50': '50th percentile (median)',
                                'q75': '75th percentile',
                                'q95': '95th percentile'
                            }.get(q_col, '')
                            column_config[q_col] = st.column_config.NumberColumn(
                                q_col,
                                width="small",
                                format="%.2f",
                                help=help_text
                            )
                    
                    # Styled table with proper column configuration
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        height=400,
                        hide_index=True,
                        column_config=column_config if column_config else None
                    )
                    
                    # Download button
                    csv_data = forecast_prob_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label='📥 Download Probability Forecast CSV',
                        data=csv_data,
                        file_name=f'probability_forecast_{selected_prob_store}_{selected_prob_item}.csv',
                        mime='text/csv',
                        use_container_width=True
                    )
                    
                    # Legend & Explanation (as per spec)
                    with st.expander("📖 Understanding Quantiles & Fan Chart", expanded=False):
                        st.markdown("""
                        **Quantile Meanings:**
                        - **q05** = 5% chance demand ≤ this value (very pessimistic scenario)
                        - **q25** = 25% chance demand ≤ this value (below median)
                        - **q50** = 50% chance demand ≤ this value (median/most likely outcome)
                        - **q75** = 75% chance demand ≤ this value (above median)
                        - **q95** = 95% chance demand ≤ this value (very optimistic scenario)
                        
                        **Fan Chart Components:**
                        - **Point Forecast (dashed black line)**: Single best estimate from the model
                        - **Median (q50, solid blue line)**: Only shown if different from point forecast
                        - **50% IQR (q25-q75, darker blue band)**: 50% of probable outcomes fall within this range
                        - **90% CI (q05-q95, lighter blue band)**: 90% of probable outcomes fall within this range
                        
                        **Interpretation:**
                        - Wider bands indicate higher uncertainty in the forecast
                        - The threshold line shows your risk threshold; P(Demand > threshold) indicates stockout risk
                        """)
                else:
                    st.warning("No probability forecast data found for the selected store-item combination.")
            else:
                st.info("Please select a specific Store and Item to view the probability forecast visualization.")
        else:
            st.warning("Probability forecast data not available. Please ensure the probability forecast file exists.")

