"""Segmentation page for Demand Forecasting Dashboard"""
import pandas as pd
import streamlit as st
import plotly.express as px
from .config import OUTPUTS_PATH, load_csv_file


def page_segmentation_and_rules():
    """Page for displaying Segmentation and Rules data"""
    # Load data from CSV files
    seg_df_path = (OUTPUTS_PATH / "seg_df.csv").resolve()
    seg_df = None
    
    # Check if files exist
    missing_files = []
    if not seg_df_path.exists():
        missing_files.append("seg_df.csv")
    if missing_files:
        st.error("Segmentation outputs not available.")
        st.info(f"""
        Segmentation outputs are missing. Please run the pipeline first:
        
        1. Go to the **Data Upload** page
        2. Upload the required CSV files
        3. Click the **Run main.py** button
        
        Missing files: {', '.join(missing_files)}
        """)
        st.stop()
    
    # Load files if they exist
    if seg_df_path.exists():
        seg_df = load_csv_file(seg_df_path)
    else:
        st.error(f" Segmentation file not found at: `{seg_df_path}`")
    
    # Use tabs to separate Segmentation
    tab1, = st.tabs(["Segmentation"])
    
    # Segmentation Tab
    with tab1:
        if seg_df is not None and not seg_df.empty:
            # Metrics row
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Records", len(seg_df))
            with col2:
                if 'store_id' in seg_df.columns and 'item_id' in seg_df.columns:
                    unique_combinations = seg_df.groupby(['store_id', 'item_id']).ngroups
                else:
                    unique_combinations = len(seg_df)
                st.metric("Unique Store-Item Combinations", unique_combinations)
            
            st.divider()
            
            # Filter controls
            with st.expander("Filter Options", expanded=False):
                filter_col1, filter_col2 = st.columns(2)
                
                with filter_col1:
                    if 'store_id' in seg_df.columns:
                        selected_stores = st.multiselect(
                            'Filter by Store',
                            options=sorted(seg_df['store_id'].unique().tolist()),
                            key='seg_store_filter'
                        )
                    else:
                        selected_stores = []
                
                with filter_col2:
                    if 'item_id' in seg_df.columns:
                        selected_items = st.multiselect(
                            'Filter by Item',
                            options=sorted(seg_df['item_id'].unique().tolist()),
                            key='seg_item_filter'
                        )
                    else:
                        selected_items = []
            
            # Apply filters
            filtered_df = seg_df.copy()
            if 'store_id' in seg_df.columns and selected_stores:
                filtered_df = filtered_df[filtered_df['store_id'].isin(selected_stores)]
            if 'item_id' in seg_df.columns and selected_items:
                filtered_df = filtered_df[filtered_df['item_id'].isin(selected_items)]
            
            # Show filtered count if filters are applied
            if selected_stores or selected_items:
                st.caption(f"Showing {len(filtered_df)} of {len(seg_df)} records")
            
            # ========== DATA TABLE SECTION (MOVED TO TOP) ==========
            st.divider()
            st.subheader("Segmentation Data Table")
            st.dataframe(
                filtered_df,
                use_container_width=True,
                height=300,
                hide_index=True
            )
            
            # ========== VISUALIZATIONS SECTION (MOVED BELOW TABLE) ==========
            st.divider()
            st.subheader("Segmentation Visualizations")
            
            # Identify segmentation columns (exclude store_id and item_id)
            seg_columns = [col for col in filtered_df.columns 
                          if col not in ['store_id', 'item_id'] and 
                          filtered_df[col].dtype == 'object']
            
            if seg_columns:
                # Create tabs for different types of visualizations
                viz_tab1, viz_tab2 = st.tabs([
                    "Distribution Charts",
                    "Cross-Segmentation Analysis"
                ])
                
                # Tab 1: Distribution Charts
                with viz_tab1:
                    # Create columns for better layout - 2 charts per row
                    num_cols = len(seg_columns)
                    if num_cols > 0:
                        cols_per_row = 2
                        for i in range(0, num_cols, cols_per_row):
                            chart_cols = st.columns(min(cols_per_row, num_cols - i))
                            for idx, col in enumerate(seg_columns[i:i+cols_per_row]):
                                with chart_cols[idx]:
                                    try:
                                        value_counts = filtered_df[col].value_counts()
                                        if not value_counts.empty:
                                            # Create enhanced donut chart with better styling
                                            total_count = value_counts.sum()
                                            
                                            # Use vibrant color palette
                                            colors = px.colors.qualitative.Set3 if len(value_counts) <= 12 else px.colors.qualitative.Pastel
                                            
                                            # Create donut chart with compact size
                                            fig_pie = px.pie(
                                                values=value_counts.values,
                                                names=value_counts.index.astype(str),
                                                title=f'{col.replace("_", " ").title()}',
                                                height=280,
                                                hole=0.5,
                                                color_discrete_sequence=colors
                                            )
                                            
                                            # Clean, compact styling
                                            fig_pie.update_traces(
                                                textposition='outside',
                                                textinfo='percent',
                                                textfont_size=10,
                                                marker=dict(
                                                    line=dict(color='#FFFFFF', width=2)
                                                ),
                                                pull=0.02,
                                                hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>Percentage: %{percent}<extra></extra>'
                                            )
                                            
                                            # Compact layout with proper spacing to avoid overlap
                                            fig_pie.update_layout(
                                                showlegend=True,
                                                legend=dict(
                                                    orientation="v",
                                                    yanchor="middle",
                                                    y=0.5,
                                                    xanchor="left",
                                                    x=1.05,
                                                    font=dict(size=9),
                                                    bgcolor='rgba(255,255,255,0.9)',
                                                    bordercolor='#E0E0E0',
                                                    borderwidth=1
                                                ),
                                                title=dict(
                                                    x=0.5,
                                                    font=dict(size=13),
                                                    xanchor='center',
                                                    y=0.95,
                                                    pad=dict(b=10)
                                                ),
                                                hovermode='closest',
                                                margin=dict(l=10, r=100, t=40, b=10),
                                                paper_bgcolor='white',
                                                plot_bgcolor='rgba(0,0,0,0)',
                                                uniformtext_minsize=8,
                                                uniformtext_mode='hide'
                                            )
                                            
                                            st.plotly_chart(fig_pie, use_container_width=True, key=f'pie_{col}')
                                            
                                            # Show summary statistics in a compact card style
                                            with st.expander(f"{col.replace('_', ' ').title()} Summary", expanded=False):
                                                stats_col1, stats_col2 = st.columns(2)
                                                with stats_col1:
                                                    st.metric("Total Categories", len(value_counts), delta=None)
                                                    st.metric("Most Common", value_counts.index[0], delta=None)
                                                with stats_col2:
                                                    st.metric("Total Records", f"{total_count:,}", delta=None)
                                                    st.metric("Most Common Count", f"{value_counts.iloc[0]:,}", delta=None)
                                    except Exception as e:
                                        st.warning(f"Could not create chart for {col}: {str(e)}")
                
                # Tab 2: Cross-Segmentation Analysis
                with viz_tab2:
                    if len(seg_columns) >= 2:
                        st.markdown("**Cross-Segmentation Analysis**")
                        
                        # Select two columns for cross-tabulation
                        cross_col1, cross_col2 = st.columns(2)
                        with cross_col1:
                            selected_col1 = st.selectbox(
                                'Select First Segment',
                                options=seg_columns,
                                key='cross_seg_1'
                            )
                        with cross_col2:
                            remaining_cols = [c for c in seg_columns if c != selected_col1]
                            if remaining_cols:
                                selected_col2 = st.selectbox(
                                    'Select Second Segment',
                                    options=remaining_cols,
                                    key='cross_seg_2'
                                )
                            else:
                                selected_col2 = None
                        
                        if selected_col1 and selected_col2:
                            try:
                                # Create cross-tabulation
                                cross_tab = pd.crosstab(
                                    filtered_df[selected_col1], 
                                    filtered_df[selected_col2],
                                    margins=True
                                )
                                
                                # Display cross-tabulation table
                                st.markdown(f"**Cross-tabulation: {selected_col1} vs {selected_col2}**")
                                st.dataframe(cross_tab, use_container_width=True)
                                
                                # Create cross-tabulation without margins for visualizations
                                cross_tab_no_margins = pd.crosstab(
                                    filtered_df[selected_col1], 
                                    filtered_df[selected_col2]
                                )
                                
                                if not cross_tab_no_margins.empty:
                                    # Visualization type selector
                                    viz_type = st.radio(
                                        'Select Visualization Type',
                                        ['Grouped Bar Chart', 'Stacked Bar Chart', 'Sunburst Chart'],
                                        horizontal=True,
                                        key=f'viz_type_{selected_col1}_{selected_col2}'
                                    )
                                    
                                    if viz_type == 'Grouped Bar Chart':
                                        # Create grouped bar chart with professional styling
                                        fig_grouped = px.bar(
                                            cross_tab_no_margins,
                                            barmode='group',
                                            title=f'<b>{selected_col1.replace("_", " ").title()}</b> Distribution by <b>{selected_col2.replace("_", " ").title()}</b>',
                                            labels={'value': 'Count', 'index': selected_col1.replace("_", " ").title()},
                                            color_discrete_sequence=px.colors.qualitative.Set3
                                        )
                                        fig_grouped.update_traces(
                                            marker=dict(
                                                line=dict(width=1.5, color='white'),
                                                opacity=0.9
                                            ),
                                            hovertemplate='<b>%{fullData.name}</b><br>' +
                                                          '<span style="color:#2E86AB;">Count:</span> %{y:,}<br>' +
                                                          '<extra></extra>'
                                        )
                                        fig_grouped.update_layout(
                                            height=320,
                                            xaxis_title=dict(text=selected_col1.replace("_", " ").title(), font=dict(size=11)),
                                            yaxis_title=dict(text="Count", font=dict(size=11)),
                                            legend_title=dict(text=selected_col2.replace("_", " ").title(), font=dict(size=10)),
                                            legend=dict(
                                                bgcolor='rgba(255,255,255,0.9)',
                                                bordercolor='#E0E0E0',
                                                borderwidth=1,
                                                font=dict(size=9)
                                            ),
                                            hovermode='x unified',
                                            margin=dict(l=50, r=20, t=50, b=50),
                                            title=dict(
                                                font=dict(size=13),
                                                pad=dict(b=15)
                                            ),
                                            paper_bgcolor='white',
                                            plot_bgcolor='white',
                                            xaxis=dict(gridcolor='#E8E8E8', gridwidth=1),
                                            yaxis=dict(gridcolor='#E8E8E8', gridwidth=1)
                                        )
                                        st.plotly_chart(fig_grouped, use_container_width=True)
                                    
                                    elif viz_type == 'Stacked Bar Chart':
                                        # Create stacked bar chart with professional styling
                                        fig_stacked = px.bar(
                                            cross_tab_no_margins,
                                            barmode='stack',
                                            title=f'<b>{selected_col1.replace("_", " ").title()}</b> Distribution by <b>{selected_col2.replace("_", " ").title()}</b> (Stacked)',
                                            labels={'value': 'Count', 'index': selected_col1.replace("_", " ").title()},
                                            color_discrete_sequence=px.colors.qualitative.Set3
                                        )
                                        fig_stacked.update_traces(
                                            marker=dict(
                                                line=dict(width=1.5, color='white'),
                                                opacity=0.9
                                            ),
                                            hovertemplate='<b>%{fullData.name}</b><br>' +
                                                          '<span style="color:#2E86AB;">Count:</span> %{y:,}<br>' +
                                                          '<span style="color:#2E86AB;">Percentage:</span> %{percentOfStack:,.1f}%<br>' +
                                                          '<extra></extra>'
                                        )
                                        fig_stacked.update_layout(
                                            height=320,
                                            xaxis_title=dict(text=selected_col1.replace("_", " ").title(), font=dict(size=11)),
                                            yaxis_title=dict(text="Count", font=dict(size=11)),
                                            legend_title=dict(text=selected_col2.replace("_", " ").title(), font=dict(size=10)),
                                            legend=dict(
                                                bgcolor='rgba(255,255,255,0.9)',
                                                bordercolor='#E0E0E0',
                                                borderwidth=1,
                                                font=dict(size=9)
                                            ),
                                            hovermode='x unified',
                                            margin=dict(l=50, r=20, t=50, b=50),
                                            title=dict(
                                                font=dict(size=13),
                                                pad=dict(b=15)
                                            ),
                                            paper_bgcolor='white',
                                            plot_bgcolor='white',
                                            xaxis=dict(gridcolor='#E8E8E8', gridwidth=1),
                                            yaxis=dict(gridcolor='#E8E8E8', gridwidth=1)
                                        )
                                        st.plotly_chart(fig_stacked, use_container_width=True)
                                    
                                    elif viz_type == 'Sunburst Chart':
                                        # Prepare data for sunburst
                                        sunburst_data = []
                                        for idx1, val1 in enumerate(cross_tab_no_margins.index):
                                            for idx2, val2 in enumerate(cross_tab_no_margins.columns):
                                                count = cross_tab_no_margins.iloc[idx1, idx2]
                                                if count > 0:  # Only add non-zero values
                                                    sunburst_data.append({
                                                        'ids': f'{val1}-{val2}',
                                                        'labels': f'{val1} - {val2}',
                                                        'parents': str(val1),
                                                        'values': count
                                                    })
                                        # Add parent nodes
                                        for val1 in cross_tab_no_margins.index:
                                            total = cross_tab_no_margins.loc[val1].sum()
                                            sunburst_data.insert(0, {
                                                'ids': str(val1),
                                                'labels': str(val1),
                                                'parents': '',
                                                'values': total
                                            })
                                        
                                        if sunburst_data:
                                            sunburst_df = pd.DataFrame(sunburst_data)
                                            fig_sunburst = px.sunburst(
                                                sunburst_df,
                                                ids='ids',
                                                parents='parents',
                                                values='values',
                                                names='labels',
                                                title=f'{selected_col1} -> {selected_col2} Relationship',
                                                color_discrete_sequence=px.colors.qualitative.Pastel
                                            )
                                            fig_sunburst.update_traces(
                                                hovertemplate='<b>%{label}</b><br>' +
                                                              '<span style="color:#2E86AB;">Count:</span> %{value:,}<br>' +
                                                              '<extra></extra>',
                                                marker=dict(line=dict(width=2, color='white'))
                                            )
                                            fig_sunburst.update_layout(
                                                height=350,
                                                margin=dict(l=20, r=20, t=50, b=20),
                                                title=dict(
                                                    font=dict(size=13),
                                                    pad=dict(b=15)
                                                ),
                                                paper_bgcolor='white',
                                                plot_bgcolor='white'
                                            )
                                            st.plotly_chart(fig_sunburst, use_container_width=True)
                                        
                                    # Show percentage breakdown
                                    st.divider()
                                    st.markdown("**Percentage Breakdown**")
                                    percentage_tab = cross_tab_no_margins.div(cross_tab_no_margins.sum(axis=1), axis=0) * 100
                                    percentage_tab = percentage_tab.round(2)
                                    st.dataframe(percentage_tab, use_container_width=True)
                                    st.caption("Row percentages showing distribution of each category")
                                    
                            except Exception as e:
                                st.warning(f"Could not create cross-tabulation: {str(e)}")

                    if 'Volume_Segment' in seg_columns and 'Variability_Segment' in seg_columns:
                        st.divider()
                        st.markdown("**Volume vs Variability Matrix**")
                        try:
                            vol_var = pd.crosstab(filtered_df['Volume_Segment'], filtered_df['Variability_Segment'])
                            if not vol_var.empty:
                                fig_matrix = px.bar(
                                    vol_var.T,
                                    barmode='group',
                                    title='Volume vs Variability Distribution',
                                    labels={'value': 'Count', 'index': 'Variability Segment'},
                                    color_discrete_sequence=px.colors.qualitative.Set3
                                )
                                fig_matrix.update_traces(
                                    marker=dict(
                                        line=dict(width=1.5, color='white'),
                                        opacity=0.85
                                    ),
                                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                                  '<span style="color:#2E86AB;">Count:</span> %{y:,}<br>' +
                                                  '<extra></extra>'
                                )
                                fig_matrix.update_layout(
                                    height=320,
                                    xaxis_title=dict(text="Variability Segment", font=dict(size=11)),
                                    yaxis_title=dict(text="Count", font=dict(size=11)),
                                    legend_title=dict(text="Volume Segment", font=dict(size=10)),
                                    legend=dict(
                                        bgcolor='rgba(255,255,255,0.9)',
                                        bordercolor='#E0E0E0',
                                        borderwidth=1,
                                        font=dict(size=9)
                                    ),
                                    margin=dict(l=50, r=20, t=50, b=50),
                                    title=dict(
                                        font=dict(size=13),
                                        pad=dict(b=15)
                                    ),
                                    paper_bgcolor='white',
                                    plot_bgcolor='white',
                                    xaxis=dict(gridcolor='#E8E8E8', gridwidth=1),
                                    yaxis=dict(gridcolor='#E8E8E8', gridwidth=1),
                                    hovermode='x unified'
                                )
                                st.plotly_chart(fig_matrix, use_container_width=True)
                        except Exception as e:
                            st.info("Could not create Volume-Variability matrix")
        else:
            st.warning("No segmentation data available. Please ensure the segmentation file exists.")
        return
    
    # Rules Tab
    with tab2:
        if rules_df is not None and not rules_df.empty:
            # Metrics row
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Records", len(rules_df))
            with col2:
                if 'store_id' in rules_df.columns and 'item_id' in rules_df.columns:
                    unique_combinations = rules_df.groupby(['store_id', 'item_id']).ngroups
                else:
                    unique_combinations = len(rules_df)
                st.metric("Unique Store-Item Combinations", unique_combinations)
            
            st.divider()
            
            # Filter controls
            with st.expander("Filter Options", expanded=False):
                filter_col1, filter_col2 = st.columns(2)
                
                with filter_col1:
                    if 'store_id' in rules_df.columns:
                        selected_stores = st.multiselect(
                            'Filter by Store',
                            options=sorted(rules_df['store_id'].unique().tolist()),
                            key='rules_store_filter'
                        )
                    else:
                        selected_stores = []
                
                with filter_col2:
                    if 'item_id' in rules_df.columns:
                        selected_items = st.multiselect(
                            'Filter by Item',
                            options=sorted(rules_df['item_id'].unique().tolist()),
                            key='rules_item_filter'
                        )
                    else:
                        selected_items = []
            
            # Apply filters
            filtered_df = rules_df.copy()
            if 'store_id' in rules_df.columns and selected_stores:
                filtered_df = filtered_df[filtered_df['store_id'].isin(selected_stores)]
            if 'item_id' in rules_df.columns and selected_items:
                filtered_df = filtered_df[filtered_df['item_id'].isin(selected_items)]
            
            # Show filtered count if filters are applied
            if selected_stores or selected_items:
                st.caption(f"Showing {len(filtered_df)} of {len(rules_df)} records")
            
            st.dataframe(
                filtered_df,
                use_container_width=True,
                height=300,
                hide_index=True
            )
        else:
            st.warning("No rules data available. Please ensure the rules file exists.")

