import pandas as pd
import numpy as np


def run_eda(df):
    def calculate_metrics(df):
        """Calculates all required metrics and prepares data structures."""

        # --- 1. OVERALL METRICS ---
        df['date'] = pd.to_datetime(df['date'])
        time_frame_start = df['date'].min().strftime('%Y-%m-%d')
        time_frame_end = df['date'].max().strftime('%Y-%m-%d')
        num_days = (df['date'].max() - df['date'].min()).days + 1
        total_units_sold = df['units_sold'].sum() # Existing calculation
        num_products = df['item_id'].nunique() # Existing calculation (Total SKUs)
        num_stores = df['store_id'].nunique()
        total_categories = df['cat_id'].nunique() # NEW CALCULATION

        overall_metrics = {
            'Time Frame': f"{time_frame_start} to {time_frame_end} ({num_days} days)",
            'Total Units Sold': total_units_sold, # ADDED to metrics dict
            'Total SKUs': num_products, # RENAMED from 'Number of Products' to match dashboard language
            'Total Product Categories': total_categories, # ADDED to metrics dict
            'Number of Stores': num_stores,
            'Average Sales Per Day (Total)': total_units_sold / num_days,
            'Average Sales Per Store': total_units_sold / num_stores,
        }

        # --- 2. STORE BREAKDOWN METRICS (INCLUDING OVERALL) ---
        
        # Calculate daily sales volume for all stores and overall
        daily_sales = df.groupby(['date', 'store_id'])['units_sold'].sum().reset_index()
        daily_sales_overall = df.groupby('date')['units_sold'].sum().reset_index()
        daily_sales_overall['store_id'] = 'Overall'
        
        daily_sales_full = pd.concat([daily_sales, daily_sales_overall], ignore_index=True)
        
        # Ensure all days are represented for Zero Sales Days calculation
        all_dates = pd.date_range(time_frame_start, time_frame_end)
        all_stores = df['store_id'].unique().tolist() + ['Overall']
        
        full_index = pd.MultiIndex.from_product([all_dates, all_stores], names=['date', 'store_id'])
        daily_sales_full = daily_sales_full.set_index(['date', 'store_id']).reindex(full_index, fill_value=0).reset_index()
        daily_sales_full.rename(columns={'units_sold': 'daily_units_sold'}, inplace=True)
        
        # Calculate store-level aggregates
        store_aggregates = daily_sales_full.groupby('store_id')['daily_units_sold'].agg(
            total_quantity_sold='sum',
            avg_sales_per_day='mean',
            std_sales='std',
            zero_sales_days=lambda x: (x == 0).sum()
        ).reset_index()
        
        # Calculate Sales Consistency (Coefficient of Variation)
        store_aggregates['Sales Consistency (CV)'] = np.where(
            store_aggregates['avg_sales_per_day'] > 0,
            store_aggregates['std_sales'] / store_aggregates['avg_sales_per_day'],
            0
        )
        store_aggregates.drop(columns=['std_sales'], inplace=True)

        # Calculate Avg Transaction Size (Units)
        trx_agg = df.groupby('store_id').agg(
            total_units_sold=('units_sold', 'sum'),
            total_transactions=('id', 'nunique')
        ).reset_index()
        
        trx_agg['Average Transaction Size (Units)'] = trx_agg['total_units_sold'] / trx_agg['total_transactions']
        
        # Add Overall transaction size
        overall_trx = {
            'store_id': 'Overall',
            'Average Transaction Size (Units)': df['units_sold'].sum() / df['id'].nunique()
        }
        trx_agg = pd.concat([trx_agg, pd.DataFrame([overall_trx])], ignore_index=True)

        # Combine all store breakdown metrics
        store_breakdown = store_aggregates.merge(
            trx_agg[['store_id', 'Average Transaction Size (Units)']],
            on='store_id',
            how='left'
        )
        
        # --- 3. PRODUCT CATEGORY DATA ---
        
        # Total sales by category for the Treemap
        category_sales_df = df.groupby('cat_id')['units_sold'].sum().reset_index()
        category_sales_df.rename(columns={'units_sold': 'Total Quantity Sold'}, inplace=True)
        
        # Daily sales by category for trend/seasonality analysis
        category_daily_sales = df.groupby(['date', 'cat_id'])['units_sold'].sum().reset_index()
        
        # List of unique items (SKUs) per category
        sku_count_by_cat = df.groupby('cat_id')['item_id'].nunique().to_dict()

        return {
            'df': df,
            'overall_metrics': overall_metrics,
            'store_breakdown': store_breakdown,
            'category_sales_treemap': category_sales_df,
            'category_daily_sales': category_daily_sales,
            'sku_count_by_cat': sku_count_by_cat,
            'num_days': num_days
        }

    # Execute the processing function to make the results available for import
    processed_data = calculate_metrics(df)

    # Extract processed data
    df_sales = processed_data['df']
    overall_metrics = processed_data['overall_metrics']
    store_breakdown_metrics = processed_data['store_breakdown']
    category_sales_treemap = processed_data['category_sales_treemap']
    category_daily_sales = processed_data['category_daily_sales']
    sku_count_by_cat = processed_data['sku_count_by_cat']
    total_days = processed_data['num_days']
    
    # Return all results
    return {
        'df_sales': df_sales,
        'overall_metrics': overall_metrics,
        'store_breakdown_metrics': store_breakdown_metrics,
        'category_daily_sales': category_daily_sales,
        'category_sales_treemap': category_sales_treemap,
        'sku_count_by_cat': sku_count_by_cat,
        'total_days': total_days
    }