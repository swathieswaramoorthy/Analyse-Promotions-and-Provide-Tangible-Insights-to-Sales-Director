import pandas as pd

campaigns_df = pd.read_csv('campaigns.csv')  # First file
stores_df = pd.read_csv('stores.csv')        # Second file
sales_data_df = pd.read_csv('sales.csv') # Third file
products_df = pd.read_csv('products.csv')     # Fourth file

merged_df = pd.merge(sales_data_df, stores_df, on='store_id', how='left')

merged_df = pd.merge(merged_df, campaigns_df, on='campaign_id', how='left')

final_df = pd.merge(merged_df, products_df, on='product_code', how='left')

final_df.to_csv('combined_data.csv', index=False)
