import pandas as pd

# Load each CSV file into a DataFrame
campaigns_df = pd.read_csv('campaigns.csv')  # First file
stores_df = pd.read_csv('stores.csv')        # Second file
sales_data_df = pd.read_csv('sales.csv') # Third file
products_df = pd.read_csv('products.csv')     # Fourth file

# Merge sales_data with stores on 'store_id'
merged_df = pd.merge(sales_data_df, stores_df, on='store_id', how='left')

# Merge the result with campaigns on 'campaign_id'
merged_df = pd.merge(merged_df, campaigns_df, on='campaign_id', how='left')

# Merge the result with products on 'product_code'
final_df = pd.merge(merged_df, products_df, on='product_code', how='left')

# Save the combined DataFrame to a new CSV file
final_df.to_csv('combined_data.csv', index=False)
