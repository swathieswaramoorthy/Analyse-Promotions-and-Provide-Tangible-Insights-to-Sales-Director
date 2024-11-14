import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV files
sales = pd.read_csv('sales.csv')
stores = pd.read_csv('stores.csv')
products = pd.read_csv('products.csv')
campaigns = pd.read_csv('campaigns.csv', dayfirst=True)

# Convert start_date and end_date to datetime format
campaigns['start_date'] = pd.to_datetime(campaigns['start_date'], dayfirst=True)
campaigns['end_date'] = pd.to_datetime(campaigns['end_date'], dayfirst=True)

# Merge sales data with stores, products, and campaigns on appropriate columns
sales = sales.merge(stores, on='store_id')
sales = sales.merge(products, on='product_code')
sales = sales.merge(campaigns[['campaign_id', 'start_date', 'end_date']], on='campaign_id', how='left')

# Define the festival date range
festival_start = pd.to_datetime('2024-12-20')
festival_end = pd.to_datetime('2024-12-31')

# Create a boolean column 'is_festival' based on the campaign period falling in the festival range
sales['is_festival'] = sales.apply(lambda row: (row['start_date'] <= festival_end) and (row['end_date'] >= festival_start), axis=1)

# 1. Compare Quantity Sold Before and After Promo
sales['quantity_diff'] = sales['quantity_sold(after_promo)'] - sales['quantity_sold(before_promo)']
promo_comparison = sales.groupby(['is_festival', 'promo_type'])['quantity_diff'].sum().reset_index()

# Plot for promo effectiveness
plt.figure(figsize=(10, 6))
sns.barplot(x='promo_type', y='quantity_diff', hue='is_festival', data=promo_comparison)
plt.title('Effect of Promotions on Quantity Sold (Festival vs. Normal Days)')
plt.xlabel('Promotion Type')
plt.ylabel('Quantity Difference')
plt.legend(title='Festival')
plt.show()

# 2. City-Wise Sales Performance
city_performance = sales.groupby(['city', 'promo_type', 'is_festival'])['quantity_sold(after_promo)'].sum().reset_index()

# Plot city-wise performance
plt.figure(figsize=(12, 8))
sns.barplot(x='city', y='quantity_sold(after_promo)', hue='is_festival', data=city_performance)
plt.title('City-Wise Sales Performance (Festival vs. Normal Days)')
plt.xlabel('City')
plt.ylabel('Quantity Sold After Promo')
plt.legend(title='Festival')
plt.xticks(rotation=45)
plt.show()

# 3. Category Performance in Festivals vs. Normal Days
category_performance = sales.groupby(['category', 'is_festival'])['quantity_sold(after_promo)'].sum().reset_index()

# Plot category performance
plt.figure(figsize=(12, 8))
sns.barplot(x='category', y='quantity_sold(after_promo)', hue='is_festival', data=category_performance)
plt.title('Product Category Sales (Festival vs. Normal Days)')
plt.xlabel('Product Category')
plt.ylabel('Quantity Sold After Promo')
plt.legend(title='Festival')
plt.xticks(rotation=45)
plt.show()
