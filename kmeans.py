import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans

# Load the data from the CSV file
df = pd.read_csv('combined_data.csv')

# Preprocess categorical data with Label Encoding
label_encoder = LabelEncoder()
for col in ['promo_type', 'city', 'campaign_name', 'category']:
    df[col] = label_encoder.fit_transform(df[col])

# Convert start_date and end_date to numeric days from a fixed start date
df['start_date'] = pd.to_datetime(df['start_date'], format='%d-%m-%Y')
df['end_date'] = pd.to_datetime(df['end_date'], format='%d-%m-%Y')
fixed_start_date = pd.to_datetime('01-01-2023', format='%d-%m-%Y')
df['start_date'] = (df['start_date'] - fixed_start_date).dt.days
df['end_date'] = (df['end_date'] - fixed_start_date).dt.days

# Select features for clustering
X = df[['base_price', 'promo_type', 'quantity_sold(before_promo)', 'quantity_sold(after_promo)', 'city', 'start_date', 'end_date']]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply KMeans clustering (let's assume 3 clusters for this example)
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Save the dataframe with cluster predictions to a new CSV file
df.to_csv('combined_data_with_kmeans_clusters.csv', index=False)

# Plotting the clusters
plt.figure(figsize=(10, 7))
plt.scatter(df['base_price'], df['quantity_sold(after_promo)'], c=df['cluster'], cmap='viridis', alpha=0.7)
plt.title('K-Means Clustering of Promotional Sales')
plt.xlabel('Base Price')
plt.ylabel('Quantity Sold (After Promo)')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

# Print cluster centers (optional)
print("Cluster Centers:")
print(kmeans.cluster_centers_)
