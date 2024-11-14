import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsRegressor

# Load the data from the CSV file
df = pd.read_csv('combined_data.csv')

# Preprocess categorical data with Label Encoding
label_encoder = LabelEncoder()

# Label encode the categorical columns
for col in ['promo_type', 'city', 'campaign_name', 'category']:
    df[col] = label_encoder.fit_transform(df[col])

# Convert 'start_date' to numeric days from a fixed start date (e.g., '01-01-2023')
df['start_date'] = pd.to_datetime(df['start_date'], format='%d-%m-%Y')
fixed_start_date = pd.to_datetime('01-01-2023', format='%d-%m-%Y')
df['start_date'] = (df['start_date'] - fixed_start_date).dt.days

# Convert 'end_date' to numeric days from the same fixed start date
df['end_date'] = pd.to_datetime(df['end_date'], format='%d-%m-%Y')
df['end_date'] = (df['end_date'] - fixed_start_date).dt.days

# Define features (X) and target variable (y)
# Choose relevant columns for X, including any new columns created
X = df[['base_price', 'promo_type', 'quantity_sold(before_promo)', 'city', 'campaign_name', 'start_date', 'end_date']]
y = df['quantity_sold(after_promo)']  # Target variable (quantity sold after promo)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply KNN regression
knn = KNeighborsRegressor(n_neighbors=3)  # Adjust n_neighbors as needed
knn.fit(X_scaled, y)

# Make predictions using the trained KNN model
df['predicted_quantity_sold'] = knn.predict(X_scaled)

# Save the dataframe with both original and predicted values to a new CSV file
df.to_csv('combined_data_with_knn_predictions.csv', index=False)

# Optionally, print a preview of the dataframe
print(df[['event_id', 'quantity_sold(after_promo)', 'predicted_quantity_sold']].head())