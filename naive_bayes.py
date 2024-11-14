import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the data from the CSV file
df = pd.read_csv('combined_data.csv')

# Preprocess categorical data: Convert 'promo_type' and 'category' to numeric using Label Encoding
label_encoder = LabelEncoder()
df['promo_type'] = label_encoder.fit_transform(df['promo_type'])
df['category'] = label_encoder.fit_transform(df['category'])
df['city'] = label_encoder.fit_transform(df['city'])

# Convert 'start_date' to a numeric value (e.g., the number of days from a start date)
df['start_date'] = pd.to_datetime(df['start_date'], format='%d-%m-%Y')
reference_date = pd.to_datetime('01-01-2022', format='%d-%m-%Y')
df['start_date'] = (df['start_date'] - reference_date).dt.days

# Features and target (X for features, y for target)
X = df[['base_price', 'promo_type', 'quantity_sold(before_promo)', 'start_date', 'category', 'city']]
y = df['quantity_sold(after_promo)']  # Target variable

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Discretize the target variable for Naive Bayes
y_bins = np.linspace(y.min(), y.max(), 5)  # Create 5 bins for quantity sold
y_discretized = np.digitize(y, bins=y_bins) - 1  # Binned labels (0 to 4)

# Apply Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X_scaled, y_discretized)

# Predict on the same data
y_pred_discretized = gnb.predict(X_scaled)

# Convert predicted bins back to approximate continuous values (average of bin range)
# Convert predicted bins back to approximate continuous values (average of bin range)
bin_centers = [(y_bins[i] + y_bins[i + 1]) / 2 for i in range(len(y_bins) - 1)]

# Adjust the predicted bins to stay within the range of bin_centers
df['predicted_quantity_sold'] = [
    bin_centers[int(pred)] if int(pred) < len(bin_centers) else bin_centers[-1]
    for pred in y_pred_discretized
]


# Save the dataframe with both original and predicted values to a new CSV file
df.to_csv('data_with_naive_bayes_predictions.csv', index=False)

# Print a preview of the dataframe
print(df[['vent_id', 'quantity_sold(after_promo)', 'predicted_quantity_sold']].head())

# Evaluate model performance
mse = mean_squared_error(y, df['predicted_quantity_sold'])
print(f"Mean Squared Error: {mse:.2f}")
