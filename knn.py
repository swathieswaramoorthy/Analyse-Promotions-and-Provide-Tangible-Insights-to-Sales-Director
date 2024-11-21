# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# Load the CSV files
campaigns = pd.read_csv('campaigns.csv')
products = pd.read_csv('products.csv')
stores = pd.read_csv('stores.csv')
sales = pd.read_csv('sales.csv')

# Check columns in each DataFrame
print("Campaigns columns:", campaigns.columns)
print("Products columns:", products.columns)
print("Stores columns:", stores.columns)
print("Sales columns:", sales.columns)

# Ensure that all necessary columns exist before merging
if 'campaign_id' in sales.columns and 'campaign_id' in campaigns.columns:
    sales_campaigns = pd.merge(sales, campaigns, on='campaign_id', how='left')
else:
    print("Missing 'campaign_id' column in sales or campaigns data.")

if 'product_code' in sales_campaigns.columns and 'product_code' in products.columns:
    sales_campaigns_products = pd.merge(sales_campaigns, products, on='product_code', how='left')
else:
    print("Missing 'product_code' column in sales or products data.")

if 'store_id' in sales_campaigns_products.columns and 'store_id' in stores.columns:
    full_data = pd.merge(sales_campaigns_products, stores, on='store_id', how='left')
else:
    print("Missing 'store_id' column in sales or stores data.")

# Display the merged data to confirm structure
print("Merged data preview:")
print(full_data.head())

# Proceed with encoding and model training if merging was successful
if 'quantity_sold(after_promo)' in full_data.columns:
    # Encode categorical variables
    le_store = LabelEncoder()
    le_campaign = LabelEncoder()
    le_product = LabelEncoder()
    le_promo = LabelEncoder()

    full_data['store_id'] = le_store.fit_transform(full_data['store_id'])
    full_data['campaign_id'] = le_campaign.fit_transform(full_data['campaign_id'])
    full_data['product_code'] = le_product.fit_transform(full_data['product_code'])
    full_data['promo_type'] = le_promo.fit_transform(full_data['promo_type'])

    # Select features and target variable
    X = full_data[['store_id', 'campaign_id', 'product_code', 'base_price', 'promo_type', 'quantity_sold(before_promo)']]
    y = full_data['quantity_sold(after_promo)']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train KNN model
    knn = KNeighborsRegressor(n_neighbors=5)  # Set K to 5
    knn.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = knn.predict(X_test)
    accuracy = knn.score(X_test, y_test)  # R-squared accuracy
    mse = mean_squared_error(y_test, y_pred)

    # Display accuracy and mean squared error
    print(f"Model Accuracy (R-squared): {accuracy:.2f}")
    print(f"Mean Squared Error: {mse:.2f}")

    # Test with a new sample
    sample = [[1, 0, 2, 860, 1, 300]]  # Replace with encoded values according to your columns
    sample_df = pd.DataFrame(sample, columns=X.columns)  # Ensure column names match the training data

    # Scale and predict for the sample
    sample_scaled = scaler.transform(sample_df)
    predicted_sales = knn.predict(sample_scaled)
    print(f"Predicted sales for sample: {predicted_sales[0]:.2f}")
else:
    print("Failed to create 'quantity_sold(after_promo)' target variable.")