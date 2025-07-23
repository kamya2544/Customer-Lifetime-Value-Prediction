import pandas as pd
import datetime as dt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
DATA_FILE = 'Online Retail.xlsx'
# Date to consider as "today" for recency/T calculation.
# We'll use the day after the last transaction in the dataset for a realistic "today".
# This will be dynamically calculated.
PREDICTION_PERIOD_MONTHS = 6 # Predict CLV for the next 6 months

# --- 1. Data Loading and Initial Preprocessing ---
print("--- 1. Loading and Preprocessing Data ---")
try:
    df = pd.read_excel(DATA_FILE)
    print(f"Successfully loaded '{DATA_FILE}'. Initial rows: {len(df)}")
except FileNotFoundError:
    print(f"Error: '{DATA_FILE}' not found. Please ensure the dataset is in the same directory.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the data: {e}")
    exit()

# Rename columns for easier access 
df.columns = df.columns.str.strip() # Remove any leading/trailing whitespace from column names
df.rename(columns={'InvoiceNo': 'invoice',
                   'StockCode': 'stock_code',
                   'Description': 'description',
                   'Quantity': 'quantity',
                   'InvoiceDate': 'invoice_date',
                   'UnitPrice': 'unit_price',
                   'CustomerID': 'customer_id',
                   'Country': 'country'}, inplace=True)

# Drop rows with missing CustomerID, as it's essential for CLV analysis
df.dropna(subset=['customer_id'], inplace=True)
df['customer_id'] = df['customer_id'].astype(int) # Convert CustomerID to integer

# Remove cancelled orders (InvoiceNo starting with 'C')
df = df[~df['invoice'].astype(str).str.contains('C', na=False)]

# Remove rows where Quantity or UnitPrice are less than or equal to 0
# These are returns or data errors and not valid purchases for CLV
df = df[df['quantity'] > 0]
df = df[df['unit_price'] > 0]

# Calculate TotalPrice for each transaction line
df['total_price'] = df['quantity'] * df['unit_price']

# Convert InvoiceDate to datetime objects
df['invoice_date'] = pd.to_datetime(df['invoice_date'])

print(f"Data after cleaning: {len(df)} rows.")

# --- 2. RFM Feature Engineering ---
print("\n--- 2. Performing RFM Feature Engineering ---")

# Determine the observation period end date (the day after the last transaction)
# This is crucial for calculating 'T' (customer's age) correctly.
observation_period_end = df['invoice_date'].max() + dt.timedelta(days=1)
print(f"Observation period ends on: {observation_period_end.strftime('%Y-%m-%d')}")

# Calculate RFM (Recency, Frequency, Monetary) summary data
# 'frequency': number of repeat purchases (transactions after the first)
# 'recency': age of the customer when they made their last purchase (duration between first and last purchase)
# 'T': age of the customer in the dataset (duration between first purchase and observation_period_end)
# 'monetary_value': average value of a customer's transactions (excluding the first purchase)
rfm_df = summary_data_from_transaction_data(
    transactions=df,
    customer_id_col='customer_id',
    datetime_col='invoice_date',
    monetary_value_col='total_price',
    observation_period_end=observation_period_end # Use the calculated end date
)

print("\nRFM Summary Data Head:")
print(rfm_df.head())
print(f"\nTotal unique customers for RFM analysis: {len(rfm_df)}")

# Filter out customers with frequency <= 0 for Gamma-Gamma model, as it applies to repeat buyers
# For BG/NBD, customers with frequency=0 are still valuable as they might purchase in the future.
rfm_df_repeat_customers = rfm_df[rfm_df['frequency'] > 0]
print(f"Customers with repeat purchases (for Gamma-Gamma model): {len(rfm_df_repeat_customers)}")

# Optional:--- Visualize RFM Distributions ---
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.histplot(rfm_df['frequency'], bins=20, kde=True)
plt.title('Frequency Distribution')
plt.subplot(1, 3, 2)
sns.histplot(rfm_df['recency'], bins=20, kde=True)
plt.title('Recency Distribution')
plt.subplot(1, 3, 3)
sns.histplot(rfm_df['T'], bins=20, kde=True)
plt.title('T (Customer Age) Distribution')
plt.tight_layout()
plt.show()


# --- 3. Probabilistic CLV Modeling ---
print("\n--- 3. Training Probabilistic CLV Models ---")

# --- 3.1. BG/NBD Model (Frequency and Churn Prediction) ---
print("\nFitting BG/NBD Model...")
bgf = BetaGeoFitter(penalizer_coef=0.1) # Added a small penalizer for regularization
bgf.fit(rfm_df['frequency'], rfm_df['recency'], rfm_df['T'])

print("\nBG/NBD Model Summary:")
print(bgf.summary)

# Plotting expected purchases for diagnostic
# plot_period_transactions(bgf)
# plt.show()

# Predict future purchases for the next `PREDICTION_PERIOD_MONTHS` months
# Convert days to months for prediction
days_in_prediction_period = PREDICTION_PERIOD_MONTHS * 30.4375 # Average days in a month
rfm_df['predicted_purchases'] = bgf.predict(
    days_in_prediction_period,
    rfm_df['frequency'],
    rfm_df['recency'],
    rfm_df['T']
)
print(f"\nPredicted purchases for the next {PREDICTION_PERIOD_MONTHS} months calculated.")

# --- 3.2. Gamma-Gamma Model (Monetary Value Prediction) ---
print("\nFitting Gamma-Gamma Model...")
ggf = GammaGammaFitter(penalizer_coef=0.1) # Added a small penalizer for regularization

# Fit Gamma-Gamma model only on customers who have made repeat purchases (frequency > 0)
# The Gamma-Gamma model assumes that the monetary value is independent of the transaction frequency.
# It's fitted on the average monetary value of repeat purchases.
ggf.fit(rfm_df_repeat_customers['frequency'],
        rfm_df_repeat_customers['monetary_value'])

print("\nGamma-Gamma Model Summary:")
print(ggf.summary)

# --- 3.3. Calculate Customer Lifetime Value (CLV) ---
print(f"\nCalculating CLV for the next {PREDICTION_PERIOD_MONTHS} months...")

# Calculate the conditional expected average profit for customers with repeat purchases
# This is where the Gamma-Gamma model predicts the average monetary value of future transactions.
rfm_df['predicted_monetary_value'] = 0.0 # Initialize with 0
# Merge predicted monetary values back to the main rfm_df
# We use the index (customer_id) to align the data
merged_monetary = ggf.conditional_expected_average_profit(
    rfm_df_repeat_customers['frequency'],
    rfm_df_repeat_customers['monetary_value']
)
rfm_df.loc[rfm_df['frequency'] > 0, 'predicted_monetary_value'] = merged_monetary.values

# Calculate CLV by multiplying predicted purchases by predicted monetary value
# For customers with no repeat purchases (frequency=0), their predicted_monetary_value will remain 0,
# resulting in a CLV of 0, which is appropriate as Gamma-Gamma doesn't apply to them directly.
rfm_df['predicted_clv'] = rfm_df['predicted_purchases'] * rfm_df['predicted_monetary_value']

print("\n--- CLV Prediction Complete ---")

# Display top customers by predicted CLV
print(f"\nTop 10 Customers by Predicted CLV (next {PREDICTION_PERIOD_MONTHS} months):")
top_clv_customers = rfm_df.sort_values(by='predicted_clv', ascending=False).head(10)
print(top_clv_customers[['predicted_purchases', 'predicted_monetary_value', 'predicted_clv']])

# Save results to a CSV file
output_filename = 'clv_predictions.csv'
rfm_df[['predicted_clv']].to_csv(output_filename, index=True) # index=True saves customer_id
print(f"\nPredicted CLV for all customers saved to '{output_filename}'")

print("\n--- Script Finished ---")
