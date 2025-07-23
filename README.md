# Customer Lifetime Value (CLV) Prediction for Online Retail
## Project Overview
This project focuses on predicting the Customer Lifetime Value (CLV) for an online retail business using historical transactional data. Customer Lifetime Value is a crucial metric that estimates the total revenue a business can reasonably expect from a customer throughout their relationship. By understanding CLV, businesses can optimize marketing spend, improve customer retention strategies, and make data-driven decisions to maximize long-term profitability.

This implementation utilizes probabilistic models, specifically the Beta-Geometric/Negative Binomial Distribution (BG/NBD) model for predicting customer purchasing behavior (frequency and churn) and the Gamma-Gamma model for estimating the average monetary value of transactions.

## Dataset
The dataset used in this project is the "Online Retail Dataset" available on Kaggle:
https://www.kaggle.com/datasets/lakshmi25npathi/online-retail-dataset

Please download the Online Retail.xlsx file from the link above and place it in the root directory of this project.

### Dataset Columns:
InvoiceNo: Invoice number. A 6-digit integral number uniquely assigned to each transaction. If this code starts with the letter 'c', it indicates a cancellation.

#### StockCode: Product (item) code. A 5-digit integral number uniquely assigned to each distinct product.

#### Description: Product (item) name.

#### Quantity: The quantities of each product (item) per transaction.

#### InvoiceDate: Invoice date and time. The day and time when a transaction was generated.

#### UnitPrice: Unit price. Product price per unit in sterling (£).

#### CustomerID: Customer number. A 5-digit integral number uniquely assigned to each customer.

#### Country: Country name. The name of the country where each customer resides.

## Methodology
The project follows these key steps:

### Data Loading and Preprocessing:

Load the Online Retail.xlsx dataset.

Handle missing CustomerID values by removing them, as they are essential for customer-level analysis.

Filter out cancelled orders (InvoiceNo starting with 'C').

Remove transactions with negative or zero Quantity or UnitPrice, as these are typically data errors or returns that don't represent valid purchases for CLV calculation.

Calculate TotalPrice for each transaction (Quantity * UnitPrice).

### Feature Engineering (RFM Analysis):

Recency (R): The number of days between a customer's first and last purchase.

Frequency (F): The number of unique purchase occasions (transactions) a customer has made after their first purchase.

Monetary (M): The average monetary value of a customer's transactions.

T (Customer's Age): The number of days between a customer's first purchase and the end of the observation period.

These features are derived from the raw transactional data using the lifetimes library's summary_data_from_transaction_data function.

### Probabilistic CLV Modeling:

BG/NBD (Beta-Geometric/Negative Binomial Distribution) Model: This model is fitted to the frequency, recency, and T data to predict the expected number of future transactions for each customer. It accounts for the probability of a customer being "alive" (still active) and their purchasing rate.

Gamma-Gamma Model: This model is fitted to the frequency and monetary_value data (for customers with at least one repeat purchase) to predict the average monetary value of a customer's future transactions. It assumes that the monetary value of a customer's transactions is independent of their purchase frequency.

The predictions from both models are then combined to calculate the overall CLV for a specified future period (e.g., 6 months).

## Installation
To run this project, you need Python installed. Then, install the required libraries using pip:

pip install pandas openpyxl lifetimes matplotlib seaborn

pandas: For data manipulation and analysis.

openpyxl: To read .xlsx files.

lifetimes: A powerful library for CLV modeling using probabilistic models.

matplotlib & seaborn: For data visualization (optional, but good for understanding model fit).

## Usage
### Download the Dataset:

Go to https://www.kaggle.com/datasets/lakshmi25npathi/online-retail-dataset.

Download the Online Retail.xlsx file.

Place Online Retail.xlsx in the same directory as clv_prediction.py.

### Run the Python Script:
Open your terminal or command prompt, navigate to the project directory, and run the script:

python clv_prediction.py

### Expected Output
The script will print:

Summary statistics of the RFM data.

Information about the fitted BG/NBD and Gamma-Gamma models.

A table showing the top 10 customers by predicted CLV for the next 6 months.

A CSV file named clv_predictions.csv containing the CustomerID and their predicted CLV.

## Future Enhancements
More Sophisticated Feature Engineering: Incorporate features like product categories purchased, time of day/week of purchases, or seasonality.

Demographic Data: If available, integrate actual customer demographic data (age, gender, income) to enhance predictions.

Different CLV Models: Experiment with other CLV models, such as machine learning regression models (e.g., Random Forest, XGBoost) if a long enough historical period is available to define a target CLV for training.

Cohort Analysis: Perform cohort analysis to understand how different customer groups behave over time.

Model Deployment: Deploy the trained model as an API for real-time CLV prediction.

Interactive Dashboard: Create an interactive dashboard (e.g., using Dash or Streamlit) to visualize CLV predictions and customer segments.
