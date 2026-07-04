import pandas as pd

# Load dataset
df = pd.read_csv("data/online_retail.csv")

# Dataset information
print("Dataset Shape:", df.shape)
print(df.head())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Remove missing CustomerID
df = df.dropna(subset=['CustomerID'])

# Remove cancelled invoices
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]

# Remove negative quantities
df = df[df['Quantity'] > 0]

# Remove zero or negative prices
df = df[df['UnitPrice'] > 0]

# Convert InvoiceDate into datetime format
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Create TotalAmount column
df['TotalAmount'] = df['Quantity'] * df['UnitPrice']

# Save cleaned dataset
df.to_csv("outputs/cleaned_data.csv", index=False)

print("\nData Cleaning Completed Successfully!")
print("Final Shape:", df.shape)