import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("outputs/cleaned_data.csv")

# Convert InvoiceDate
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Latest date in dataset
snapshot_date = df['InvoiceDate'].max()

# Create RFM table
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalAmount': 'sum'
})

# Rename columns
rfm.columns = ['Recency', 'Frequency', 'Monetary']

# Save RFM data
rfm.to_csv("outputs/rfm.csv")

print(rfm.head())

# -------------------
# Recency Distribution
# -------------------
plt.figure(figsize=(8,5))
sns.histplot(rfm['Recency'], bins=30)
plt.title("Recency Distribution")
plt.savefig("outputs/recency_distribution.png")
plt.show()

# -------------------
# Frequency Distribution
# -------------------
plt.figure(figsize=(8,5))
sns.histplot(rfm['Frequency'], bins=30)
plt.title("Frequency Distribution")
plt.savefig("outputs/frequency_distribution.png")
plt.show()

# -------------------
# Monetary Distribution
# -------------------
plt.figure(figsize=(8,5))
sns.histplot(rfm['Monetary'], bins=30)
plt.title("Monetary Distribution")
plt.savefig("outputs/monetary_distribution.png")
plt.show()

print("RFM Analysis Completed Successfully!")