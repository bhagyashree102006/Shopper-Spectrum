import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned data
df = pd.read_csv("outputs/cleaned_data.csv")

# Convert date column
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# --------------------------
# Graph 1: Top Countries
# --------------------------
plt.figure(figsize=(10,5))
df['Country'].value_counts().head(10).plot(kind='bar')
plt.title("Top 10 Countries")
plt.xlabel("Country")
plt.ylabel("Transactions")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/top_countries.png")
plt.show()

# --------------------------
# Graph 2: Top Products
# --------------------------
plt.figure(figsize=(10,5))
df['Description'].value_counts().head(10).plot(kind='bar')
plt.title("Top 10 Products")
plt.xlabel("Product")
plt.ylabel("Sales")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("outputs/top_products.png")
plt.show()

# --------------------------
# Graph 3: Sales Distribution
# --------------------------
plt.figure(figsize=(8,5))
sns.histplot(df['TotalAmount'], bins=50)
plt.title("Sales Distribution")
plt.tight_layout()
plt.savefig("outputs/sales_distribution.png")
plt.show()

# --------------------------
# Graph 4: Quantity Distribution
# --------------------------
plt.figure(figsize=(8,5))
sns.boxplot(x=df['Quantity'])
plt.title("Quantity Distribution")
plt.tight_layout()
plt.savefig("outputs/quantity_distribution.png")
plt.show()

# --------------------------
# Graph 5: Monthly Sales
# --------------------------
df['Month'] = df['InvoiceDate'].dt.month

monthly_sales = df.groupby('Month')['TotalAmount'].sum()

plt.figure(figsize=(8,5))
monthly_sales.plot(marker='o')
plt.title("Monthly Sales")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.grid()
plt.tight_layout()
plt.savefig("outputs/monthly_sales.png")
plt.show()

print("EDA Completed Successfully!")