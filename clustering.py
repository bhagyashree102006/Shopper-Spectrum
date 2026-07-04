import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load RFM data
rfm = pd.read_csv("outputs/rfm.csv")

X = rfm[['Recency', 'Frequency', 'Monetary']]

# Standardization
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Elbow Method
wcss = []

for i in range(2, 11):
    kmeans = KMeans(n_clusters=i,
                    random_state=42,
                    n_init=10)

    kmeans.fit(X_scaled)

    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(2,11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Clusters")
plt.ylabel("WCSS")
plt.savefig("outputs/elbow_curve.png")
plt.show()

# Final Model
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

rfm['Cluster'] = kmeans.fit_predict(X_scaled)

# Save cluster data
rfm.to_csv(
    "outputs/customer_clusters.csv",
    index=False
)

# Save models
joblib.dump(kmeans, "models/kmeans.pkl")
joblib.dump(scaler, "models/scaler.pkl")

# Silhouette Score
score = silhouette_score(
    X_scaled,
    rfm['Cluster']
)

print("Silhouette Score:", score)

# Cluster Scatter Plot
plt.figure(figsize=(8,5))
plt.scatter(
    rfm['Recency'],
    rfm['Monetary'],
    c=rfm['Cluster']
)

plt.xlabel("Recency")
plt.ylabel("Monetary")
plt.title("Customer Clusters")

plt.savefig("outputs/clusters.png")
plt.show()

print("Clustering Completed Successfully!")