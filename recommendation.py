import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# ============================================
# Load Cleaned Dataset
# ============================================

df = pd.read_csv("outputs/cleaned_data.csv")

# Remove missing product names
df = df.dropna(subset=["Description"])

# ============================================
# Customer-Product Matrix
# ============================================

pivot = df.pivot_table(
    index="CustomerID",
    columns="Description",
    values="Quantity",
    aggfunc="sum",
    fill_value=0
)

# ============================================
# Cosine Similarity
# ============================================

similarity = cosine_similarity(pivot.T)

similarity_df = pd.DataFrame(
    similarity,
    index=pivot.columns,
    columns=pivot.columns
)

# ============================================
# Save only Top 10 Recommendations
# ============================================

top_recommendations = {}

for product in similarity_df.index:

    top_products = (
        similarity_df[product]
        .sort_values(ascending=False)
        .iloc[1:11]
    )

    top_recommendations[product] = top_products

# ============================================
# Save Optimized Model
# ============================================

joblib.dump(
    top_recommendations,
    "models/similarity.pkl",
    compress=3
)

print("Optimized Recommendation Model Created Successfully!")