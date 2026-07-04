import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide"
)

# -----------------------
# LOAD DATA
# -----------------------
df = pd.read_csv("outputs/cleaned_data.csv")
clusters = pd.read_csv("outputs/customer_clusters.csv")

kmeans = joblib.load("models/kmeans.pkl")
scaler = joblib.load("models/scaler.pkl")

# Recommendation Dictionary
similarity = joblib.load("models/similarity.pkl")

# Sample Similarity Matrix (30×30)
similarity_matrix = joblib.load("models/similarity_matrix.pkl")

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

df["Month"] = df["InvoiceDate"].dt.to_period("M").astype(str)

# -----------------------
# SIDEBAR
# -----------------------
st.sidebar.title("🛒 Shopper Spectrum")
country_filter = st.sidebar.selectbox(
    "Select Country",
    ["All"] + sorted(df["Country"].unique())
)

if country_filter != "All":
    df = df[df["Country"] == country_filter]

page = st.sidebar.radio(
    "Navigation",
    [
        "Executive Dashboard",
        "Sales Analytics",
        "Country Analysis",
        "RFM Analysis",
        "Elbow Method",
        "Customer Segmentation",
        "Similarity Matrix",
        "Product Recommendation",
        "Customer Prediction",
        "Business Insights"
    ]
)
if page == "Executive Dashboard":

    st.title("📊 Executive Dashboard")

    revenue = round(df["TotalAmount"].sum(), 2)
    customers = df["CustomerID"].nunique()
    orders = df["InvoiceNo"].nunique()
    products = df["Description"].nunique()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Revenue", f"₹ {revenue:,.0f}")
    c2.metric("Customers", customers)
    c3.metric("Orders", orders)
    c4.metric("Products", products)
    st.metric(
    "Silhouette Score",
    "0.616"
)

    st.divider()
    st.subheader("📌 Project Overview")

    st.write("""
    ✔ Customer Segmentation

    ✔ Product Recommendation

    ✔ RFM Analysis

    ✔ Customer Prediction

    ✔ Business Insights

    ✔ Sales Analytics
    """)

    col1, col2 = st.columns(2)

    country = (
        df["Country"]
        .value_counts()
        .head(10)
    )

    fig1 = px.bar(
        x=country.values,
        y=country.index,
        orientation="h",
        title="Top Countries"
    )

    col1.plotly_chart(
        fig1,
        use_container_width=True
    )

    cluster_count = (
        clusters["Cluster"]
        .value_counts()
    )

    fig2 = px.pie(
        values=cluster_count.values,
        names=cluster_count.index,
        title="Customer Segments"
    )

    col2.plotly_chart(
        fig2,
        use_container_width=True
    )
elif page == "Sales Analytics":

    st.title("📈 Sales Analytics")

    monthly_sales = (
        df.groupby("Month")["TotalAmount"]
        .sum()
        .reset_index()
    )

    fig = px.line(
        monthly_sales,
        x="Month",
        y="TotalAmount",
        markers=True,
        title="Monthly Sales Trend"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Top Products")

    top_products = (
        df["Description"]
        .value_counts()
        .head(10)
    )

    fig2 = px.bar(
        x=top_products.values,
        y=top_products.index,
        orientation="h"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )
elif page == "Country Analysis":

    st.title("🌍 Country Analysis")

    country_sales = (
        df.groupby("Country")["TotalAmount"]
        .sum()
        .sort_values(ascending=False)
        .head(15)
    )

    fig = px.bar(
        x=country_sales.index,
        y=country_sales.values,
        title="Revenue by Country"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
elif page == "RFM Analysis":

    st.title("🧠 RFM Analysis")

    col1, col2, col3 = st.columns(3)

    fig1 = px.histogram(
        clusters,
        x="Recency",
        title="Recency"
    )

    col1.plotly_chart(fig1)

    fig2 = px.histogram(
        clusters,
        x="Frequency",
        title="Frequency"
    )

    col2.plotly_chart(fig2)

    fig3 = px.histogram(
        clusters,
        x="Monetary",
        title="Monetary"
    )

    col3.plotly_chart(fig3)

    # =====================================================
# ELBOW METHOD
# =====================================================

elif page == "Elbow Method":

    st.title("📈 Elbow Method")

    st.image(
        "outputs/elbow_curve.png",
        caption="Optimal Number of Clusters"
    )

    st.info(
        "The elbow method was used to determine the optimal number of customer clusters."
    )
    # =====================================================
# CUSTOMER SEGMENTATION
# =====================================================

elif page == "Customer Segmentation":

    st.title("👥 Customer Segmentation")

    col1, col2 = st.columns(2)

    fig1 = px.scatter(
        clusters,
        x="Recency",
        y="Monetary",
        color="Cluster",
        size="Frequency",
        title="Customer Clusters"
    )

    col1.plotly_chart(
        fig1,
        use_container_width=True
    )

    fig2 = px.scatter_3d(
        clusters,
        x="Recency",
        y="Frequency",
        z="Monetary",
        color="Cluster",
        title="3D Customer Segmentation"
    )

    col2.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.subheader("Cluster Profile")

    profile = (
        clusters.groupby("Cluster")
        [["Recency", "Frequency", "Monetary"]]
        .mean()
        .round(2)
    )
# =====================================================
# CUSTOMER SEGMENTATION
# =====================================================
# =====================================================
# CUSTOMER SEGMENTATION
# =====================================================

elif page == "Customer Segmentation":

    st.title("👥 Customer Segmentation")

    col1, col2 = st.columns(2)

    fig1 = px.scatter(
        clusters,
        x="Recency",
        y="Monetary",
        color="Cluster",
        size="Frequency",
        title="Customer Clusters"
    )

    col1.plotly_chart(
        fig1,
        use_container_width=True
    )

    fig2 = px.scatter_3d(
        clusters,
        x="Recency",
        y="Frequency",
        z="Monetary",
        color="Cluster",
        title="3D Customer Segmentation"
    )

    col2.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.subheader("Cluster Profile")

    profile = (
        clusters.groupby("Cluster")
        [["Recency", "Frequency", "Monetary"]]
        .mean()
        .round(2)
    )

    st.dataframe(
        profile,
        use_container_width=True
    )

    st.subheader("📋 Cluster Interpretation")

    cluster_info = pd.DataFrame({
        "Cluster": [0, 1, 2, 3],
        "Customer Segment": [
            "🏆 High Value Customer",
            "😊 Regular Customer",
            "🛍 Occasional Customer",
            "⚠ At Risk Customer"
        ],
        "Characteristics": [
            "High spending and frequent purchases",
            "Moderate spending and regular purchases",
            "Low purchase frequency",
            "Long inactivity period"
        ],
        "Business Strategy": [
            "Reward loyalty",
            "Offer promotions",
            "Increase engagement",
            "Retention campaigns"
        ]
    })

    st.dataframe(
        cluster_info,
        use_container_width=True
    )

    st.download_button(
        label="📥 Download Customer Cluster Data",
        data=clusters.to_csv(index=False),
        file_name="customer_clusters.csv",
        mime="text/csv"
    )

    st.info("""
🏆 High Value Customers contribute maximum revenue.

😊 Regular Customers maintain stable sales.

🛍 Occasional Customers can become regular buyers.

⚠ At Risk Customers need retention campaigns.
""")
# =====================================================
# SIMILARITY MATRIX
# =====================================================

# =====================================================
# SIMILARITY MATRIX
# =====================================================

elif page == "Similarity Matrix":

    st.title("🔥 Product Similarity Matrix")

    st.markdown("""
    This heatmap visualizes the cosine similarity between a sample of products.

    Darker colors indicate lower similarity, while brighter colors represent highly similar products.
    """)

    fig = px.imshow(
        similarity_matrix,
        color_continuous_scale="Viridis",
        title="Sample Product Similarity Heatmap (30 × 30)"
    )

    fig.update_layout(
        height=700,
        xaxis_title="Products",
        yaxis_title="Products"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.info("""
📌 **Note**

To optimize deployment and reduce model size, the dashboard displays a representative 30×30 sample of the product similarity matrix.

The recommendation engine still uses the optimized recommendation model to generate accurate product suggestions.
""")
    
    # =====================================================
# PRODUCT RECOMMENDATION
# =====================================================

elif page == "Product Recommendation":

    st.title("📦 Product Recommendation System")

    st.write(
        "Enter a product name to find similar products."
    )

    product = st.text_input("Product Name")

    if st.button("Recommend Products"):

        recommendations = similarity.get(product)

        if recommendations is None:

            st.error("Product not found.")

        else:

            st.success("Recommended Products")

            for i, item in enumerate(recommendations.index, 1):

                st.markdown(
                    f"""
                    <div style="
                    background-color:#1E293B;
                    padding:15px;
                    border-radius:10px;
                    margin-bottom:10px;
                    border-left:5px solid #2563EB;
                    color:white;
                    ">
                    <h4>✅ Recommendation {i}</h4>
                    <p>{item}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        st.success(
                "Recommended Products"
            )

        for i, item in enumerate(
                recommendations.index,
                1
            ):

                st.markdown(
                    f"""
                    <div style="
                    background-color:#1E293B;
                    padding:15px;
                    border-radius:10px;
                    margin-bottom:10px;
                    border-left:5px solid #2563EB;
                    color:white;
                    ">
                    <h4>✅ Recommendation {i}</h4>
                    <p>{item}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        else:

            st.error(
                "Product not found."
            )
            # =====================================================
# CUSTOMER PREDICTION
# =====================================================

elif page == "Customer Prediction":

    st.title("🤖 Customer Prediction")

    st.write(
        "Enter RFM values to predict customer type."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        recency = st.number_input(
            "Recency",
            min_value=0
        )

    with col2:
        frequency = st.number_input(
            "Frequency",
            min_value=0
        )

    with col3:
        monetary = st.number_input(
            "Monetary",
            min_value=0.0
        )

    if st.button(
        "Predict Customer"
    ):

        values = [[
            recency,
            frequency,
            monetary
        ]]

        scaled = scaler.transform(
            values
        )

        cluster = (
            kmeans.predict(
                scaled
            )[0]
        )

        labels = {
            0: "🏆 High Value Customer",
            1: "😊 Regular Customer",
            2: "🛍 Occasional Customer",
            3: "⚠ At Risk Customer"
        }

        st.success(
            labels[cluster]
        )
        # =====================================================
# BUSINESS INSIGHTS
# =====================================================

elif page == "Business Insights":

    st.title("📋 Business Insights")

    revenue = round(
        clusters["Monetary"].sum(),
        2
    )

    avg_spend = round(
        clusters["Monetary"].mean(),
        2
    )
    col1, col2 = st.columns(2)

    col1.metric(
    "Average Customer Spending",
    f"₹ {avg_spend:.0f}"
)

    col2.metric(
    "Total Revenue",
    f"₹ {revenue:,.0f}"
)

    top_country = (
        df["Country"]
        .value_counts()
        .idxmax()
    )

    top_product = (
        df["Description"]
        .value_counts()
        .idxmax()
    )

    st.success(f"""
📈 Total Revenue: ₹ {revenue:,.0f}

🌍 Top Country: {top_country}

🛒 Best Selling Product:
{top_product}

🏆 High-value customers contribute maximum revenue.

⚠ At-risk customers need retention campaigns.

💡 Regular customers can be converted into premium customers.
""")
    st.markdown("---")

st.markdown(
    """
    <center>
    <h5>Shopper Spectrum Dashboard</h5>
    <p>Customer Segmentation & Recommendation System</p>
    </center>
    """,
    unsafe_allow_html=True
)