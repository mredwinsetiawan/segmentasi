import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Dashboard layout
st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

# Sidebar (left panel)
st.sidebar.image("https://upload.wikimedia.org/wikipedia/id/thumb/2/2d/Undip.png/360px-Undip.png", use_column_width=True)  # Logo (adjust the link or use a local image)
st.sidebar.title("Customer Segmentation")

# Menu selection
menu = st.sidebar.selectbox("Menu", ["Home", "Hak Cipta"])


# Title and Introduction
st.title("Customer Segmentation App (RFM Clustering)")
st.write("""
    Upload a CSV or Excel file containing customer data. This application calculates RFM values and performs KMeans clustering on normalized data.
""")

# File Upload
uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx'])

if uploaded_file is not None:
    # Load data
    if uploaded_file.name.endswith('csv'):
        data = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('xlsx'):
        data = pd.read_excel(uploaded_file, engine='openpyxl')

    # Handling missing values
    st.subheader('Handling Missing Values')
    st.write("Original data shape:", data.shape)
    st.write("Number of missing values before handling:", data.isnull().sum().sum())
    
    # Drop rows with any NaN values
    data.dropna(inplace=True)

    # Verify data integrity post handling missing values
    st.write("Number of missing values after handling:", data.isnull().sum().sum())

    # Display Data
    if st.checkbox('Show raw data'):
        st.subheader('Raw data')
        st.write(data)

 
    # Perhitungan RFM
    st.subheader('RFM Calculation')

    # Pastikan ada kolom 'CustomerID', 'TransactionDate', dan 'MonetaryValue' yang sesuai
    if 'CustomerID' in data.columns and 'TransactionDate' in data.columns and 'MonetaryValue' in data.columns:
        # Convert 'TransactionDate' to datetime if it's not already
        data['TransactionDate'] = pd.to_datetime(data['TransactionDate'])

        # Current date for recency calculation
        current_date = data['TransactionDate'].max() + pd.DateOffset(1)

        # Group by CustomerID to calculate RFM
        rfm = data.groupby('CustomerID').agg({
            'TransactionDate': [
                lambda x: (current_date - x.max()).days,  # Recency
                lambda x: x.nunique()  # Frequency dihitung dari jumlah hari transaksi unik
            ],
            'MonetaryValue': 'sum'  # Monetary
        }).reset_index()

        # Flatten multi-index columns
        rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

        st.write("RFM Table:")
        st.write(rfm)



        # # Normalisasi nilai RFM
        # st.subheader('Normalisasi Nilai RFM')
        # scaler = StandardScaler()
        # rfm_normalized = rfm[['Recency', 'Frequency', 'Monetary']]
        # rfm_scaled = scaler.fit_transform(rfm_normalized)
        # rfm_scaled_df = pd.DataFrame(rfm_scaled, columns=['Recency', 'Frequency', 'Monetary'])

        # st.write("RFM Normalized:")
        # st.write(rfm_scaled_df)

        # Normalisasi Min-Max pada RFM
        st.subheader('Normalisasi Nilai RFM dengan Min-Max Normalization')

        # Normalisasi Min-Max untuk setiap kolom RFM
        rfm_normalized = rfm.copy()  # Salin data RFM untuk normalisasi

        # Menggunakan Min-Max Normalization untuk setiap kolom RFM
        rfm_normalized['Recency'] = (rfm['Recency'] - rfm['Recency'].min()) / (rfm['Recency'].max() - rfm['Recency'].min())
        rfm_normalized['Frequency'] = (rfm['Frequency'] - rfm['Frequency'].min()) / (rfm['Frequency'].max() - rfm['Frequency'].min())
        rfm_normalized['Monetary'] = (rfm['Monetary'] - rfm['Monetary'].min()) / (rfm['Monetary'].max() - rfm['Monetary'].min())

        st.write("RFM Normalized (Min-Max):")
        st.write(rfm_normalized)
        # RFM setelah normalisasi
        rfm_scaled = rfm_normalized[['Recency', 'Frequency', 'Monetary']].values

        # Elbow Method
        st.subheader('Elbow Method')
        sse = []
        silhouette_scores = []
        k_range = range(2, 11)
        for k in k_range:
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(rfm_scaled)
            sse.append(kmeans.inertia_)
            
            # Calculate silhouette score for each k
            score = silhouette_score(rfm_scaled, kmeans.labels_)
            silhouette_scores.append(score)

        # Plot Elbow Method
        fig, ax = plt.subplots()
        ax.plot(k_range, sse, marker='o')
        ax.set_xlabel('Number of clusters')
        ax.set_ylabel('SSE (Inertia)')
        ax.set_title('Elbow Method For Optimal k')
        st.pyplot(fig)

        # Plot Silhouette Scores
        st.subheader('Silhouette Scores for Different k')
        fig, ax = plt.subplots()
        ax.plot(k_range, silhouette_scores, marker='o')
        ax.set_xlabel('Number of clusters')
        ax.set_ylabel('Silhouette Score')
        ax.set_title('Silhouette Score for Each k')
        st.pyplot(fig)

        # Menentukan jumlah kluster terbaik
        optimal_k_silhouette = k_range[silhouette_scores.index(max(silhouette_scores))]

        st.write(f"Based on Silhouette Score, the optimal number of clusters is {optimal_k_silhouette} with a score of {max(silhouette_scores):.4f}.")

        # Clustering with chosen number of clusters
        st.subheader('Clustering')
        num_clusters = st.slider('Select number of clusters', 2, 10, optimal_k_silhouette)
        kmeans = KMeans(n_clusters=num_clusters)
        rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

        # Silhouette Score for selected number of clusters
        st.write(f"Silhouette Score for {num_clusters} clusters: {silhouette_score(rfm_scaled, rfm['Cluster']):.4f}")

        # Segmentasi Pelanggan Berdasarkan Cluster
        st.subheader('Customer Segmentation by Cluster')
        
        # Defining customer segment names
        segment_names = {
            0: 'Loyal VIP',
            1: 'Frequent Buyers',
            2: 'High Spenders',
            3: 'Potential Loyalists',
            4: 'New Customers',
            5: 'Occasional Buyers',
            6: 'At-Risk Customers',
            7: 'Lost Customers',
            8: 'Bargain Hunters',
            9: 'Window Shoppers'
        }

        # Assign segment names based on clusters
        rfm['CustomerSegment'] = rfm['Cluster'].map(segment_names)

        # Display customer segments
        for cluster in range(num_clusters):
            st.write(f"### Cluster {cluster+1}: {segment_names.get(cluster, 'Unknown Segment')}")
            cluster_data = rfm[rfm['Cluster'] == cluster]
            st.write(cluster_data.describe())

            # Treatment untuk setiap Cluster
            if cluster == 0:
                st.write("**Treatment**: VIP customers - Offer personalized rewards, loyalty programs, and premium customer service.")
            elif cluster == 1:
                st.write("**Treatment**: Frequent Buyers - Offer loyalty rewards and targeted promotions to increase frequency.")
            elif cluster == 2:
                st.write("**Treatment**: High Spenders - Upsell premium products and offer exclusive experiences.")
            elif cluster == 3:
                st.write("**Treatment**: Potential Loyalists - Strengthen engagement through loyalty programs and personalized offers.")
            elif cluster == 4:
                st.write("**Treatment**: New Customers - Provide welcome offers and guide them through product discovery.")
            elif cluster == 5:
                st.write("**Treatment**: Occasional Buyers - Re-engage with discounts or tailored recommendations.")
            elif cluster == 6:
                st.write("**Treatment**: At-Risk Customers - Re-engage with personalized offers and win-back campaigns.")
            elif cluster == 7:
                st.write("**Treatment**: Lost Customers - Offer significant discounts to win them back.")
            elif cluster == 8:
                st.write("**Treatment**: Bargain Hunters - Highlight promotions and special offers.")
            elif cluster == 9:
                st.write("**Treatment**: Window Shoppers - Incentivize with first-time discounts and personalized recommendations.")


        # Visualize Clusters
        st.subheader('Cluster Visualization')
        fig, ax = plt.subplots()
        sns.scatterplot(x=rfm['Recency'], y=rfm['Monetary'], hue=rfm['Cluster'], palette='viridis', ax=ax)
        ax.set_title('RFM Clustering Visualization')
        st.pyplot(fig)

        # Kesimpulan
        st.subheader('Kesimpulan')
        # Tabel Segmentasi Pelanggan Berdasarkan Cluster
        st.subheader('Customer Segmentation by Cluster')

        # Prepare a dataframe to show cluster info
        cluster_info = pd.DataFrame({
            'Cluster': [f'Cluster {i+1}' for i in range(num_clusters)],
            'Segment Name': [segment_names.get(i, 'Unknown Segment') for i in range(num_clusters)],
            'Customer Count': [rfm[rfm['Cluster'] == i].shape[0] for i in range(num_clusters)],
            'Treatment': [
                "VIP customers - Offer personalized rewards, loyalty programs, and premium customer service.",
                "Frequent Buyers - Offer loyalty rewards and targeted promotions to increase frequency.",
                "High Spenders - Upsell premium products and offer exclusive experiences.",
                "Potential Loyalists - Strengthen engagement through loyalty programs and personalized offers.",
                "New Customers - Provide welcome offers and guide them through product discovery.",
                "Occasional Buyers - Re-engage with discounts or tailored recommendations.",
                "At-Risk Customers - Re-engage with personalized offers and win-back campaigns.",
                "Lost Customers - Offer significant discounts to win them back.",
                "Bargain Hunters - Highlight promotions and special offers.",
                "Window Shoppers - Incentivize with first-time discounts and personalized recommendations."
            ][:num_clusters]  # Limit the treatments based on number of clusters
        })

        # Display the cluster info table
        st.write(cluster_info)


        # # Cluster Summary
        # st.write("Cluster Summary:")
        # st.dataframe(rfm)


        st.write("""
            Based on the clustering results and Silhouette Score, we can observe that the customers have been successfully segmented into distinct groups.
            Each group has different characteristics based on the RFM values. These insights can be used for targeted marketing, 
            improving customer retention, and personalizing customer experiences.
        """)
    else:
        st.write("Make sure your dataset contains the necessary columns: 'CustomerID', 'TransactionDate', 'MonetaryValue', and 'Frequency'.")

    # Footer
    st.markdown("---")
    st.write("© 2024 [Edwin Setiawan]. All rights reserved.")
