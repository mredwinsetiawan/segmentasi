import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Dashboard layout
st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

# Sidebar (left panel)
st.sidebar.image("https://upload.wikimedia.org/wikipedia/id/thumb/2/2d/Undip.png/360px-Undip.png", use_column_width=True)
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
    data.dropna(inplace=True)
    st.write("Number of missing values after handling:", data.isnull().sum().sum())

    # Display Data
    if st.checkbox('Show raw data'):
        st.subheader('Raw data')
        st.write(data)

    # Perhitungan RFM
    st.subheader('RFM Calculation')
    if 'CustomerID' in data.columns and 'TransactionDate' in data.columns and 'MonetaryValue' in data.columns:
        data['TransactionDate'] = pd.to_datetime(data['TransactionDate'], errors='coerce')
        data = data.dropna(subset=['TransactionDate'])

        current_date = data['TransactionDate'].max() + pd.DateOffset(1)
        rfm = data.groupby('CustomerID').agg({
            'TransactionDate': [
                lambda x: (current_date - x.max()).days,  # Recency
                lambda x: x.nunique()  # Frequency dihitung dari jumlah hari transaksi unik
            ],
            'MonetaryValue': 'sum'  # Monetary
        }).reset_index()

        rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
        st.write("RFM Table:")
        st.write(rfm)

        rfm['Recency'] = pd.to_numeric(rfm['Recency'], errors='coerce')
        rfm['Frequency'] = pd.to_numeric(rfm['Frequency'], errors='coerce')
        rfm['Monetary'] = pd.to_numeric(rfm['Monetary'], errors='coerce')
        rfm.dropna(inplace=True)

        # Normalisasi Min-Max pada RFM
        st.subheader('Normalisasi Nilai RFM dengan Min-Max Normalization')
        scaler = MinMaxScaler()
        rfm[['Recency', 'Frequency', 'Monetary']] = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
        st.write("RFM Normalized (Min-Max):")
        st.write(rfm)

        # Elbow Method
        st.subheader('Elbow Method')
        sse = []
        silhouette_scores = []
        k_range = range(2, 11)
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=0)
            kmeans.fit(rfm[['Recency', 'Frequency', 'Monetary']])
            sse.append(kmeans.inertia_)
            if len(set(kmeans.labels_)) > 1:  # Calculate silhouette only if more than one cluster
                silhouette_scores.append(silhouette_score(rfm[['Recency', 'Frequency', 'Monetary']], kmeans.labels_))
            else:
                silhouette_scores.append(None)  # Skip silhouette score if only one cluster

        fig, ax = plt.subplots()
        ax.plot(k_range, sse, marker='o')
        ax.set_xlabel('Number of clusters')
        ax.set_ylabel('SSE (Inertia)')
        ax.set_title('Elbow Method For Optimal k')
        st.pyplot(fig)

        fig, ax = plt.subplots()
        valid_silhouette_scores = [score for score in silhouette_scores if score is not None]
        valid_k_range = [k for k, score in zip(k_range, silhouette_scores) if score is not None]
        ax.plot(valid_k_range, valid_silhouette_scores, marker='o')
        ax.set_xlabel('Number of clusters')
        ax.set_ylabel('Silhouette Score')
        ax.set_title('Silhouette Score for Each k')
        st.pyplot(fig)

        if valid_silhouette_scores:
            optimal_k = valid_k_range[valid_silhouette_scores.index(max(valid_silhouette_scores))]
            st.write(f"Optimal number of clusters (Silhouette Score): {optimal_k} with score {max(valid_silhouette_scores):.4f}")
        else:
            st.write("No valid Silhouette Score available due to single cluster in some configurations.")

        # Clustering with chosen number of clusters
        st.subheader('Clustering')
        num_clusters = st.slider('Select number of clusters', 2, 10, optimal_k)
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        rfm['Cluster'] = kmeans.fit_predict(rfm[['Recency', 'Frequency', 'Monetary']])

        # Periksa apakah jumlah cluster lebih dari satu untuk perhitungan Silhouette Score
        if len(set(rfm['Cluster'])) > 1:
            silhouette_avg = silhouette_score(rfm[['Recency', 'Frequency', 'Monetary']], rfm['Cluster'])
            st.write(f"Silhouette Score for {num_clusters} clusters: {silhouette_avg:.4f}")
        else:
            st.write("Silhouette Score cannot be calculated with only one unique cluster.")

        # Tabel Segmentasi Pelanggan dengan Nama Cluster, Jumlah Anggota, Customer Segment, dan DBI Weight
            st.subheader('Tabel Segmentasi Pelanggan')

            # Hitung jumlah anggota per cluster dan buat DataFrame untuk informasi segmen
            segment_summary = pd.DataFrame({
                'Cluster Name': [f'Cluster {i+1}' for i in range(num_clusters)],
                'Customer Segment': [segment_names.get(i, 'Unknown Segment') for i in range(num_clusters)],
                'Member Count': [rfm[rfm['Cluster'] == i].shape[0] for i in range(num_clusters)],
                'DBI Weight': [segment_dbi_weights.get(segment_names.get(i, 'Unknown Segment'), 'N/A') for i in range(num_clusters)]
            })

            # Tampilkan tabel segmentasi pelanggan
            st.write(segment_summary)

        # Segmentasi Pelanggan Berdasarkan Cluster
        st.subheader('Customer Segmentation by Cluster')
        segment_names = {
            0: 'VIP customers',
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
        rfm['CustomerSegment'] = rfm['Cluster'].map(segment_names)

        # Tabel Segmentasi Pelanggan dengan Keterangan Perilaku dan Treatment
        st.subheader('Tabel Segmentasi Pelanggan dengan Perilaku dan Treatment')

        # Buat DataFrame untuk tabel segmentasi
        segment_behaviors = pd.DataFrame({
            'DBI Weight': [
              0.75, 1.25, 0.75, 1.5, 1.75, 1.6, 1.8, 1.9, 1.4, 1.7
             ],
            'Customer Segment': [
                'VIP Customers', 'Frequent Buyers', 'High Spenders', 'Potential Loyalists', 
                'New Customers', 'Occasional Buyers', 'At-Risk Customers', 'Lost Customers', 
                'Bargain Hunters', 'Window Shoppers'
            ],
            'Behavior': [
                'Pelanggan paling berharga dan setia dengan transaksi tinggi.',
                'Pelanggan yang sering bertransaksi, walaupun dengan nilai pembelian tidak selalu tinggi.',
                'Pelanggan dengan nilai transaksi tinggi namun tidak sering membeli.',
                'Pelanggan yang menunjukkan potensi loyalitas namun belum sepenuhnya konsisten.',
                'Pelanggan baru yang masih dalam tahap eksplorasi dan belum konsisten.',
                'Pelanggan yang membeli secara sporadis atau jarang.',
                'Pelanggan yang pernah aktif tetapi sekarang menurun aktivitasnya.',
                'Pelanggan yang dulu aktif namun sudah lama tidak membeli.',
                'Pelanggan yang sangat sensitif terhadap harga dan mencari diskon.',
                'Pelanggan yang sering melihat-lihat tetapi jarang membeli.'
            ],
            'Treatment': [
                'Berikan hadiah yang dipersonalisasi, program loyalitas, dan layanan pelanggan premium.',
                'Tawarkan hadiah loyalitas dan promosi khusus untuk meningkatkan frekuensi pembelian.',
                'Upsell produk premium atau tawarkan pengalaman eksklusif.',
                'Tingkatkan keterlibatan dengan penawaran yang dipersonalisasi dan program loyalitas.',
                'Berikan penawaran selamat datang dan pandu mereka dalam penemuan produk.',
                'Berikan diskon atau rekomendasi yang sesuai untuk mendorong pembelian lebih sering.',
                'Re-engage dengan penawaran yang dipersonalisasi dan kampanye pemulihan pelanggan.',
                'Tawarkan diskon besar untuk menarik mereka kembali.',
                'Soroti promosi dan penawaran khusus.',
                'Berikan insentif seperti diskon pertama atau rekomendasi produk yang dipersonalisasi.'
            ]
        })

        # Tampilkan tabel segmentasi pelanggan
        st.write(segment_behaviors)

        # Menambahkan bobot DBI untuk setiap segmen
        segment_dbi_weights = {
            'VIP customers': 0.75,
            'Frequent Buyers': 1.25,
            'High Spenders': 0.75,
            'Potential Loyalists': 1.5,
            'New Customers': 1.75,
            'Occasional Buyers': 1.6,
            'At-Risk Customers': 1.8,
            'Lost Customers': 1.9,
            'Bargain Hunters': 1.4,
            'Window Shoppers': 1.7
        }

        # Map DBI weight to each customer based on their segment
        rfm['DBI_Weight'] = rfm['CustomerSegment'].map(segment_dbi_weights)

        # Tampilkan tabel segmentasi dengan bobot DBI untuk setiap customer
        st.subheader('Tabel Segmentasi Pelanggan dengan Bobot DBI')
        st.write(rfm[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'CustomerSegment', 'DBI_Weight']])

        # Display customer segments and treatments
        for cluster in range(num_clusters):
            st.write(f"### Cluster {cluster+1}: {segment_names.get(cluster, 'Unknown Segment')}")
            cluster_data = rfm[rfm['Cluster'] == cluster]
            st.write(cluster_data.describe())
            
            # Treatment untuk setiap Cluster
            treatment = {
                'VIP customers': "Berikan hadiah yang dipersonalisasi, program loyalitas, dan layanan pelanggan premium.",
                'Frequent Buyers': "Tawarkan hadiah loyalitas dan promosi khusus untuk meningkatkan frekuensi pembelian.",
                'High Spenders': "Tawarkan produk premium tambahan dan pengalaman eksklusif.",
                'Potential Loyalists': "Perkuat keterlibatan melalui program loyalitas dan penawaran yang dipersonalisasi.",
                'New Customers': "Berikan penawaran selamat datang dan pandu mereka dalam penemuan produk.",
                'Occasional Buyers': "Ajak kembali dengan diskon atau rekomendasi yang disesuaikan.",
                'At-Risk Customers': "Ajak kembali dengan penawaran yang dipersonalisasi dan kampanye pemulihan pelanggan.",
                'Lost Customers': "Tawarkan diskon besar untuk menarik mereka kembali.",
                'Bargain Hunters': "Soroti promosi dan penawaran khusus.",
                'Window Shoppers': "Berikan insentif dengan diskon pertama dan rekomendasi yang dipersonalisasi."
            }
            st.write(f"**Treatment**: {treatment.get(segment_names[cluster], 'N/A')}")

        # Visualize Clusters
        st.subheader('Cluster Visualization')
        fig, ax = plt.subplots()
        sns.scatterplot(x=rfm['Recency'], y=rfm['Monetary'], hue=rfm['Cluster'], palette='viridis', ax=ax)
        ax.set_title('RFM Clustering Visualization')
        st.pyplot(fig)
    else:
        st.warning("Ensure 'CustomerID', 'TransactionDate', and 'MonetaryValue' columns are present in the uploaded file.")
