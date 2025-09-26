# Increase Pandas Styler limit to handle large datasets

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ===========================
# 🎵 Amazon Music Clustering Dashboard (All Methods)
# ===========================
pd.set_option("styler.render.max_elements", 1000000)  # or higher if needed

st.set_page_config(
    page_title="Amazon Music Clusters Explorer",
    page_icon="🎶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===========================
# 📂 Load Data
# ===========================
@st.cache_data
def load_data():
    zon = pd.read_csv('amazon_music_clusters_all_methods.csv')
    return zon

try:
    zon = load_data()
except FileNotFoundError:
    st.error("❌ File 'amazon_music_clusters_all_methods.csv' not found. Please ensure it's in the same folder as this script.")
    st.stop()

# ===========================
# 🧠 Cluster Label Mapping
# ===========================
CLUSTER_LABELS_KMEANS = {
    0: "🔥 Party / High Energy",
    1: "🍃 Chill Acoustic",
    2: "😊 Happy Pop",
    3: "🎤 Rap / Spoken Word",
}

CLUSTER_LABELS_DBSCAN = {
    -1: "⚠️ Noise / Outlier",
    0: "Cluster 0",
    1: "Cluster 1",
    2: "Cluster 2",
    3: "Cluster 3",
}

CLUSTER_LABELS_HC = {
    0: "Group A",
    1: "Group B",
    2: "Group C",
    3: "Group D",
}

# Apply labels
zon['cluster_kmeans_label'] = zon['cluster'].map(CLUSTER_LABELS_KMEANS).fillna('Unknown')
zon['cluster_dbscan_label'] = zon['cluster_dbscan'].map(CLUSTER_LABELS_DBSCAN).fillna('Unknown')
zon['cluster_hc_label'] = zon['cluster_hc'].map(CLUSTER_LABELS_HC).fillna('Unknown')

# ===========================
# 🧭 Sidebar Filters
# ===========================
st.sidebar.title("🎛️ Filter & Explore")

method = st.sidebar.radio(
    "Clustering Method",
    ["K-Means", "DBSCAN", "Hierarchical"]
)

if method == "K-Means":
    cluster_col = 'cluster'
    label_col = 'cluster_kmeans_label'
    labels_dict = CLUSTER_LABELS_KMEANS
elif method == "DBSCAN":
    cluster_col = 'cluster_dbscan'
    label_col = 'cluster_dbscan_label'
    labels_dict = CLUSTER_LABELS_DBSCAN
else:
    cluster_col = 'cluster_hc'
    label_col = 'cluster_hc_label'
    labels_dict = CLUSTER_LABELS_HC

# ===========================
# 🏠 Main Dashboard
# ===========================
st.title("🎧 Amazon Music Clustering Dashboard")
st.markdown(f"### Exploring clusters using **{method}**")

zon_filtered = zon.copy()

# ===========================
# 📊 Summary Stats
# ===========================
col1, col2, col3 = st.columns(3)
col1.metric("Total Songs", len(zon))
col2.metric("Filtered Songs", len(zon_filtered))
col3.metric("Unique Clusters", zon[cluster_col].nunique())

st.markdown("---")

# ===========================
# 📈 Cluster Distribution
# ===========================
st.subheader("Cluster Distribution")

cluster_counts = zon[label_col].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(10, 4))
palette = sns.color_palette("Set2", len(cluster_counts))
bars = sns.barplot(x=cluster_counts.index, y=cluster_counts.values, ax=ax, palette=palette)
plt.xticks(rotation=45, ha='right')
plt.title(f"Distribution of Songs by {method} Clusters")
plt.ylabel("Number of Songs")
for bar in bars.patches:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            int(bar.get_height()), ha='center', va='bottom')
st.pyplot(fig)

st.markdown("---")

# ===========================
# 🖼️ Cluster Visualizations (Dynamic)
# ===========================
st.markdown("---")
st.subheader("🖼️ Cluster Visualizations")

viz_options = ["PCA Plot", "t-SNE Plot", "Heatmap"]
selected_viz = st.sidebar.selectbox("Choose Visualization", viz_options)

# Numeric audio features
reco_feature_cols = ['danceability', 'energy', 'valence', 'tempo']
available_features = [c for c in reco_feature_cols if c in zon.columns]

if len(available_features) >= 2:
    X = zon[available_features].astype(float)

    if selected_viz == "PCA Plot":
        pca = PCA(n_components=2)
        pcs = pca.fit_transform(X)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            x=pcs[:, 0], y=pcs[:, 1],
            hue=zon[label_col],
            palette="Set2", s=60, ax=ax
        )
        ax.set_title(f"PCA Visualization of {method} Clusters")
        st.pyplot(fig)

    elif selected_viz == "t-SNE Plot":
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate="auto")
        tsne_results = tsne.fit_transform(X)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            x=tsne_results[:, 0], y=tsne_results[:, 1],
            hue=zon[label_col],
            palette="Set2", s=60, ax=ax
        )
        ax.set_title(f"t-SNE Visualization of {method} Clusters")
        st.pyplot(fig)

    elif selected_viz == "Heatmap":
        cluster_means = zon.groupby(label_col)[available_features].mean()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(cluster_means, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
        ax.set_title(f"Cluster Feature Heatmap ({method})")
        st.pyplot(fig)
else:
    st.warning("Not enough numeric features available for visualization.")

# ===========================
# 🎶 Playlist Generator by Cluster
# ===========================
st.markdown("---")
st.subheader("🎶 Playlist Generator by Cluster")

if len(available_features) >= 2 and zon[cluster_col].notna().any():
    colp1, colp2 = st.columns([2, 1])
    with colp1:
        playlist_cluster = st.sidebar.selectbox(
            "Choose cluster for playlist",
            options=sorted(zon[cluster_col].dropna().unique()),
            format_func=lambda x: f"{x}: {labels_dict.get(x, 'Unknown')}"
        )
    with colp2:
        playlist_size = st.sidebar.number_input("Playlist size", min_value=5, max_value=100, value=20, step=1)

    cluster_subset = zon[zon[cluster_col] == playlist_cluster]
    if len(cluster_subset) == 0:
        st.info("No songs in the selected cluster.")
    else:
        X = zon[available_features].astype(float)
        Xz = (X - X.mean()) / (X.std(ddof=0) + 1e-9)
        centroid = Xz.loc[cluster_subset.index].mean().values.reshape(1, -1)
        sims_cluster = cosine_similarity(centroid, Xz.loc[cluster_subset.index].values)[0]
        order = np.argsort(-sims_cluster)
        top_idx = cluster_subset.index[order][:int(playlist_size)]
        playlist_zon = zon.loc[top_idx, ['name_song', 'name_artists', 'genres', cluster_col, label_col] + available_features].copy()
        playlist_zon.insert(0, 'cluster_similarity', sims_cluster[order][:int(playlist_size)])

        st.dataframe(
            playlist_zon.reset_index(drop=True).style.format({
                'cluster_similarity': '{:.3f}'
            }),
            use_container_width=True,
            height=350
        )
else:
    st.info("Playlist generator requires valid clusters and feature columns.")

# ===========================
# 💡 About & Download
# ===========================
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader("💡 About This Dashboard")
    st.markdown("""
    - Built for **Amazon Music Clustering Project**
    - Compares **K-Means, DBSCAN, Hierarchical** clustering
    - Uses **audio features** to group similar songs
    - Great for playlist curation, recommendation, artist analysis
    """)

with col2:
    st.subheader("📥 Download Data")
    csv = zon_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download Filtered Songs (CSV)",
        csv,
        "filtered_songs.csv",
        "text/csv",
        key='download-csv'
    )

# ===========================
# 🎉 Footer
# ===========================
st.markdown("---")
st.caption("🎵 Amazon Music Clustering Dashboard — Powered by Streamlit & Unsupervised Learning")

# Button: Show Recommendations
if st.button("🎶 Show Recommendations"):
    st.info(f"Top recommended songs from {method}")

    features = ["danceability", "energy", "valence", "tempo"]
    X = zon[features].astype(float)
    Xz = (X - X.mean()) / (X.std(ddof=0) + 1e-9)

    for cid in sorted(zon[cluster_col].dropna().unique()):
        cluster_data = zon[zon[cluster_col] == cid]
        if len(cluster_data) > 5:
            centroid = Xz.loc[cluster_data.index].mean().values.reshape(1, -1)
            sims = cosine_similarity(centroid, Xz.loc[cluster_data.index])[0]
            top_idx = cluster_data.index[np.argsort(-sims)[:5]]

            st.markdown(f"### 🎵 Cluster {cid}: {labels_dict.get(cid, 'Unknown')}")
            st.table(zon.loc[top_idx, ["name_song", "name_artists", "genres"]])
