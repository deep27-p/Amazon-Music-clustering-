# Amazon-Music-clustering-
🎧 Amazon Music Clustering Explorer – Project Overview

1️⃣ Objective

The goal of this project is to analyze Amazon Music dataset using clustering techniques and build an interactive dashboard that helps explore music patterns, user preferences, and song similarities.
It allows users to:
Discover hidden patterns in music features (danceability, energy, valence, tempo, etc.).
Explore clusters of songs grouped by similarity.
Search for specific songs or artists.
Get song recommendations based on similarity.
Visualize music data with interactive charts.

2️⃣ Dataset

The dataset contains Amazon Music songs with features such as:

🎵 Name of song
🎤 Artist(s)
🎭 Genres
🎶 Danceability, Energy, Valence, Tempo, Acousticness, Popularity, Duration
📊 Cluster labels (KMeans, DBSCAN, Hierarchical)

3️⃣ Tools & Technologies

Python – Core programming language
Streamlit – Interactive web dashboard
Pandas, NumPy – Data cleaning & preprocessing
Matplotlib, Seaborn, Plotly – Data visualization
Scikit-learn – Machine learning (clustering, similarity, PCA, t-SNE)

🌟 Key Features

🔍 Search & Explore

Song Search → Find songs instantly by name.
Artist Analysis → Explore an artist’s top songs, clusters, and styles.
Genre Filtering → Filter dataset by specific genres.

🎯 Clustering & Insights

KMeans, DBSCAN, Hierarchical Clustering → Group songs by similarity.
Cluster Distribution → Visualize number of songs in each cluster.
Feature Comparison → Compare clusters across danceability, energy, valence, etc.
Outlier Detection (DBSCAN) → Identify unique/unusual songs.

📊 Visualizations

Interactive PCA & t-SNE Plots → Explore clusters in 2D.Bar, Pie, and Heatmaps → Analyze features and similarities.
Radar Charts → Compare song/cluster feature profiles.
🎵 Recommendations

Similar Songs → Suggests songs using cosine similarity.
Cluster-based Recommendations → Discover related tracks within a cluster.
Artist-based Suggestions → Find songs close in style to a chosen artist.
⚡ User-Friendly DashboardBuilt with Streamlit → Fast, interactive, and web-based.
Sidebar Filters → Adjust energy, tempo, danceability ranges.
Export Options → Download filtered data as CSV/JSON.
Interactive Cards & Metrics → Quick stats (total songs, clusters, avg energy).

✅ In short: This project is a music analytics + recommendation system, wrapped into an interactive dashboard with strong data visualization + ML clustering features.
