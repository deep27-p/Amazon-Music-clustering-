# Amazon-Music-clustering-
Amazon Music Clustering
Unsupervised Learning for Music Discovery & Playlist Intelligence

 Transform raw audio data into intelligent music clusters — and build AI-powered playlists that feel human.

The Amazon Music Clustering Dashboard is an end-to-end unsupervised machine learning project designed to discover hidden patterns in music listening behavior using audio features like danceability, energy, valence, and tempo. Built for music analysts, data scientists, and streaming platform teams, this tool reveals how songs naturally group together — not by genre labels, but by how they sound.

Using K-Means, DBSCAN, and Hierarchical Clustering, we uncover 4–6 distinct musical “personas” across 95,000+ tracks — from chill acoustic ballads to high-energy dance anthems. The results are visualized through an interactive Streamlit dashboard, enabling users to explore clusters, generate smart playlists, search songs, and export insights — all without writing a single line of code.

This project demonstrates the full lifecycle of an unsupervised ML application: ✅ Exploratory Data Analysis (EDA) ✅ Feature engineering & normalization ✅ Multi-algorithm clustering ✅ Dimensionality reduction (PCA) ✅ Cluster interpretation & labeling ✅ Interactive deployment via Streamlit

Perfect for building personalized recommendation engines, optimizing radio stations, or understanding listener segmentation in music streaming services.

🎯 Goal
The primary goal is to automatically group similar songs into meaningful musical clusters based on their audio characteristics — and turn those clusters into actionable, user-friendly tools for music discovery and playlist generation.

This enables:

Music platforms to auto-generate mood-based playlists (e.g., “Chill Vibes,” “Workout Energy”) Artists & labels to understand how their music fits within broader listener trends Data teams to validate and refine genre classifications using objective audio metrics Listeners to discover new music aligned with their sonic preferences — not just popularity

📊 Dataset Insight
The dataset contains 95,837 songs from Spotify/Amazon-style metadata, enriched with:

Feature Type	Fields
🎵 Audio Features	danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo
📑 Metadata	name_song, name_artists, genres, popularity_songs, duration_ms, explicit
🔖 Cluster Labels	cluster (K-Means), cluster_dbscan, cluster_hc (Hierarchical)
👨‍🎤 Artist Info	artist_popularity, follower_count
💡 Key Insight: Songs are grouped not by human-assigned genres (which can be inconsistent), but by measurable sonic traits — revealing true musical DNA. 

Example Cluster Interpretations:

Cluster	Label	Description
0	🎵 Acoustic & Instrumental	High acousticness, low energy/danceability, moderate valence
1	🔥 High Energy & Dance	High danceability, energy, tempo; loud, low speechiness
2	🎤 Rap & Spoken Word	High speechiness, low instrumentalness/acousticness
3	😊 Happy Pop & Upbeat	High valence, danceability, energy; medium tempo
^              ^

🛠 Tech Stack
Python – Core programming language for data analysis and modeling
Pandas – Data manipulation, cleaning, and wrangling
NumPy – Numerical computations and preprocessing
Scikit-learn – Machine learning models, preprocessing, and evaluation
Matplotlib / Seaborn – Data visualization and statistical plots
Streamlit – Interactive web application for model deployment
Pickle – Model serialization and saving pipelines
All tools are open-source, Python-native, and optimized for data science workflows. 

🚀 Key Features
Multi-Algorithm Clustering — Compare results from K-Means, DBSCAN, and Hierarchical Clustering side-by-side
Interactive Cluster Exploration — Filter by genre, artist, popularity, duration, explicit content, and audio feature ranges
Radar Charts — Visually compare cluster profiles across 9 audio dimensions
PCA Visualization — See high-dimensional clusters projected into 2D space
Correlation Heatmap — Understand relationships between audio features (e.g., energy ↔ danceability)
Smart Playlist Generator — Build custom playlists per cluster with adjustable size and similarity threshold
AI-Powered Similarity Engine — Find songs most similar to any track using cosine distance on normalized audio features
Song & Artist Search — Instantly locate songs or artists (case-insensitive, partial match support)
Artist Analysis — View which clusters an artist dominates, and their average audio profile
Export Capabilities — Download filtered data as CSV/JSON, save generated playlists, or export summary reports
Real-Time Filtering — Sliders and selectors update all visuals instantly — no page reload needed
Mobile-Friendly UI — Fully responsive design works on desktop, tablet, and phone
