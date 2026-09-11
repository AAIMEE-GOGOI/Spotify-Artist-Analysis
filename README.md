# 🎵 Smoothyfy — Spotify Artist & Song Intelligence

<p align="center">
  <img src="smoothfy-logo.png" width="180">
</p>

<p align="center">
  <b>Explore • Analyze • Discover</b>
</p>

<p align="center">
  An interactive music analytics application built with Python, Streamlit and Machine Learning.
</p>

---

## 🚀 Live Demo

<p align="center">

👉 **[🎧 Open Smoothyfy Live](https://aaimee-gogoi-spotify-artist-analysis-smoothyfy-corrected-jmrllu.streamlit.app/)**

</p>

---

## 🎶 About Smoothyfy

**Smoothyfy** is an interactive Spotify Artist & Song Analysis application designed to turn music data into meaningful insights.

Users can search for artists, explore their statistics, discover associated songs, analyze popularity, and interact with data-driven visualizations through a simple Streamlit dashboard.

---

## ✨ Features

🔎 **Artist Search**  
Search and explore artists from the available dataset.

👤 **Artist Insights**  
View artist information, followers and popularity metrics.

🎵 **Song Exploration**  
Explore songs associated with selected artists.

⭐ **Popularity Analysis**  
Analyze artist and song popularity using data-driven insights.

📊 **Interactive Visualizations**  
Understand music trends through visual and interactive charts.

🤖 **Machine Learning**  
Use clustering and prediction models to analyze Spotify song data.

🎧 **Spotify Integration**  
Open artists directly on Spotify from the application.

---

## 🧠 Machine Learning

Smoothyfy includes machine-learning components for analyzing Spotify song data.

### Models & Techniques

- 🔹 **K-Means Clustering** — groups songs based on their characteristics
- 🔹 **Feature Scaling** — prepares numerical features for machine learning
- 🔹 **Popularity Prediction** — analyzes and predicts song popularity
- 🔹 **Pre-trained Models** — `.pkl` files are loaded directly into the application

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| 🐍 Python | Core programming |
| 🎈 Streamlit | Web application |
| 🐼 Pandas | Data analysis |
| 🔢 NumPy | Numerical computation |
| 🤖 Scikit-learn | Machine Learning |
| 📦 Joblib | Model loading |
| 📈 Matplotlib | Visualization |
| 📊 Seaborn | Statistical visualization |
| 📉 Plotly | Interactive visualization |

---

## 📂 Dataset

The project uses Spotify song and artist data containing information related to:

- Artist information
- Song characteristics
- Popularity
- Audio-related features

The dataset is processed using **Pandas** and used for exploratory analysis and machine-learning tasks.

---

## 📁 Project Structure

```text
Spotify-Artist-Analysis/
│
├── 🎵 Smoothify_corrected.py
├── 📊 spotifydataset.csv
├── 📊 clustered_songs.csv
│
├── 🤖 kmeans_model.pkl
├── 🤖 popularity_model.pkl
├── ⚙️ scaler.pkl
│
├── 📦 requirements.txt
├── 🖼️ smoothfy-logo.png
└── 📖 README.md
