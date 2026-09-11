# 🎵 Smoothify — Spotify Artist & Song Intelligence

Smoothify is an interactive **Spotify Artist & Song Analysis** web application built with Python and Streamlit.

The application allows users to explore artists, analyze song characteristics, view popularity insights, and discover songs through an interactive dashboard.

## 🚀 Live Demo

👉 **[Open Smoothify Live](https://aaimee-gogoi-spotify-artist-analysis-smoothyfy-corrected-jmrllu.streamlit.app/)**

## 📌 Features

- 🔎 Search and explore Spotify artists
- 👤 View artist information and statistics
- 🎵 Explore songs associated with artists
- ⭐ Analyze artist/song popularity
- 📊 Interactive data visualizations
- 🤖 Machine-learning based song analysis
- 🎧 Direct links to open artists on Spotify
- 📁 Data-driven analysis using Spotify datasets

## 🛠️ Technologies Used

- **Python**
- **Streamlit** — Interactive web application
- **Pandas** — Data manipulation and analysis
- **NumPy** — Numerical computing
- **Scikit-learn** — Machine learning
- **Joblib** — Model loading and serialization
- **Matplotlib** — Data visualization
- **Seaborn** — Statistical visualization
- **Plotly** — Interactive visualizations

## 🤖 Machine Learning

Smoothify includes machine-learning components for analyzing Spotify song data.

The project uses:

- **K-Means Clustering** for grouping similar songs
- **Scaling/Normalization** of numerical features
- **Popularity Prediction Model** for analyzing song popularity

Pre-trained models are stored in `.pkl` files and loaded by the Streamlit application.

## 📊 Dataset

The project uses Spotify song and artist data containing information related to music characteristics and popularity.

The data is processed using Pandas and used for both exploratory analysis and machine-learning tasks.

## 💻 Project Structure

```text
Spotify-Artist-Analysis/
│
├── Smoothify_corrected.py
├── spotifydataset.csv
├── clustered_songs.csv
├── kmeans_model.pkl
├── popularity_model.pkl
├── scaler.pkl
├── requirements.txt
└── README.md
