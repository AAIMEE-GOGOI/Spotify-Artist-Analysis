import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import json
import urllib.request
import urllib.parse
from sklearn.ensemble import RandomForestRegressor

# =========================================================
# SMOOTHYFY - Spotify Artist & Song Intelligence
# =========================================================

st.set_page_config(
    page_title="Smoothyfy",
    page_icon="🎀",
    layout="wide"
)

FEATURE_COLS = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo"
]

PINK_PLACEHOLDER = (
    "https://placehold.co/300x300/FF6FB5/FFFFFF/png?text=%F0%9F%8E%80"
)


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def get_spotify_poster(spotify_url: str) -> str:
    """Fetch a real poster image (artist photo or album art) via Spotify's
    public oEmbed endpoint — no API key required. Falls back to a pink
    placeholder if the URL is missing or the request fails (e.g. offline
    sandbox, network hiccup, rate limit)."""
    if not spotify_url or not isinstance(spotify_url, str) or "open.spotify.com" not in spotify_url:
        return PINK_PLACEHOLDER
    try:
        oembed_url = "https://open.spotify.com/oembed?url=" + urllib.parse.quote(spotify_url, safe="")
        req = urllib.request.Request(oembed_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        thumb = data.get("thumbnail_url")
        return thumb if thumb else PINK_PLACEHOLDER
    except Exception:
        return PINK_PLACEHOLDER


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def search_spotify_track_url(track_name: str, artist_name: str) -> str:
    """Best-effort: find a track's Spotify page via the public search
    embed so we can pull album art for songs where we only stored the
    artist_url. Returns '' if nothing is found."""
    try:
        query = urllib.parse.quote(f"{track_name} {artist_name}")
        search_url = f"https://open.spotify.com/search/{query}"
        return search_url
    except Exception:
        return ""

# =========================================================
# LOAD MODELS AND DATA
# =========================================================

@st.cache_resource
def load_models():
    with open("kmeans_model.pkl", "rb") as f:
        kmeans_model = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return kmeans_model, scaler


@st.cache_data
def load_data():
    df = pd.read_csv("clustered_songs.csv")

    # Keep every track from the supplied CSV.  artist_url is used as the
    # unique artist identifier because several artist_name values are
    # corrupted in the supplied dataset (for example A / a / Â / Ã / Å).
    for col in ["artist_name", "artist_url", "track_name"]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    df = df[df["track_name"].ne("")].copy()
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")

    def is_corrupt_name(name):
        return name in {"A", "a", "Â", "Ã", "Å", "alex_g_offline"} or len(name) <= 1

    df["name_is_corrupt"] = df["artist_name"].map(is_corrupt_name)

    def make_artist_key(row):
        if row["artist_url"]:
            return row["artist_url"]
        return "name:" + row["artist_name"].casefold()

    df["artist_key"] = df.apply(make_artist_key, axis=1)
    return df.reset_index(drop=True)


def build_artist_catalog(df):
    rows = []
    for key, g in df.groupby("artist_key", sort=False):
        good_names = [
            x for x in g.loc[~g["name_is_corrupt"], "artist_name"].unique()
            if x and len(x) > 1
        ]
        display_name = sorted(good_names, key=lambda x: (-len(x), x))[0] if good_names else ""
        url = g["artist_url"].iloc[0] if g["artist_url"].iloc[0] else ""
        artist_id = url.rstrip("/").split("/")[-1] if url else key
        if not display_name:
            display_name = f"Unknown Artist — {artist_id[-8:]}"
        rows.append({
            "artist_key": key,
            "display_name": display_name,
            "artist_id": artist_id,
            "artist_url": url,
            "songs": len(g),
            "corrupt_name": not bool(good_names),
        })
    return pd.DataFrame(rows).sort_values(
        ["corrupt_name", "display_name"], ascending=[True, True]
    ).reset_index(drop=True)


@st.cache_resource
def train_popularity_model(df):
    """No standalone popularity model was supplied, so we train a light
    Random Forest on the audio features to predict track_popularity."""
    X = df[FEATURE_COLS]
    y = df["track_popularity"]
    model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
    model.fit(X, y)
    return model


kmeans_model, scaler = load_models()
df = load_data()
popularity_model = train_popularity_model(df)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

/* ===== SPOTIFY-STYLE THEME ===== */

html, body, [data-testid="stAppViewContainer"] {
    background: #121212 !important;
    color: white !important;
}

[data-testid="stAppViewContainer"] {
    background: #121212 !important;
}

[data-testid="stHeader"] {
    background: transparent !important;
    height: 0px !important;
    min-height: 0px !important;
}

[data-testid="stToolbar"] {
    display: none !important;
}

[data-testid="stDecoration"] {
    display: none !important;
}

/* Reduce top black space */
.block-container {
    padding-top: 0.8rem !important;
    padding-bottom: 2rem !important;
}


/* ===== SMOOTHYFY TITLE ===== */

.title {
    font-family: Arial, sans-serif !important;
    font-size: 58px !important;
    font-weight: 800 !important;
    text-align: center !important;
    margin-top: 0px !important;
    margin-bottom: 5px !important;
    color: #1DB954 !important;
    text-shadow: 0 0 18px rgba(29,185,84,0.35) !important;
    line-height: 1.1 !important;
}

.subtitle {
    text-align: center !important;
    font-size: 19px !important;
    font-weight: 500 !important;
    margin-top: 0px !important;
    margin-bottom: 25px !important;
    color: #B3B3B3 !important;
}


/* ===== SEARCH ===== */

[data-testid="stTextInput"] label {
    color: white !important;
    font-weight: 700 !important;
}

[data-testid="stTextInput"] input {
    background: #181818 !important;
    color: white !important;
    border: 1px solid #535353 !important;
    border-radius: 10px !important;
}

[data-testid="stTextInput"] input:focus {
    border: 2px solid #1DB954 !important;
    box-shadow: 0 0 8px rgba(29,185,84,0.3) !important;
}


/* ===== BUTTONS ===== */

.stButton > button {
    background: #1DB954 !important;
    color: #000000 !important;
    border: none !important;
    border-radius: 25px !important;
    font-weight: 700 !important;
}

.stButton > button:hover {
    background: #1ED760 !important;
    color: #000000 !important;
}


/* ===== CARDS ===== */

.artist-card,
.song-card {
    background: #181818 !important;
    border: 1px solid #282828 !important;
    border-radius: 14px !important;
}


/* ===== METRICS ===== */

[data-testid="stMetric"] {
    background: #181818 !important;
    border: 1px solid #282828 !important;
    border-radius: 12px !important;
    padding: 15px !important;
}

[data-testid="stMetricLabel"] {
    color: #B3B3B3 !important;
}

[data-testid="stMetricValue"] {
    color: #1DB954 !important;
}


/* ===== SELECT BOX ===== */

[data-baseweb="select"] {
    background: #181818 !important;
}

[data-baseweb="select"] > div {
    background: #181818 !important;
    border-color: #535353 !important;
    color: white !important;
}


/* ===== LINKS ===== */

a {
    color: #1DB954 !important;
}

a:hover {
    color: #1ED760 !important;
}


/* ===== DIVIDERS ===== */

hr {
    border-color: #282828 !important;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================




st.markdown('<div class="title">🎵 Smoothyfy</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Spotify Artist &amp; Song Intelligence🎀</div>',
    unsafe_allow_html=True
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown("## 🎀 Smoothyfy")
st.sidebar.write(
    "Explore your favorite artists and analyze track "
    "characteristics using Machine Learning"
)

st.sidebar.divider()
st.sidebar.markdown("### 🧭 Navigation")

if "page" not in st.session_state:
    st.session_state.page = "Search Artist"

nav_items = [
    ("Search Artist", "🔍"),
    ("About", "💗"),
    ("How It Works", "✨"),
]
for label, icon in nav_items:
    is_active = st.session_state.page == label
    if st.sidebar.button(
        f"{icon} {label}",
        key=f"nav_{label}",
        use_container_width=True,
        type="primary" if is_active else "secondary",
    ):
        st.session_state.page = label

st.sidebar.divider()
st.sidebar.markdown("### 🎤 Singer List")

artist_catalog = build_artist_catalog(df)

# Use artist_key (Spotify artist URL) as the real selection key. This keeps
# different artists separate even when the CSV gives several of them the
# same corrupted name such as "A".
artist_options = artist_catalog["artist_key"].tolist()
artist_labels = dict(zip(artist_catalog["artist_key"], artist_catalog["display_name"]))

selected_sidebar_artist = st.sidebar.selectbox(
    f"Artist List ({len(artist_catalog)} artists in dataset)",
    ["__ALL__"] + artist_options,
    format_func=lambda x: "All Artists" if x == "__ALL__" else artist_labels[x],
)

st.sidebar.caption(
    "Select an artist first. The song selector will then show only songs "
    "belonging to that exact artist record."
)

st.sidebar.divider()
st.sidebar.info(
    f"📊 Dataset: {len(artist_catalog)} unique artist IDs · {len(df):,} tracks. "
    "Names are taken from the uploaded CSV."
)

bad_artist_count = int(artist_catalog["corrupt_name"].sum())
if bad_artist_count:
    with st.sidebar.expander("⚠️ Data-quality note"):
        st.write(
            f"{bad_artist_count} artist records have corrupted/placeholder "
            "names in the CSV. They are kept separate rather than being "
            "incorrectly merged or renamed."
        )



# =========================================================
# PAGE ROUTING
# =========================================================

if st.session_state.page == "Search Artist":

    st.markdown('<div class="section-header">🔍 Search Artist</div>', unsafe_allow_html=True)

    search_col, btn_col = st.columns([5, 1])
    with search_col:
        search_query = st.text_input(
            "Search for an artist...", placeholder="Search for an artist...", label_visibility="collapsed"
        )
    with btn_col:
        search_clicked = st.button("💗 Search", use_container_width=True)

    # Resolve the active artist from the complete dataset-backed catalog.
    active_artist_key = None

    if search_clicked and search_query.strip():
        q = search_query.strip().casefold()
        matches = artist_catalog[
            artist_catalog["display_name"].str.casefold().str.contains(
                q, regex=False, na=False
            )
        ]
        if len(matches):
            active_artist_key = matches.iloc[0]["artist_key"]

    elif selected_sidebar_artist != "__ALL__":
        active_artist_key = selected_sidebar_artist

    else:
        named = artist_catalog[~artist_catalog["corrupt_name"]]
        if len(named):
            active_artist_key = named.iloc[0]["artist_key"]
        elif len(artist_catalog):
            active_artist_key = artist_catalog.iloc[0]["artist_key"]

    st.divider()


    # =========================================================
    # ARTIST PROFILE
    # =========================================================

    if active_artist_key:

        artist_df = df[df["artist_key"] == active_artist_key].sort_values(
            "track_popularity", ascending=False
        )
        active_artist = artist_labels[active_artist_key]

        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)

            p1, p2 = st.columns([1, 3])

            with p1:
                artist_url_for_photo = (
                    artist_df["artist_url"].dropna().iloc[0]
                    if artist_df["artist_url"].notna().any() else None
                )
                poster_url = get_spotify_poster(artist_url_for_photo)
                st.markdown(
                    f"""
                    <div style="width:100%;aspect-ratio:1/1;border-radius:18px;
                    overflow:hidden;box-shadow:0 8px 20px rgba(255,46,146,0.35);
                    border:2px solid #FF4FA3;">
                    <img src="{poster_url}" style="width:100%;height:100%;object-fit:cover;" />
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with p2:
                st.markdown(f"### {active_artist} 💖")

                m1, m2, m3 = st.columns(3)
                m1.metric("👥 Followers", f"{int(artist_df['followers'].max()):,}")
                m2.metric("⭐ Artist Popularity", f"{int(artist_df['artist_popularity'].max())}/100")
                m3.metric("🎵 Songs in Dataset", f"{len(artist_df)}")
                st.caption(
                    f"Every song shown below belongs to **{active_artist}** "
                    "in the uploaded dataset."
                )

                genres = artist_df["genres"].dropna().unique()
                if len(genres):
                    st.write(f"**Genres:** {', '.join(genres[:5])}")

                artist_url = artist_df["artist_url"].dropna().iloc[0] if artist_df["artist_url"].notna().any() else None
                if artist_url:
                    st.link_button("🎧 Open Artist on Spotify", artist_url)

            st.markdown('</div>', unsafe_allow_html=True)

        st.divider()

        # =====================================================
        # TOP SONGS TABLE
        # =====================================================

        st.markdown('<div class="section-header">🎶 Top 10 Songs in Dataset</div>', unsafe_allow_html=True)

        top_songs = artist_df.head(10).copy()

        # Poster for the artist (used as the track thumbnail since we only
        # have one Spotify URL per artist row, not one per track)
        artist_poster = get_spotify_poster(
            artist_df["artist_url"].dropna().iloc[0] if artist_df["artist_url"].notna().any() else None
        )

        header_cols = st.columns([0.4, 0.7, 2.2, 1.8, 1, 1.3, 1])
        headers = ["#", "", "Song", "Album", "Pop.", "Release Date", "Genre"]
        for c, h in zip(header_cols, headers):
            c.markdown(
                f'<span style="color:#FF6FB5;font-weight:800;letter-spacing:0.5px;'
                f'text-transform:uppercase;font-size:13px;">{h}</span>',
                unsafe_allow_html=True
            )

        for i, (_, row) in enumerate(top_songs.iterrows(), start=1):
            c1, c2, c3, c4, c5, c6, c7 = st.columns([0.4, 0.7, 2.2, 1.8, 1, 1.3, 1])
            c1.write(f"{i}")
            c2.markdown(
                f'<img src="{artist_poster}" style="width:42px;height:42px;'
                f'border-radius:8px;object-fit:cover;border:1px solid #FF4FA3;" />',
                unsafe_allow_html=True
            )
            c3.markdown(f'<span style="color:#FFFFFF;font-weight:600;">{row["track_name"]}</span>', unsafe_allow_html=True)
            c4.markdown(f'<span style="color:#D9A8C6;">{row["album_name"]}</span>', unsafe_allow_html=True)
            c5.markdown(f"🩷 {int(row['track_popularity'])}")
            c6.write(row["release_date"].strftime("%Y-%m-%d") if pd.notna(row["release_date"]) else "—")
            c7.write(row["genres"])
            st.progress(min(max(row["track_popularity"] / 100, 0.0), 1.0))

        st.divider()

        # =====================================================
        # ANALYZE A SONG / AUDIO FEATURES
        # =====================================================

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown('<div class="section-header">🎼 Analyze a Song</div>', unsafe_allow_html=True)
            song_options = (
                artist_df["track_name"]
                .dropna()
                .astype(str)
                .drop_duplicates()
                .tolist()
            )
            song_choice = st.selectbox(
                f"Songs by {active_artist} ({len(song_options)} in dataset)",
                song_options,
                key=f"song_select_{active_artist_key}",
            )
            song_row = artist_df[artist_df["track_name"] == song_choice].iloc[0]

            song_poster = get_spotify_poster(song_row.get("artist_url"))

            sp1, sp2 = st.columns([1, 2])
            with sp1:
                st.markdown(
                    f'<img src="{song_poster}" style="width:100%;aspect-ratio:1/1;'
                    f'object-fit:cover;border-radius:14px;border:2px solid #FF4FA3;'
                    f'box-shadow:0 6px 14px rgba(255,105,180,0.3);" />',
                    unsafe_allow_html=True
                )
            with sp2:
                release_str = song_row['release_date'].strftime('%Y-%m-%d') if pd.notna(song_row['release_date']) else 'N/A'
                st.markdown(
                    f"""
                    <div style="padding:22px;border-radius:18px;border:1px solid #FF4FA3;
                    background:linear-gradient(160deg,#241722,#170e18);
                    box-shadow:0 0 18px rgba(255,46,146,0.2);height:100%;">
                        <div style="color:#FF6FB5;font-weight:700;font-size:15px;margin-bottom:10px;">
                            💿 {song_row['track_name']}
                        </div>
                        <div style="color:#F5E5F0;line-height:1.9;font-size:14px;">
                            <b style="color:#FF9EC9;">Album:</b> {song_row['album_name']}<br>
                            <b style="color:#FF9EC9;">Release Date:</b> {release_str}<br>
                            <b style="color:#FF9EC9;">Genre:</b> {song_row['genres']}<br>
                            <b style="color:#FF9EC9;">Popularity:</b> {int(song_row['track_popularity'])}/100
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            if pd.notna(song_row.get("artist_url")):
                st.link_button("🎧 Open on Spotify", song_row["artist_url"])

        with col_right:
            st.markdown('<div class="section-header">🎚️ Audio Feature Analysis</div>', unsafe_allow_html=True)

            fa, fb = st.columns(2)
            with fa:
                st.write(f"**Danceability**  \n{song_row['danceability']:.3f}")
                st.progress(min(max(song_row['danceability'], 0.0), 1.0))
                st.write(f"**Energy**  \n{song_row['energy']:.3f}")
                st.progress(min(max(song_row['energy'], 0.0), 1.0))
                st.write(f"**Loudness**  \n{song_row['loudness']:.3f} dB")
                st.progress(min(max((song_row['loudness'] + 60) / 60, 0.0), 1.0))
                st.write(f"**Speechiness**  \n{song_row['speechiness']:.3f}")
                st.progress(min(max(song_row['speechiness'], 0.0), 1.0))
                st.write(f"**Acousticness**  \n{song_row['acousticness']:.3f}")
                st.progress(min(max(song_row['acousticness'], 0.0), 1.0))

            with fb:
                st.write(f"**Instrumentalness**  \n{song_row['instrumentalness']:.3f}")
                st.progress(min(max(song_row['instrumentalness'], 0.0), 1.0))
                st.write(f"**Liveness**  \n{song_row['liveness']:.3f}")
                st.progress(min(max(song_row['liveness'], 0.0), 1.0))
                st.write(f"**Valence**  \n{song_row['valence']:.3f}")
                st.progress(min(max(song_row['valence'], 0.0), 1.0))
                st.write(f"**Tempo**  \n{song_row['tempo']:.1f} BPM")
                st.progress(min(max(song_row['tempo'] / 220, 0.0), 1.0))

        st.divider()

        # =====================================================
        # MACHINE LEARNING RESULTS
        # =====================================================

        st.markdown('<div class="section-header">🤖 Machine Learning Results</div>', unsafe_allow_html=True)

        features = np.array([[song_row[c] for c in FEATURE_COLS]])
        scaled_features = scaler.transform(features)
        cluster = int(kmeans_model.predict(scaled_features)[0])
        predicted_popularity = float(np.clip(popularity_model.predict(features)[0], 0, 100))
        actual_popularity = int(song_row["track_popularity"])

        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric("🧩 K-Means Music Cluster", f"Cluster {cluster}")
            st.caption(f"This song belongs to Cluster {cluster}. This cluster contains {int((df['cluster'] == cluster).sum())} tracks in the dataset.")
        with r2:
            st.metric("🌳 Random Forest Prediction", f"{predicted_popularity:.1f}/100")
            st.caption("Predicted by Random Forest using audio features.")
        with r3:
            st.metric("⭐ Actual Popularity", f"{actual_popularity}/100")
            st.caption("Actual popularity from the dataset.")

        st.divider()

        b1, b2 = st.columns(2)

        with b1:
            st.markdown('<div class="section-header">📊 Actual vs Predicted Popularity</div>', unsafe_allow_html=True)
            compare_df = pd.DataFrame({
                "Type": ["Actual Popularity", "Predicted Popularity"],
                "Popularity": [actual_popularity, predicted_popularity]
            }).set_index("Type")
            st.bar_chart(compare_df, color="#FF5FAE")

        with b2:
            st.markdown('<div class="section-header">🧵 Cluster Characteristics (Top 3)</div>', unsafe_allow_html=True)
            cluster_means = df[df["cluster"] == cluster][FEATURE_COLS].mean().sort_values(ascending=False)
            top3 = cluster_means.head(3)
            for i, (feat, val) in enumerate(top3.items(), start=1):
                st.write(f"**{i}. {feat.capitalize()}**")
                st.progress(min(max(val, 0.0), 1.0) if val <= 1 else 1.0)
                st.caption(f"{val:.3f}")

    else:
        st.info("No artists found. Try a different search term. 💗")


elif st.session_state.page == "About":

    st.markdown('<div class="section-header">💗 About Smoothyfy</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div style="padding:28px;border-radius:20px;border:1px solid #FF4FA3;
        background:linear-gradient(160deg,#241722,#170e18);
        box-shadow:0 0 22px rgba(255,46,146,0.25);color:#F5E5F0;
        line-height:1.9;font-size:15px;">
        <b style="color:#FF6FB5;font-size:19px;">🎀 What is Smoothyfy?</b><br><br>
        Smoothyfy is a Spotify Artist &amp; Song Intelligence dashboard. It lets you
        search for an artist, browse their most popular tracks, and dig into the
        audio DNA of any song — danceability, energy, valence, tempo, and more.<br><br>
        <b style="color:#FF6FB5;font-size:17px;">🧠 What powers it?</b><br><br>
        • A <b style="color:#FF9EC9;">K-Means clustering model</b> groups every track in the
        dataset into sonic "moods" based on its audio features.<br>
        • A <b style="color:#FF9EC9;">Random Forest model</b> predicts how popular a track
        should be, purely from its audio characteristics — no metadata, no hype,
        just the sound.<br>
        • Artist and album artwork are pulled live from Spotify's public oEmbed
        service.<br><br>
        <b style="color:#FF6FB5;font-size:17px;">💌 Why "Smoothyfy"?</b><br><br>
        Because good music should feel effortless to explore — smooth search,
        smooth insights, wrapped in a Barbie-pink bow.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    m1, m2, m3 = st.columns(3)
    m1.metric("🎵 Total Tracks", f"{len(df):,}")
    m2.metric("🎤 Total Artists", f"{df['artist_key'].nunique():,}")
    m3.metric("🧩 Clusters Found", f"{df['cluster'].nunique()}")


elif st.session_state.page == "How It Works":

    st.markdown('<div class="section-header">✨ How It Works</div>', unsafe_allow_html=True)

    steps = [
        ("1️⃣", "Search for an artist",
         "Type a name in the search bar, tap a Popular pill, or pick someone "
         "from the Singer List in the sidebar."),
        ("2️⃣", "Browse their Top 10 Songs",
         "See each track's poster, album, popularity score, release date, and "
         "genre — ranked from most to least popular in the dataset."),
        ("3️⃣", "Pick a song to analyze",
         "Select any track from that artist to see its full audio feature "
         "breakdown: danceability, energy, loudness, valence, tempo, and more."),
        ("4️⃣", "See the Machine Learning results",
         "The K-Means model places the song into a sonic cluster, and the "
         "Random Forest model predicts its popularity from audio features alone "
         "— then compares that prediction to the song's real popularity."),
    ]

    for icon, title, desc in steps:
        st.markdown(
            f"""
            <div style="padding:20px 24px;border-radius:18px;border:1px solid #FF4FA3;
            background:linear-gradient(160deg,#241722,#170e18);
            box-shadow:0 0 16px rgba(255,46,146,0.18);margin-bottom:14px;">
            <span style="font-size:20px;">{icon}</span>
            <span style="color:#FF6FB5;font-weight:700;font-size:16px;"> {title}</span><br>
            <span style="color:#D9A8C6;font-size:14px;">{desc}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

# =========================================================
# FOOTER
# =========================================================

st.divider()
st.markdown(
    '<div class="footer-text">Smoothyfy • Spotify Artist &amp; Song Intelligence • Built with Streamlit &amp; Machine Learning 💗</div>',
    unsafe_allow_html=True
)
