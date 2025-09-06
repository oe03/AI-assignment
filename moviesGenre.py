import math  # ✅ CHANGED: needed for ceil if you want it
import streamlit as st
import pandas as pd
import re
import random

class GenreRecommender:
    def __init__(self, movies_file, ratings_file):
        # Load data
        self.movies = pd.read_csv(movies_file)
        self.ratings = pd.read_csv(ratings_file)

        # Extract year from title
        self.movies['year'] = self.movies['title'].apply(
            lambda x: int(re.search(r'\((\d{4})\)', x).group(1)) if re.search(r'\((\d{4})\)', x) else None
        )

        # Create clean title (without year)
        self.movies['clean_title'] = self.movies['title'].apply(
            lambda x: re.sub(r'\s*\(\d{4}\)', '', x)
        )

        # ---------- ⭐ Use the same formula as your rating module ----------
        # Per-movie stats: votes (v) and plain average (R -> avg)
        stats = (self.ratings.groupby('movieId')
                 .agg(v=('rating', 'count'), R=('rating', 'mean'))
                 .reset_index())
        stats = stats.rename(columns={'R': 'avg'})  # keep plain average for display

        # Global mean C and vote threshold m (80th percentile like your module)
        self.C = float(self.ratings['rating'].mean())
        m_q = float(stats['v'].quantile(0.80))
        self.m = int(m_q)  # or int(math.ceil(m_q)) if you prefer a ceiling

        # IMDb-style weighted score for ranking ONLY
        stats['score'] = (stats['v'] / (stats['v'] + self.m)) * stats['avg'] + (self.m / (stats['v'] + self.m)) * self.C

        # Merge stats into movies dataframe (now movies has: avg, v, score)
        self.movies = self.movies.merge(stats, on='movieId', how='left')

    def recommend(self, selected_genres, year_range, top_n=50):
        # Filter by selected genres and year range
        df = self.movies.copy()
        if selected_genres:
            df = df[df['genres'].str.contains('|'.join(selected_genres), case=False, na=False)]
        df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]

        # (Optional) enforce vote floor like your module:
        # df = df[df['v'] >= self.m]

        # ✅ CHANGED: sort by Bayesian score (fair ranking), but DISPLAY only avg later
        df = df.sort_values('score', ascending=False)

        # Return only the columns your teammate wants to show (no score shown)
        out = df[['clean_title', 'genres', 'year', 'avg']].head(top_n).copy()
        out['avg'] = out['avg'].round(2)  # nice formatting
        return out

# ---------------- STREAMLIT APP ----------------
st.markdown(
    """
    <div style='text-align: center;'>
        <h1 style='color: #FF4B4B; margin-bottom: 0; white-space: nowrap;'>
            🍿 Movie Recommender System 🍿
        </h1>
        <h2 style='color: #FF4B4B; margin-top: 5px;'>
            Genres Explorer
        </h2>
        <h3 style='color: #444; font-weight: normal; margin-top: 5px;'>
            🎬 Pick your favorite genres 🎬 <br>
            Or Let Us Surprise You With A Hidden Gem ✨
        </h3>
        <hr style='border: 1px solid #ddd; margin-top: 10px;'>
    </div>
    """,
    unsafe_allow_html=True
)

# Initialize recommender
recommender = GenreRecommender("dataset/movies.csv", "dataset/ratings.csv")

# Initialize session state
if "filtered_movies" not in st.session_state:
    st.session_state["filtered_movies"] = pd.DataFrame()

# User input: genre selection
all_genres = sorted(set(g for gs in recommender.movies['genres'].dropna().str.split('|') for g in gs))
selected_genres = st.multiselect("🎭 Select genres:", all_genres, default=[])

# User input: year range
min_year, max_year = int(recommender.movies['year'].min()), int(recommender.movies['year'].max())
year_range = st.slider(
    "📅 Select year range:",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year)
)

# ---------------- Show Recommendations ----------------
if st.button("📌 Show Recommendations"):
    if selected_genres:
        filtered = recommender.recommend(selected_genres, year_range, top_n=500)
    else:
        # no genres picked: still sort by score internally via recommend()
        filtered = recommender.recommend([], year_range, top_n=500)

    st.session_state["filtered_movies"] = filtered

# Display table if filtered data exists
if not st.session_state["filtered_movies"].empty:
    filtered_movies_display = st.session_state["filtered_movies"].copy()
    filtered_movies_display['year'] = filtered_movies_display['year'].astype(int)
    filtered_movies_display = filtered_movies_display[['clean_title', 'genres', 'year', 'avg']]  # ✅ CHANGED: show avg only
    filtered_movies_display = filtered_movies_display.rename(columns={
        'clean_title': 'Movies Title',
        'genres': 'Genres',
        'year': 'Year',
        'avg': 'Average Rating'  # ✅ CHANGED: nicer label
    })

    # Summary
    st.info(f"Found {len(filtered_movies_display)} movies matching your selection ({year_range[0]}–{year_range[1]})")

    # Sort dropdown (keep UI behavior; sorts *display* only)
    sort_option = st.selectbox(
        "🔃 Sort movies by:",
        options=["Title (A-Z)", "Year (Ascending)", "Year (Descending)", "Random"]
    )
    if sort_option == "Title (A-Z)":
        filtered_movies_display = filtered_movies_display.sort_values(by='Movies Title', ascending=True)
    elif sort_option == "Year (Ascending)":
        filtered_movies_display = filtered_movies_display.sort_values(by='Year', ascending=True)
    elif sort_option == "Year (Descending)":
        filtered_movies_display = filtered_movies_display.sort_values(by='Year', ascending=False)
    elif sort_option == "Random":
        filtered_movies_display = filtered_movies_display.sample(frac=1).reset_index(drop=True)

    st.dataframe(filtered_movies_display, width=1200, height=800)

# ---------------- Surprise Me ----------------
if st.button("🎲 Surprise Me With RANDOM Suggestion !"):
    # Build the same filtered pool
    pool = recommender.movies.copy()
    if selected_genres:
        pool = pool[pool['genres'].str.contains('|'.join(selected_genres), case=False, na=False)]
    pool = pool[(pool['year'] >= year_range[0]) & (pool['year'] <= year_range[1])]

    # Keep only reasonably good movies by avg if you want (same as before)
    pool = pool[pool['avg'] >= 4.0]

    # ✅ CHANGED: pick from top-K by score for fairness, but DISPLAY avg only
    topK = 50
    pool = pool.sort_values('score', ascending=False).head(topK)

    if len(pool) > 0:
        surprise = pool.sample(1).iloc[0]
        title = surprise['clean_title']
        year = int(surprise['year'])
        rating = round(float(surprise['avg']), 2)  # display average rating only
        genres = surprise['genres']

        st.success(
            f"🎉 Tonight’s Randomly Pick : **{title} ({year})** 🎬\n\n"
            f"⭐ Average Rating : {rating}\n\n"
            f"🎭 Genre : {genres}\n\n"
            f"🔥 Grab your popcorn and enjoy the show!"
        )
    else:
        st.warning("No movies found for this filter with average rating ≥ 4.0.")
