import streamlit as st
import pandas as pd
import re, math, random

MOVIES_PATH  = "dataset/movies.csv"
RATINGS_PATH = "dataset/ratings.csv"

# ------------------ Init Session State ------------------
if "top10" not in st.session_state:
    st.session_state["top10"] = None
    st.session_state["total_matches"] = 0
    st.session_state["C"] = 0
    st.session_state["m"] = 0

# ------------------ Helpers ------------------
@st.cache_data
def load_data(movies_path: str, ratings_path: str):
    movies  = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)

    # Extract year
    if "year" not in movies:
        movies["year"] = movies["title"].apply(
            lambda x: int(re.search(r"\((\d{4})\)", x).group(1)) if re.search(r"\((\d{4})\)", x) else None
        )
    # Clean title
    if "clean_title" not in movies:
        movies["clean_title"] = movies["title"].apply(lambda x: re.sub(r"\s*\(\d{4}\)", "", x))

    return movies, ratings

def compute_weighted_table(ratings, movies, genre_filter=None, year_range=None):
    stats = (
        ratings.groupby("movieId")
               .agg(v=("rating","count"), R=("rating","mean"))
               .reset_index()
    )
    C   = float(ratings["rating"].mean())
    m_q = float(stats["v"].quantile(0.80))
    m   = int(math.ceil(m_q))

    stats["WeightedRating"] = (stats["v"]/(stats["v"]+m))*stats["R"] + (m/(stats["v"]+m))*C
    table = stats.merge(movies, on="movieId", how="left")

    # Apply filters
    if genre_filter and genre_filter != ["All"]:
        regex = "|".join(genre_filter)
        table = table[table["genres"].str.contains(regex, case=False, na=False)]
    if year_range:
        y0, y1 = year_range
        yr = table["year"].astype("Int64")
        table = table[(yr.fillna(-1) >= y0) & (yr <= y1)]

    table = table.query("v >= @m").copy()
    return table, C, m

# ------------------ UI ------------------
st.set_page_config(page_title="Movie Recommender — Genres Module", layout="wide")

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
            🎭 Pick your favorite genres 🎭 <br>
            Or Let Us Surprise You With A Hidden Gem ✨
        </h3>
        <hr style='border: 1px solid #ddd; margin-top: 10px;'>
    </div>
    """,
    unsafe_allow_html=True,
)

movies, ratings = load_data(MOVIES_PATH, RATINGS_PATH)

# Genres list
all_genres = sorted({g for gs in movies["genres"].dropna().str.split("|") for g in gs if g != "(no genres listed)"})
GENRES = ["All"] + all_genres

# Dynamic year range
min_year, max_year = int(movies["year"].min()), int(movies["year"].max())

# User inputs
selected_genres = st.multiselect("🎭 Select genres:", GENRES, default=["All"])
year_range = st.slider("📅 Year range:", min_year, max_year, (min_year, max_year))

# ---------------- Show Recommendations ----------------
if st.button("📌 Show Recommendations"):
    table, C, m = compute_weighted_table(
        ratings, movies,
        genre_filter=selected_genres,
        year_range=year_range
    )

    if not table.empty:
        # Random Top 10 from filter
        top10 = table.sample(n=min(10, len(table)), random_state=random.randint(0, 10000))
        top10 = (
            top10[["clean_title", "genres", "year", "R"]]
            .rename(columns={
                "clean_title": "Movies Title",
                "genres": "Genres",
                "year": "Year",
                "R": "Average Rating"
            })
            .reset_index(drop=True)
        )
        top10["Average Rating"] = top10["Average Rating"].round(1)

        # Save Top 10 in session
        st.session_state["top10"] = top10
        st.session_state["total_matches"] = len(table)
        st.session_state["C"] = C
        st.session_state["m"] = m
    else:
        st.warning("No movies found for this filter.")

# Always show recommendations if available
if st.session_state["top10"] is not None:
    st.markdown(
        f"### 📌 Here Are 10 Randomly Selected Movies (Out of {st.session_state['total_matches']} matches)"
    )
    st.dataframe(st.session_state["top10"], use_container_width=True)

# Surprise Me button
if st.button("🎲 Surprise Me From Top 10"):
    if st.session_state["top10"] is not None:
        surprise = st.session_state["top10"].sample(1).iloc[0]
        st.success(
            f"🎉 Surprise Pick (From The Top 10 List Above): **{surprise['Movies Title']} ({int(surprise['Year'])})** 🎬\n\n"
            f"⭐ Average Rating: {surprise['Average Rating']}\n\n"
            f"🎭 Genres: {surprise['Genres']}"
        )
    else:
        st.warning("⚠️ Please click 'Show Recommendations' first before using Surprise Me!")