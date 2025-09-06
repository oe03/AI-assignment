# movieRating.py
import math, re #math for cell to round the vote threshold up, re -> regular expression (to pull the year out of movie title)
import pandas as pd #to load the csv
import streamlit as st

MOVIES_PATH  = "dataset/movies.csv"
RATINGS_PATH = "dataset/ratings.csv"

# ------------------ Helpers ------------------
@st.cache_data
def load_data(movies_path: str, ratings_path: str):
    #Load CSV into Data Frame
    movies  = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)

    #if the movies does not already have a year column, extract year from the title 
    if "year" not in movies:
        movies["year"] = movies["title"].apply(
            lambda x: int(re.search(r"\((\d{4})\)", x).group(1)) if re.search(r"\((\d{4})\)", x) else None
        )
    if "clean_title" not in movies:
        movies["clean_title"] = movies["title"].apply(lambda x: re.sub(r"\s*\(\d{4}\)", "", x))

    #Build genre list using dropdown list
    genres = sorted({g for gs in movies["genres"].dropna().str.split("|") for g in gs if g != "(no genres listed)"})
    #Compute min/max year (slider)
    y_min, y_max = int(movies["year"].min()), int(movies["year"].max())
    #return the data
    return movies, ratings, ["All"] + genres, (y_min, y_max)

#create a score table of movies using IMDb
#u may filter by genre and year and choose how to set the min votes threshold (m)
def compute_weighted_table(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    min_votes_quantile: float = 0.80,
    genre_filter: str | None = None,
    year_range: tuple[int,int] | None = None,
    min_votes_abs: int | None = None,
):
    #groups all rows in ratings by movieId
    #v = number of ratings for each movie, R = avg ratings for each movie 
    #(E.g. if there are 3 same movieId with different ratings, then group it all the 3 movieId (v = 3) and sum up the three movieId's ratings and divided by 3
    stats = (
        ratings.groupby("movieId")
               .agg(v=("rating","count"), R=("rating","mean"))
               .reset_index()
    )
    #Compute global constants
    C   = float(ratings["rating"].mean()) #C: global mean rating across all movies
    m_q = float(stats["v"].quantile(min_votes_quantile)) #m_q: the vote count at the chosen quantile.
    m   = int(min_votes_abs) if min_votes_abs is not None else int(math.ceil(m_q)) #the minimum votes threshold. If you didn’t pass min_votes_abs, it uses ceil(m_q).
    
    #Compute weighted rating 
    #Weighted by how many votes (v) and threshold (m)
    stats["WeightedRating"] = (stats["v"]/(stats["v"]+m))*stats["R"] + (m/(stats["v"]+m))*C
    table = stats.merge(movies, on="movieId", how="left") #attach title, genres, year from movies.csv

    #Apply filter
    if genre_filter and genre_filter.lower() != "all":
        table = table[table["genres"].str.contains(genre_filter, case=False, na=False)]
    #If you selected a year range, only keep movies within that range
    if year_range:
        y0, y1 = year_range
        yr = table["year"].astype("Int64")
        table = table[(yr.fillna(-1) >= y0) & (yr <= y1)]

    #Enforce that only movies with at least m votes remain 
    #Sort by weighted score descending (best first)
    table = table.query("v >= @m").sort_values("WeightedRating", ascending=False).copy()
    return table, C, m

def get_top_rated(
    ratings: pd.DataFrame, movies: pd.DataFrame,
    n: int = 10,  # fixed Top-10
    min_votes_quantile: float = 0.80,
    genre_filter: str | None = None,
    year_range: tuple[int,int] | None = None,
    min_votes_abs: int | None = None,
) -> pd.DataFrame:
    table, C, m = compute_weighted_table(
        ratings, movies,
        min_votes_quantile=min_votes_quantile,
        genre_filter=genre_filter,
        year_range=year_range,
        min_votes_abs=min_votes_abs
    )
    out = (
        table[["clean_title","genres","year","v","R","WeightedRating","movieId"]]  # remove "title"
        .head(n)
        .rename(columns={"clean_title":"Movies Title","v":"votes","R":"avg","WeightedRating":"score"})
        .reset_index(drop=True)
    )
    # round to 1 decimal place
    out["avg"]   = out["avg"].round(1)
    out["score"] = out["score"].round(1)
    out.attrs["global_mean_C"] = round(C, 1)
    out.attrs["min_votes_m"]   = m
    return out

# ------------------ UI ------------------
st.set_page_config(page_title="Movie Recommender — Rating Module", layout="wide")

st.markdown(
    """
       <div style='text-align: center;'>
        <h1 style='color: #FF4B4B; margin-bottom: 0; white-space: nowrap;'>
            🍿 Movie Recommender System 🍿
        </h1>
        <h2 style='color: #FF4B4B; margin-top: 5px;'>
            Rating Explorer
        </h2>
        <h3 style='color: #444; font-weight: normal; margin-top: 5px;'>
            ⭐ Top 10 Rated Movies
        </h3>
        <hr style='border: 1px solid #ddd; margin-top: 10px;'>
    </div>
    """,
    unsafe_allow_html=True,
)

movies, ratings, GENRES, (YMIN, YMAX) = load_data(MOVIES_PATH, RATINGS_PATH)

quantile = st.slider("Min votes quantile (m from quantile)", 0.50, 0.95, 0.80, 0.01)
genre = st.selectbox("Genre (optional)", GENRES)
yr = st.slider("Year range", YMIN, YMAX, (YMIN, YMAX))

top = get_top_rated(
    ratings, movies,
    n=10,  # fix to Top 10 rated movies
    min_votes_quantile=quantile,
    genre_filter=None if genre == "All" else genre,
    year_range=yr,
)

st.caption(f"Global mean C = {top.attrs['global_mean_C']:.2f}  |  m = {int(top.attrs['min_votes_m'])} votes")
st.dataframe(
    top.rename(columns={"clean_title":"Movies Title"}),
    use_container_width=True
)

# ------------------ Evaluation (Precision@10) ------------------
def precision_at_k(recs: pd.DataFrame, threshold: float = 4.0, k: int = 10) -> float:
    """Compute Precision@K: fraction of recommended movies with avg >= threshold."""
    topk = recs.head(k)
    relevant = (topk["avg"] >= threshold).sum()
    return relevant / k

if not top.empty:
    prec10 = precision_at_k(top, threshold=4.0, k=10)
    st.metric(label="Precision@10", value=f"{prec10:.2f}")

