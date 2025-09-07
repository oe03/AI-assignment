import re
import os
import numpy as np
import pandas as pd
import streamlit as st
from typing import Tuple
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender — Rating Explorer", layout="wide")

MOVIES_PATH = "dataset/movies.csv"
RATINGS_PATH = "dataset/ratings.csv"

# -----------------------------
# Utilities
# -----------------------------
def _clean_title_get_year(title: str) -> Tuple[str, int]:
    m = re.search(r"\((\d{4})\)", title or "")
    year = int(m.group(1)) if m else None
    clean = re.sub(r"\s*\(\d{4}\)\s*$", "", title or "").strip()
    return clean, year

@st.cache_data(show_spinner=False)
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    movies = pd.read_csv(MOVIES_PATH)
    ratings = pd.read_csv(RATINGS_PATH)

    if "year" not in movies.columns or "clean_title" not in movies.columns:
        cleaned = movies["title"].apply(_clean_title_get_year)
        movies["clean_title"] = cleaned.apply(lambda x: x[0])
        if "year" not in movies.columns:
            movies["year"] = cleaned.apply(lambda x: x[1])

    movies = movies.dropna(subset=["movieId"]).copy()
    ratings = ratings.dropna(subset=["userId", "movieId", "rating"]).copy()

    movies["movieId"] = movies["movieId"].astype(int)
    ratings["userId"] = ratings["userId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)
    ratings["rating"] = ratings["rating"].astype(float)

    return movies, ratings

# -----------------------------
# Item-based CF
# -----------------------------
def build_item_item_model(ratings: pd.DataFrame, min_ratings_per_item: int = 5):
    counts = ratings.groupby("movieId")["rating"].count()
    keep_items = counts[counts >= min_ratings_per_item].index
    r_small = ratings[ratings["movieId"].isin(keep_items)].copy()

    user_item = r_small.pivot_table(index="userId", columns="movieId", values="rating")
    item_means = user_item.mean(axis=0)
    centered = user_item.subtract(item_means, axis=1).fillna(0.0)

    sim = cosine_similarity(centered.T)
    item_ids = centered.columns.to_numpy()
    item_sim = pd.DataFrame(sim, index=item_ids, columns=item_ids)
    np.fill_diagonal(item_sim.values, 0.0)

    return user_item, item_sim, item_means

def recommend_itemcf(user_id: int, user_item: pd.DataFrame, item_sim: pd.DataFrame, item_means: pd.Series) -> pd.Series:
    if user_id not in user_item.index:
        return pd.Series(dtype=float)

    user_ratings = user_item.loc[user_id]
    seen = user_ratings.dropna().index
    if len(seen) == 0:
        return pd.Series(dtype=float)

    centered_by_item = (user_ratings - item_means).dropna()
    rated_items = centered_by_item.index

    candidates = item_sim.index.difference(seen)
    sims = item_sim.loc[candidates, rated_items]

    num = (sims.values * centered_by_item.loc[rated_items].values).sum(axis=1)
    denom = np.abs(sims.values).sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        adj = np.divide(num, denom, out=np.zeros_like(num), where=denom != 0)

    base = item_means.loc[candidates].values
    scores = base + adj

    recs = pd.Series(scores, index=candidates, name="score").sort_values(ascending=False)
    return recs.head(10)

# -----------------------------
# User-based CF
# -----------------------------
def build_user_user_model(ratings: pd.DataFrame, min_ratings_per_user: int = 5):
    counts = ratings.groupby("userId")["rating"].count()
    keep_users = counts[counts >= min_ratings_per_user].index
    r_small = ratings[ratings["userId"].isin(keep_users)].copy()

    user_item = r_small.pivot_table(index="userId", columns="movieId", values="rating")
    user_means = user_item.mean(axis=1)
    centered = user_item.sub(user_means, axis=0).fillna(0.0)

    sim = cosine_similarity(centered.values)
    user_ids = centered.index.to_numpy()
    user_sim = pd.DataFrame(sim, index=user_ids, columns=user_ids)
    np.fill_diagonal(user_sim.values, 0.0)
    return user_item, user_sim, user_means

def recommend_usercf(user_id: int, user_item: pd.DataFrame, user_sim: pd.DataFrame, user_means: pd.Series) -> pd.Series:
    if user_id not in user_item.index:
        return pd.Series(dtype=float)

    target = user_item.loc[user_id]
    unseen = target[target.isna()].index
    if len(unseen) == 0:
        return pd.Series(dtype=float)

    sims = user_sim.loc[user_id]
    others_ratings = user_item[unseen]
    others_centered = others_ratings.sub(user_means, axis=0)

    num = (others_centered.T * sims).T.sum(axis=0)
    denom = (np.abs(sims).values.reshape(-1,1) * (~others_ratings.isna()).astype(float)).sum(axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        adj = np.divide(num.values, denom, out=np.zeros_like(num.values), where=denom!=0)

    base = float(user_means.loc[user_id])
    scores = base + adj
    recs = pd.Series(scores, index=unseen, name="score").sort_values(ascending=False)
    return recs.head(10)

# -----------------------------
# Evaluation (Precision@10 only)
# -----------------------------
def precision_at_k(recommended_ids, relevant_ids, k=10):
    if not relevant_ids:
        return np.nan
    hits = sum(1 for mid in recommended_ids[:k] if mid in relevant_ids)
    return hits / k

def random_user_train_test_split(ratings: pd.DataFrame,
                                 test_size_per_user: int = 2,
                                 min_items_per_user: int = 5,
                                 seed: int = 42):
    rng = np.random.default_rng(seed)
    train_parts, test_parts = [], []
    for uid, g in ratings.groupby("userId"):
        if len(g) >= min_items_per_user:
            idx = g.index.to_list()
            third = len(idx) // 3
            size = min(test_size_per_user, third if third > 0 else 1)
            test_idx = rng.choice(idx, size=size, replace=False)
            test_mask = g.index.isin(test_idx)
            test_parts.append(g.loc[test_mask])
            train_parts.append(g.loc[~test_mask])
        else:
            train_parts.append(g)
    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True) if test_parts else ratings.iloc[0:0].copy()
    return train, test

def evaluate_precision(ratings: pd.DataFrame, algo: str, rel_threshold: float = 4.0, k: int = 10) -> float:
    train, test = random_user_train_test_split(ratings, test_size_per_user=2, min_items_per_user=5, seed=42)

    if algo == "Item-based CF":
        user_item, item_sim, item_means = build_item_item_model(train)
    else:
        user_item, user_sim, user_means = build_user_user_model(train)

    ps = []
    for uid, gtest in test.groupby("userId"):
        rel = set(gtest.loc[gtest["rating"] >= rel_threshold, "movieId"].tolist())
        if not rel:
            continue
        if uid not in user_item.index:
            continue
        if algo == "Item-based CF":
            recs = recommend_itemcf(uid, user_item, item_sim, item_means)
        else:
            recs = recommend_usercf(uid, user_item, user_sim, user_means)
        rec_ids = recs.index.tolist() if not recs.empty else []
        p = precision_at_k(rec_ids, rel, k)
        if not np.isnan(p):
            ps.append(p)

    return float(np.mean(ps)) if ps else np.nan

# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    # Header (styled like Genres Explorer)
    st.markdown(
        '''
        <div style="text-align: center;">
            <h1 style="color: #FF4B4B; margin-bottom: 0; white-space: nowrap;">
                🍿 Movie Recommender System 🍿
            </h1>
            <h2 style="color: #FF4B4B; margin-top: 5px;">
                Rating Explorer
            </h2>
            <hr style="border: 1px solid #ddd; margin-top: 10px;">
        </div>
        ''',
        unsafe_allow_html=True,
    )

    if not (os.path.exists(MOVIES_PATH) and os.path.exists(RATINGS_PATH)):
        st.error("Cannot find dataset files. Expected dataset/movies.csv and dataset/ratings.csv")
        st.stop()

    movies, ratings = load_data()

    user_ids = sorted(ratings["userId"].unique().tolist())
    uid = st.selectbox("Select a user", user_ids, index=0)
    mode = st.radio("Algorithm", ["Item-based CF", "User-based CF"], horizontal=True)

    if mode == "Item-based CF":
        user_item, item_sim, item_means = build_item_item_model(ratings)
        recs = recommend_itemcf(uid, user_item, item_sim, item_means)
    else:
        user_item_u, user_sim, user_means = build_user_user_model(ratings)
        recs = recommend_usercf(uid, user_item_u, user_sim, user_means)

    df = pd.DataFrame({"movieId": recs.index, "score": recs.values})
    df["score"] = df["score"].round(1)
    df = df.merge(movies[["movieId", "clean_title", "genres", "year"]], on="movieId", how="left")
    df = df.rename(columns={"clean_title": "title"})

    st.subheader("Top-10 Personalized Recommendations")
    st.dataframe(df[["movieId", "title", "year", "genres", "score"]], use_container_width=True)

    st.markdown("---")
    st.subheader("Evaluate Precision@10")
    rel_thr = st.slider("Relevant if rating ≥", 3.0, 5.0, 4.0, 0.5)
    if st.button("Run Precision Evaluation"):
        p10 = evaluate_precision(ratings, mode, rel_threshold=rel_thr, k=10)
        if np.isnan(p10):
            st.warning("Not enough eligible users for evaluation.")
        else:
            st.success(f"Precision@10 = {p10:.3f}")

if __name__ == "__main__":
    main()
