import streamlit as st
import pandas as pd
import numpy as np

class PopularityRecommender:
    def __init__(self, movies_file):
        # Load dataset
        self.movies = pd.read_csv(movies_file)

        # Check required columns
        if "title" not in self.movies.columns or "popularity" not in self.movies.columns:
            raise ValueError("CSV must contain 'title' and 'popularity' columns.")

        # Clean dataset
        self.movies = self.movies[['title', 'popularity']].dropna()
        self.movies = self.movies[self.movies['popularity'] > 0].reset_index(drop=True)

    def recommend_by_popularity(self, popularity, locked_range=None):
        """Recommend movies within ±15% popularity range"""
        if not locked_range:
            lower = popularity * 0.85
            upper = popularity * 1.15
        else:
            lower, upper = locked_range

        candidates = self.movies[
            (self.movies['popularity'] >= lower) & (self.movies['popularity'] <= upper)
        ]
        return candidates[['title', 'popularity']], (lower, upper)


# ---------------- STREAMLIT APP ----------------
st.markdown(
    """
    <div style='text-align: center;'>
        <h1 style='color: #FF4B4B; margin-bottom: 0; white-space: nowrap;'>
            🎬 Movie Recommender System 🎬
        </h1>
        <h2 style='color: #FF4B4B; margin-top: 5px;'>
            Popularity Explorer
        </h2>
        <h3 style='color: #444; font-weight: normal; margin-top: 5px;'>
            🌟 Discover Movies with Similar Popularity 🌟 <br>
            🎯 Select up to 10 movies and get smart recommendations ✨
        </h3>
        <hr style='border: 1px solid #ddd; margin-top: 10px;'>
    </div>
    """,
    unsafe_allow_html=True
)

# Initialize recommender
recommender = PopularityRecommender("dataset/RevenueMovies.csv")

# Session state
if "locked_range" not in st.session_state:
    st.session_state["locked_range"] = None
if "selected_movies" not in st.session_state:
    st.session_state["selected_movies"] = []
if "recommendations" not in st.session_state:
    st.session_state["recommendations"] = pd.DataFrame()
if "sample_movies" not in st.session_state:
    st.session_state["sample_movies"] = recommender.movies.sample(min(20, len(recommender.movies))).reset_index(drop=True)
if "selected_recommended" not in st.session_state:
    st.session_state["selected_recommended"] = []  # likes
if "disliked_recommended" not in st.session_state:
    st.session_state["disliked_recommended"] = []  # explicit dislikes
if "user_preferences" not in st.session_state:
    st.session_state["user_preferences"] = []


# Callback helpers to enforce mutual exclusivity
def _on_like_change(title):
    # If like was checked, force dislike to False
    like_key = f"like_{title}"
    dislike_key = f"dislike_{title}"
    if st.session_state.get(like_key, False):
        st.session_state[dislike_key] = False


def _on_dislike_change(title):
    dislike_key = f"dislike_{title}"
    like_key = f"like_{title}"
    if st.session_state.get(dislike_key, False):
        st.session_state[like_key] = False


# ================== Movie Selection ==================
st.subheader("🎥 Select Movies You Watched (Max 10)")

# Keep previously selected movies always visible
selected_df = recommender.movies[recommender.movies['title'].isin(st.session_state["selected_movies"])]
remaining_df = st.session_state["sample_movies"][~st.session_state["sample_movies"]['title'].isin(st.session_state["selected_movies"])]
all_movies_to_show = pd.concat([selected_df, remaining_df]).drop_duplicates().reset_index(drop=True)

new_selected_movies = []
for _, row in all_movies_to_show.iterrows():
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(f"**{row['title']}** (Popularity: {row['popularity']:.2f})")
    with col2:
        key = f"movie_{row['title']}"
        # Ensure key exists so checkbox preserves state across reruns
        if key not in st.session_state:
            st.session_state[key] = (row['title'] in st.session_state["selected_movies"])
        checked = st.checkbox("Select", key=key)
        if checked:
            new_selected_movies.append(row['title'])

# cap at 10
if len(new_selected_movies) > 10:
    st.warning("⚠️ You can only select up to 10 movies.")
    new_selected_movies = new_selected_movies[:10]

# Update session state with current checked movies (unchecking removes it from calculations)
st.session_state["selected_movies"] = new_selected_movies

if st.session_state["selected_movies"]:
    st.info(f"✅ Selected Movies: {', '.join(st.session_state['selected_movies'])}")


# ================== Buttons: Show + Refresh ==================
colA, colB = st.columns([1, 1])
with colA:
    show_recs = st.button("📌 Show Recommendations")
with colB:
    refresh_list = st.button("🔄 Refresh Movie List")

# Refresh list of initial movies (keep selected intact & visible, no dups)
if refresh_list:
    # Keep selected visible, fill remaining slots with new random sample excluding selected
    pool = recommender.movies[~recommender.movies['title'].isin(st.session_state["selected_movies"])]
    slots = max(0, 20 - len(st.session_state["selected_movies"]))
    new_sample = pool.sample(min(slots, len(pool))).reset_index(drop=True)
    st.session_state["sample_movies"] = pd.concat([
        recommender.movies[recommender.movies['title'].isin(st.session_state["selected_movies"])],
        new_sample
    ]).drop_duplicates(subset=['title']).reset_index(drop=True)


# ================== Recommendations ==================
if show_recs:
    all_recs = pd.DataFrame()
    st.session_state["locked_range"] = None  # reset lock when generating new recommendations
    for title in st.session_state["selected_movies"]:
        # Guard if title not found (edge case)
        match = recommender.movies[recommender.movies['title'] == title]
        if match.empty:
            continue
        row = match.iloc[0]
        recs, locked = recommender.recommend_by_popularity(row['popularity'], st.session_state["locked_range"])
        st.session_state["locked_range"] = locked
        recs = recs[recs['title'] != title]
        all_recs = pd.concat([all_recs, recs])

    if not all_recs.empty:
        # Remove any items already marked as watched
        all_recs = all_recs[~all_recs['title'].isin(st.session_state["selected_movies"])]
        st.session_state["recommendations"] = all_recs.drop_duplicates(subset=['title']).sample(min(10, len(all_recs)))
    else:
        st.session_state["recommendations"] = pd.DataFrame()


# Display recommendations
if not st.session_state["recommendations"].empty:
    st.subheader("🎯 Interested in any of the movies below? Tick to mark as interested, or mark as not interested. Refresh if not interested in any.")

    # Render like / dislike checkboxes with mutual exclusivity enforced via callbacks
    for _, row in st.session_state["recommendations"].reset_index(drop=True).iterrows():
        title = row['title']
        col1, col2, col3 = st.columns([4, 1.2, 1.8])
        with col1:
            st.write(f"**{title}** (Popularity: {row['popularity']:.2f})")
        like_key = f"like_{title}"
        dislike_key = f"dislike_{title}"

        # Initialize keys if they don't exist
        if like_key not in st.session_state:
            st.session_state[like_key] = (title in st.session_state["selected_recommended"])
        if dislike_key not in st.session_state:
            st.session_state[dislike_key] = (title in st.session_state["disliked_recommended"])

        with col2:
            st.checkbox("Like", key=like_key, on_change=_on_like_change, args=(title,))
        with col3:
            st.checkbox("Not interested", key=dislike_key, on_change=_on_dislike_change, args=(title,))

    # After rendering, collect selections
    like_titles = [t for t in st.session_state["recommendations"]["title"].tolist() if st.session_state.get(f"like_{t}", False)]
    dislike_titles = [t for t in st.session_state["recommendations"]["title"].tolist() if st.session_state.get(f"dislike_{t}", False)]

    # Optional cap on positive selections
    if len(like_titles) > 5:
        st.warning("⚠️ You can only choose up to 5 liked movies from recommendations.")
        like_titles = like_titles[:5]

    # Persist selections across reruns
    st.session_state["selected_recommended"] = like_titles
    st.session_state["disliked_recommended"] = dislike_titles

    if st.session_state["selected_recommended"]:
        st.success(f"✨ Liked: {', '.join(st.session_state['selected_recommended'])}")
    if st.session_state["disliked_recommended"]:
        st.info(f"🙅 Not interested: {', '.join(st.session_state['disliked_recommended'])}")

    # Capture user preferences for future reference (only likes)
    if st.session_state["selected_recommended"]:
        chosen = st.session_state["recommendations"][
            st.session_state["recommendations"]["title"].isin(st.session_state["selected_recommended"])
        ]
        st.session_state["user_preferences"].extend(chosen.to_dict('records'))

    # Refresh recommendations button (keep selections)
    if st.button("🔄 Refresh Recommendations"):
        st.session_state["recommendations"] = st.session_state["recommendations"].sample(frac=1).reset_index(drop=True)

    # ================== Precision (Feedback-based) ==================
    # We compute precision using only items the user explicitly evaluated (liked or not interested)
    evaluated = len(set(st.session_state["selected_recommended"]) | set(st.session_state["disliked_recommended"]))
    relevant = len(st.session_state["selected_recommended"])

    if evaluated > 0:
        precision = relevant / evaluated
        st.subheader("📊 Recommendation Precision (Based on Your Feedback)")
        st.metric("Precision", f"{precision:.2f}")
        with st.expander("What does this mean?"):
            st.write(
                "Precision = Liked ÷ (Liked + Not interested). "
            )
    else:
        st.info("Mark some recommendations as Like or Not interested to see precision.")