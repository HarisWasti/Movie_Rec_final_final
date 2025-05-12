# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# --- MUST BE FIRST ---
st.set_page_config(page_title="Movie Recommender", layout="wide")

# --- Load resources ---
@st.cache_resource
def load_assets():
    data_dir = "data"
    df = pd.read_csv(os.path.join(data_dir, "extra_values_filtered.csv"))
    tfidf_vectorizer = joblib.load(os.path.join(data_dir, "tfidf_vectorizer.pkl"))
    tfidf_matrix = sp.load_npz(os.path.join(data_dir, "tfidf_matrix.npz"))
    tfidf_index = joblib.load(os.path.join(data_dir, "tfidf_index.pkl"))
    ease_B = joblib.load(os.path.join(data_dir, "ease_B.pkl"))
    ease_user_map = joblib.load(os.path.join(data_dir, "ease_user_map.pkl"))
    ease_item_map = joblib.load(os.path.join(data_dir, "ease_item_map.pkl"))
    ease_idx2item = joblib.load(os.path.join(data_dir, "ease_idx2item.pkl"))
    movieId_to_tmdbId = joblib.load(os.path.join(data_dir, "movieId_to_tmdbId.pkl"))
    return df, tfidf_vectorizer, tfidf_matrix, tfidf_index, ease_B, ease_user_map, ease_item_map, ease_idx2item, movieId_to_tmdbId

extra_values, tfidf_vectorizer, tfidf_matrix, tfidf_index, ease_B, ease_user_map, ease_item_map, ease_idx2item, movieId_to_tmdbId = load_assets()

# --- Hybrid Recommendation ---
def get_hybrid_recommendations(user_id, top_n=9, weight_content=0.6):
    scores = {}

    if user_id in ease_user_map:
        u_idx = ease_user_map[user_id]
        seen_movies = extra_values[extra_values['userId'] == user_id]['movieId']

        ease_scores = np.dot((seen_movies.map(ease_item_map).dropna().apply(lambda x: ease_B[x].toarray() if sp.issparse(ease_B) else ease_B[x]).sum(axis=0)), 1)
        ease_scores = MinMaxScaler().fit_transform(ease_scores.reshape(1, -1)).flatten()

        for mid in seen_movies:
            if mid in ease_item_map:
                ease_scores[ease_item_map[mid]] = 0.0

        for idx, score in enumerate(ease_scores):
            movie_id = ease_idx2item[idx]
            tmdb_id = movieId_to_tmdbId.get(movie_id)
            if tmdb_id:
                scores[tmdb_id] = (1 - weight_content) * score

    liked_tmdb_ids = extra_values[(extra_values['userId'] == user_id) & (extra_values['rating'] >= 4.0)]['tmdbId']
    liked_indices = [tfidf_index.get(tmdb_id) for tmdb_id in liked_tmdb_ids if tmdb_id in tfidf_index]

    tfidf_sim = np.zeros(tfidf_matrix.shape[0])
    for idx in liked_indices:
        if idx is not None:
            tfidf_sim += cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    if liked_indices:
        tfidf_sim /= len(liked_indices)
        for idx, score in enumerate(tfidf_sim):
            tmdb_id = extra_values.iloc[idx]['tmdbId']
            scores[tmdb_id] = scores.get(tmdb_id, 0) + weight_content * score

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_tmdb = [tmdb for tmdb, _ in ranked[:top_n]]
    return extra_values[extra_values['tmdbId'].isin(top_tmdb)].drop_duplicates('tmdbId')

# --- Streamlit UI ---
st.title(" Hybrid Movie Recommender")

# --- Session state for navigation ---
if 'page' not in st.session_state:
    st.session_state.page = 'start'

def reset():
    st.session_state.page = 'start'

# --- User Choice ---
if st.session_state.page == 'start':
    mode = st.radio("Do you have a user ID?", ["Yes", "No"])

    if mode == "Yes":
        user_id_input = st.text_input("Enter your User ID:")
        if st.button("Get Recommendations") and user_id_input.isdigit():
            user_id = int(user_id_input)
            if user_id in ease_user_map:
                st.session_state.page = 'user_recs'
                st.session_state.user_id = user_id
            else:
                st.warning(" Invalid User ID")

    elif mode == "No":
        st.session_state.page = 'cold_start'

# --- Existing User Rec Page ---
if st.session_state.page == 'user_recs':
    user_id = st.session_state.user_id
    recs = get_hybrid_recommendations(user_id)
    st.subheader(f"Top 9 Recommendations for User {user_id}")
    cols = st.columns(3)
    for i, (_, row) in enumerate(recs.iterrows()):
        with cols[i % 3]:
            if isinstance(row.get('poster_url'), str) and row['poster_url'].startswith('http'):
                st.image(row['poster_url'], use_container_width=True)
            st.markdown(f"**{row['title']}**")
            st.caption(f"{row['genres']} | {row['director']}")
    if st.button(" Go Back"):
        reset()

# --- Cold Start Page ---
if st.session_state.page == 'cold_start':
    st.info("Tell us what you like and we'll personalize suggestions")
    all_titles = extra_values['title'].dropna().unique().tolist()
    selected_movies = st.multiselect("Pick 5 movies you like", sorted(all_titles))
    genres_list = extra_values['genres'].dropna().str.split().explode().unique().tolist()
    selected_genres = st.multiselect("Pick 3 genres you enjoy", sorted(genres_list))

    if st.button("Recommend"):
        if len(selected_movies) >= 5 and len(selected_genres) >= 3:
            liked_tmdb_ids = extra_values[extra_values['title'].isin(selected_movies)]['tmdbId'].tolist()
            liked_indices = [tfidf_index.get(tmdb) for tmdb in liked_tmdb_ids if tmdb in tfidf_index]

            tfidf_sim = np.zeros(tfidf_matrix.shape[0])
            for idx in liked_indices:
                if idx is not None:
                    tfidf_sim += cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

            if liked_indices:
                tfidf_sim /= len(liked_indices)
            genre_mask = extra_values['genres'].apply(lambda g: any(genre in g for genre in selected_genres))
            extra_values['content_score'] = tfidf_sim
            filtered = extra_values[genre_mask].sort_values('content_score', ascending=False).drop_duplicates('tmdbId')
            st.subheader(" Top 9 Personalized Picks")
            cols = st.columns(3)
            for i, (_, row) in enumerate(filtered.head(9).iterrows()):
                with cols[i % 3]:
                    if isinstance(row.get('poster_url'), str) and row['poster_url'].startswith('http'):
                        st.image(row['poster_url'], use_container_width=True)
                    st.markdown(f"**{row['title']}**")
                    st.caption(f"{row['genres']} | {row['director']}")
    if st.button("⬅️ Start Over"):
        reset()
