import streamlit as st
import pandas as pd
import numpy as np
from rec_engine import hybrid_recommendations
import joblib
from scipy import sparse

# Load preprocessed data
movie_meta = pd.read_csv('data/movie_meta.csv')
tfidf_matrix = joblib.load('data/tfidf_matrix.pkl')
user_movie_ratings = joblib.load('data/user_movie_ratings.pkl')
item_movie_matrix = joblib.load('data/item_movie_matrix.pkl')
knn = joblib.load('data/knn_model.pkl')

st.set_page_config(page_title="Movie Recommender", layout="wide")

st.title("🎬 Movie Recommendation System (Session-Based)")

# Helper for genre selection
all_genres = sorted(set(g for genre_list in movie_meta['genres'].dropna() for g in genre_list.split('|')))
genre_grid_cols = 3
genre_chunks = [all_genres[i:i+genre_grid_cols] for i in range(0, len(all_genres), genre_grid_cols)]

# Step 1: Cold Start Questions
if 'preferences' not in st.session_state:
    st.subheader("👋 Let's get to know your taste")

    st.markdown("### Pick your favorite genre")
    selected_genre = None
    for row in genre_chunks:
        cols = st.columns(genre_grid_cols)
        for idx, genre in enumerate(row):
            if cols[idx].button(genre):
                selected_genre = genre
                st.session_state['selected_genre'] = genre

    st.markdown("### Search for your favorite movie")
    movie_options = movie_meta['title'].dropna().unique().tolist()
    fav_movie = st.selectbox("Select a movie", movie_options)

    if selected_genre and fav_movie and st.button("Submit"):
        st.session_state['preferences'] = {
            'genre': selected_genre,
            'movie': fav_movie,
            'watched': [],
            'disliked': [],
            'rec_index': 9  # start from the 10th when watched
        }
        st.rerun()

    st.stop()

# Step 2: Generate initial recommendations
if 'recommendations' not in st.session_state:
    initial_movie = st.session_state['preferences']['movie']
    st.session_state['recommendations'] = hybrid_recommendations(
        initial_movie, movie_meta, tfidf_matrix, user_movie_ratings, item_movie_matrix, knn)

# Step 3: Display 3x3 Grid of Recommendations
st.subheader("🎥 Recommended Movies for You")
cols = st.columns(3)

for i, movie in enumerate(st.session_state['recommendations'][:9]):
    col = cols[i % 3]
    with col:
        movie_info = movie_meta[movie_meta['title'] == movie].iloc[0]
        with st.container():
            if movie_info['poster_url']:
                st.image(movie_info['poster_url'], use_container_width=True)
            st.markdown(f"**{movie}**")

            # Metadata
            meta_str = f"🎬 {movie_info['genres']} | 👨‍🎓 {movie_info['director']} | 🔞 {movie_info['age_rating'] or 'N/A'}"
            st.caption(meta_str)

            # Hoverable description (flip-like)
            if movie_info['description']:
                with st.expander("🛈 Description"):
                    st.write(movie_info['description'])

            col1, col2 = st.columns(2)
            if col1.button("✅ Watched", key=f"watched_{i}"):
                st.session_state['preferences']['watched'].append(movie)
                recs = st.session_state['recommendations']
                while st.session_state['preferences']['rec_index'] < len(recs):
                    next_rec = recs[st.session_state['preferences']['rec_index']]
                    st.session_state['preferences']['rec_index'] += 1
                    if next_rec not in recs[:9]:
                        st.session_state['recommendations'][i] = next_rec
                        break
                st.rerun()

            if col2.button("🚫 Not Interested", key=f"disliked_{i}"):
                st.session_state['preferences']['disliked'].append(movie)
                recs = st.session_state['recommendations']
                while st.session_state['preferences']['rec_index'] < len(recs):
                    next_rec = recs[st.session_state['preferences']['rec_index']]
                    st.session_state['preferences']['rec_index'] += 1
                    if next_rec not in recs[:9]:
                        st.session_state['recommendations'][i] = next_rec
                        break
                st.rerun()
