import streamlit as st
import pandas as pd
import os
import gdown
from rec import (
    load_embeddings_and_metadata,
    load_ease_model,
    load_ease_mappings,
    get_hybrid_recommendations
)

st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Hybrid Movie Recommender (EASE + BERT)")

# --- Load Data ---
st.info("Loading data and models...")
extra_values, embeddings, cosine_sim, indices = load_embeddings_and_metadata()
ease_B = load_ease_model()
ease_user_map, ease_item_map, ease_idx2item = load_ease_mappings()

# --- UI Controls ---
st.subheader("Tell us what you like")

all_titles = extra_values['title'].dropna().unique().tolist()
selected_movie = st.selectbox("🎞️ Pick a movie you enjoy", [""] + sorted(all_titles))

user_ids = sorted(extra_values['movieId'].map(ease_item_map).dropna().unique())
selected_user = st.selectbox("👤 Pick a sample user", sorted(ease_user_map.keys()))

weight = st.slider("🔄 Content vs Collaborative Weight", 0.0, 1.0, 0.6, 0.05)

# --- Recommend ---
if selected_movie and st.button("🎯 Get Recommendations"):
    tmdb_id = extra_values[extra_values['title'] == selected_movie]['tmdbId'].iloc[0]

    recs = get_hybrid_recommendations(
        user_id=selected_user,
        movie_id_cb=tmdb_id,
        extra_values=extra_values,
        cosine_sim=cosine_sim,
        indices=indices,
        ease_B=ease_B,
        ease_user_map=ease_user_map,
        ease_item_map=ease_item_map,
        ease_idx2item=ease_idx2item,
        weight_content=weight,
        top_n=10
    )

    st.subheader(f"🎬 Recommendations for: {selected_movie}")
    cols = st.columns(3)
    for idx, (_, row) in enumerate(recs.iterrows()):
        col = cols[idx % 3]
        with col:
            if isinstance(row['poster_url'], str) and row['poster_url'].startswith('http'):
                st.image(row['poster_url'], use_column_width=True)
            st.markdown(f"**{row['title']}**")
            st.caption(f"{row['genres']} | {row['director']} | {row.get('actor1', '')}")
