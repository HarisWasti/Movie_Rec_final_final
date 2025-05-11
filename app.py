import streamlit as st
import pandas as pd
import numpy as np
import os
import gdown
from rec import (
    load_embeddings_and_metadata,
    load_and_train_ease,
    get_hybrid_recommendations
)

st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Hybrid Movie Recommender (EASE + BERT)")

# --- Download df_train from Google Drive ---
TRAIN_PATH = "data/df_train.csv"
GDRIVE_URL = "https://drive.google.com/uc?id=1BfpN3osymU_SzJTFCDt64LRDyJM44WOH"

os.makedirs("data", exist_ok=True)

if not os.path.exists(TRAIN_PATH):
    st.info("Downloading training data...")
    gdown.download(GDRIVE_URL, TRAIN_PATH, quiet=False)

# --- Load data ---
df_train = pd.read_csv(TRAIN_PATH)
extra_values, embeddings, cosine_sim, indices = load_embeddings_and_metadata()
ease_B, ease_user_map, ease_item_map, ease_idx2item = load_and_train_ease(df_train)

# --- UI Controls ---
st.subheader("Tell us what you like")

all_titles = extra_values['title'].dropna().unique().tolist()
selected_movie = st.selectbox("🎞️ Pick a movie you enjoy", [""] + sorted(all_titles))

user_ids = sorted(df_train['userId'].unique())
selected_user = st.selectbox("👤 Pick a sample user", user_ids)

weight = st.slider("🔄 Content vs Collaborative Weight", 0.0, 1.0, 0.6, 0.05)

# --- Recommend ---
if selected_movie and st.button("🎯 Get Recommendations"):
    tmdb_id = extra_values[extra_values['title'] == selected_movie]['tmdbId'].iloc[0]
    recs = get_hybrid_recommendations(
        user_id=selected_user,
        movie_id_cb=tmdb_id,
        df_train=df_train,
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
