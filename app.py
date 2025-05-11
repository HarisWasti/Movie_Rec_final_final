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

@st.cache_resource
def load_df_train():
    url = "https://drive.google.com/uc?id=1RS4_iGakDHUwfa0TA8d4QVdcZrW3GM17"
    output_path = "data/df_train.csv"

    if not os.path.exists(output_path):
        gdown.download(url, output_path, quiet=False)

    df_train = pd.read_csv(output_path)
    return df_train


df_train = load_df_train()




st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("Hybrid Movie Recommender")

# --- Load Data ---
st.info("Loading data and models...")
extra_values, embeddings, cosine_sim, indices = load_embeddings_and_metadata()
ease_B = load_ease_model()
ease_user_map, ease_item_map, ease_idx2item = load_ease_mappings()

# --- UI Controls ---
st.subheader("Tell us what you like")

all_titles = extra_values['title'].dropna().unique().tolist()
selected_movie = st.selectbox("🎞 Pick a movie you enjoy", [""] + sorted(all_titles))

selected_user = st.selectbox(" Pick a sample user", sorted(ease_user_map.keys()))

# --- Recommend ---
if selected_movie and st.button(" Get Recommendations"):
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
        weight_content=0.5,
        top_n=9
    )

    st.subheader(f" Recommendations for: {selected_movie}")
    cols = st.columns(3)
    for idx, (_, row) in enumerate(recs.iterrows()):
        col = cols[idx % 3]
        with col:
            if isinstance(row['poster_url'], str) and row['poster_url'].startswith('http'):
                st.image(row['poster_url'], use_container_width=True)
            st.markdown(f"**{row['title']}**")
            st.caption(f"{row['genres']} | {row['director']} | {row.get('actor1', '')}")
