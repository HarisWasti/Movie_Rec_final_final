import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import os
import streamlit as st

@st.cache_data
def load_embeddings_and_metadata():
    extra_values = pd.read_csv("data/extra_values.csv")
    embeddings = np.load("data/embeddings.npy")

    cosine_sim = cosine_similarity(embeddings)
    indices = pd.Series(extra_values.index, index=extra_values['tmdbId']).drop_duplicates()

    return extra_values, embeddings, cosine_sim, indices

@st.cache_resource
def load_and_train_ease(df_train, lambda_=300.0):
    user_map = {uid: idx for idx, uid in enumerate(df_train['userId'].unique())}
    item_map = {mid: idx for idx, mid in enumerate(df_train['movieId'].unique())}
    idx2item = {v: k for k, v in item_map.items()}

    rows = df_train['userId'].map(user_map)
    cols = df_train['movieId'].map(item_map)
    data = np.ones(len(df_train))

    X = np.zeros((len(user_map), len(item_map)))
    X[rows, cols] = 1

    G = X.T @ X
    np.fill_diagonal(G, G.diagonal() + lambda_)
    P = np.linalg.inv(G)
    B = P / (-np.diag(P)[:, None])
    np.fill_diagonal(B, 0)

    return B.astype(np.float32), user_map, item_map, idx2item

@st.cache_resource
def get_content_scores(movie_id_cb, cosine_sim, extra_values, indices):
    if movie_id_cb not in indices:
        return {}
    idx = indices[movie_id_cb]
    scores = cosine_sim[idx]
    tmdb_ids = extra_values['tmdbId'].tolist()
    scaled = MinMaxScaler().fit_transform(np.array(scores).reshape(-1, 1)).flatten()
    return dict(zip(tmdb_ids, scaled))

def get_hybrid_recommendations(user_id, movie_id_cb, df_train, extra_values, cosine_sim,
                                indices, ease_B, ease_user_map, ease_item_map, ease_idx2item,
                                weight_content=0.6, top_n=10):
    scores = {}

    # Content-based component
    content_scores = get_content_scores(movie_id_cb, cosine_sim, extra_values, indices)
    for tmdb_id, score in content_scores.items():
        scores[tmdb_id] = weight_content * score

    # Collaborative component (EASE)
    if user_id in ease_user_map:
        u_idx = ease_user_map[user_id]
        user_interacted = df_train[df_train['userId'] == user_id]['movieId']
        seen_idxs = [ease_item_map[mid] for mid in user_interacted if mid in ease_item_map]

        x_u = np.zeros(len(ease_item_map))
        x_u[seen_idxs] = 1

        preds = x_u @ ease_B
        scaled_preds = MinMaxScaler().fit_transform(preds.reshape(-1, 1)).flatten()

        for idx in seen_idxs:
            scaled_preds[idx] = 0.0

        for idx, score in enumerate(scaled_preds):
            movie_id = ease_idx2item[idx]
            tmdb_match = df_train[df_train['movieId'] == movie_id]['tmdbId']
            if not tmdb_match.empty:
                tmdb_id = tmdb_match.iloc[0]
                scores[tmdb_id] = scores.get(tmdb_id, 0) + (1 - weight_content) * score

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_tmdb = [tmdb for tmdb, _ in ranked[:top_n]]

    return extra_values[extra_values['tmdbId'].isin(top_tmdb)].drop_duplicates('tmdbId')

