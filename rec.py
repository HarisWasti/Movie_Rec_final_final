import pandas as pd
import numpy as np
import os
import joblib
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

def load_assets(data_dir="data"):
    df = pd.read_csv(os.path.join(data_dir, "extra_values_filtered.csv"))
    tfidf_vectorizer = joblib.load(os.path.join(data_dir, "tfidf_vectorizer.pkl"))
    tfidf_matrix = sp.load_npz(os.path.join(data_dir, "tfidf_matrix.npz"))
    tfidf_index = joblib.load(os.path.join(data_dir, "tfidf_index.pkl"))
    ease_B = joblib.load(os.path.join(data_dir, "ease_B.pkl"))
    ease_user_map = joblib.load(os.path.join(data_dir, "ease_user_map.pkl"))
    ease_item_map = joblib.load(os.path.join(data_dir, "ease_item_map.pkl"))
    ease_idx2item = joblib.load(os.path.join(data_dir, "ease_idx2item.pkl"))
    movieId_to_tmdbId = joblib.load(os.path.join(data_dir, "movieId_to_tmdbId.pkl"))

    return (
        df, tfidf_vectorizer, tfidf_matrix, tfidf_index,
        ease_B, ease_user_map, ease_item_map, ease_idx2item, movieId_to_tmdbId
    )


def get_hybrid_recommendations(
    user_id, extra_values, tfidf_matrix, tfidf_index,
    ease_B, ease_user_map, ease_item_map, ease_idx2item, movieId_to_tmdbId,
    top_n=9, weight_content=0.6
):
    scores = {}

    if user_id in ease_user_map:
        u_idx = ease_user_map[user_id]
        seen_movies = extra_values[extra_values['userId'] == user_id]['movieId']

        ease_scores = np.dot(
            seen_movies.map(ease_item_map).dropna().apply(
                lambda x: ease_B[x].toarray() if sp.issparse(ease_B) else ease_B[x]
            ).sum(axis=0),
            1
        )
        ease_scores = MinMaxScaler().fit_transform(ease_scores.reshape(1, -1)).flatten()

        for mid in seen_movies:
            if mid in ease_item_map:
                ease_scores[ease_item_map[mid]] = 0.0

        for idx, score in enumerate(ease_scores):
            movie_id = ease_idx2item[idx]
            tmdb_id = movieId_to_tmdbId.get(movie_id)
            if tmdb_id:
                scores[tmdb_id] = (1 - weight_content) * score

    liked_tmdb_ids = extra_values[
        (extra_values['userId'] == user_id) & (extra_values['rating'] >= 4.0)
    ]['tmdbId']
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

