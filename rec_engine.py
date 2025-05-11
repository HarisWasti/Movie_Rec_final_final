import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# Helper to safely search for movie
def search_movie(title_input, movie_meta):
    title_input = title_input.lower()
    match = movie_meta[movie_meta['title'].str.lower().str.contains(title_input, regex=False)]
    return match['title'].iloc[0] if not match.empty else None

# Hybrid recommendation function
def hybrid_recommendations(title_input, movie_meta, tfidf_matrix, item_movie_matrix, user_movie_ratings, knn, alpha=0.6, penalty_weight=0.2, top_k=9):
    title = search_movie(title_input, movie_meta)
    if not title:
        return ["Movie title not found."]

    idx = movie_meta[movie_meta['title'] == title].index[0]
    movie_id = movie_meta.loc[idx, 'movieId']

    # ---- Content-based similarity ----
    content_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    # ---- Collaborative filtering similarity ----
    try:
        movie_idx_cf = user_movie_ratings.columns.get_loc(movie_id)
    except KeyError:
        return ["Not enough data for collaborative filtering."]

    movie_vector = item_movie_matrix[movie_idx_cf].reshape(1, -1)
    distances, indices_cf = knn.kneighbors(movie_vector, n_neighbors=11)  # include self

    cf_scores = np.zeros_like(content_sim)
    for i, cf_idx in enumerate(indices_cf.flatten()[1:]):  # skip self
        sim = 1 - distances.flatten()[i + 1]  # convert distance to similarity
        cf_scores[cf_idx] = sim

    # ---- Normalize scores ----
    content_sim /= content_sim.max() if content_sim.max() != 0 else 1
    cf_scores /= cf_scores.max() if cf_scores.max() != 0 else 1

    # ---- Hybrid blending ----
    hybrid_score = alpha * cf_scores + (1 - alpha) * content_sim

    # ---- Penalize popularity ----
    if 'rating' in movie_meta.columns:
        popularity = movie_meta['rating'].fillna(0).values
        popularity = (popularity - popularity.min()) / (popularity.max() - popularity.min())
        hybrid_score -= penalty_weight * popularity

    # ---- Exclude original movie ----
    hybrid_score[idx] = -1

    # ---- Get top recommendations ----
    top_indices = hybrid_score.argsort()[::-1][:top_k]
    return movie_meta['title'].iloc[top_indices].tolist()



