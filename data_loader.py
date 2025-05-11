import joblib
import pandas as pd

# Define absolute paths (adjust if you move the project)
DATA_DIR = r"C:\Users\haris\OneDrive\Desktop\Streamlit\data"

PATHS = {
    "movie_meta": f"{DATA_DIR}\\movie_meta.csv",
    "tfidf_matrix": f"{DATA_DIR}\\tfidf_matrix.pkl",
    "user_movie_ratings": f"{DATA_DIR}\\user_movie_ratings.pkl",
    "item_movie_matrix": f"{DATA_DIR}\\item_movie_matrix.pkl",
    "knn": f"{DATA_DIR}\\knn_model.pkl"
}

def load_all_data():
    return {
        "movie_meta": pd.read_csv(PATHS["movie_meta"]),
        "tfidf_matrix": joblib.load(PATHS["tfidf_matrix"]),
        "user_movie_ratings": joblib.load(PATHS["user_movie_ratings"]),
        "item_movie_matrix": joblib.load(PATHS["item_movie_matrix"]),
        "knn": joblib.load(PATHS["knn"])
    }
