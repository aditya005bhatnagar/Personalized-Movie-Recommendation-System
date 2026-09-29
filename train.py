import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


os.makedirs("saved_models", exist_ok=True)

movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")

movies["genres"] = movies["genres"].str.replace(
    "|", " ", regex=False
)


# Content-Based Model

tfidf = TfidfVectorizer()

tfidf_matrix = tfidf.fit_transform(
    movies["genres"]
)

content_model = NearestNeighbors(
    metric="cosine",
    algorithm="brute"
)

content_model.fit(tfidf_matrix)


# Collaborative Filtering Model

user_movie_matrix = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
).fillna(0)

collaborative_model = NearestNeighbors(
    metric="cosine",
    algorithm="brute"
)

collaborative_model.fit(
    user_movie_matrix.T
)


# Save Models

joblib.dump(
    tfidf,
    "saved_models/tfidf.pkl"
)

joblib.dump(
    content_model,
    "saved_models/content_model.pkl"
)

joblib.dump(
    collaborative_model,
    "saved_models/collaborative_model.pkl"
)

joblib.dump(
    user_movie_matrix.columns.tolist(),
    "saved_models/movie_ids.pkl"
)

movies.to_pickle(
    "saved_models/movies.pkl"
)

print("Training completed successfully.")

print("\nMovies:", len(movies))
print("Ratings:", len(ratings))
print("Users:", len(user_movie_matrix))
print("Rated Movies:", len(user_movie_matrix.columns))

print("\nModels saved in saved_models/")