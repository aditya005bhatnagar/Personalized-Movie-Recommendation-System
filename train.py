import os
import joblib
import pandas as pd
from sklearn.neighbors import NearestNeighbors

os.makedirs("saved_models", exist_ok=True)

movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")

user_movie_matrix = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
).fillna(0)

user_model = NearestNeighbors(
    metric="cosine",
    algorithm="brute"
)

user_model.fit(user_movie_matrix)

joblib.dump(
    user_model,
    "saved_models/user_model.pkl"
)

joblib.dump(
    user_movie_matrix,
    "saved_models/user_movie_matrix.pkl"
)

joblib.dump(
    user_movie_matrix.index.tolist(),
    "saved_models/user_ids.pkl"
)

movies.to_pickle(
    "saved_models/movies.pkl"
)

print("Training completed successfully.")

print("\nMovies:", len(movies))
print("Ratings:", len(ratings))
print("Users:", len(user_movie_matrix))
print("Rated Movies:", len(user_movie_matrix.columns))

print("\nUser-based collaborative filtering model saved successfully.")