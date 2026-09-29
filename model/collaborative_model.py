import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

ratings = pd.read_csv("data/ratings.csv")
movies = pd.read_csv("data/movies.csv")

user_movie_matrix = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
).fillna(0)

print("User-Movie Matrix Shape:")
print(user_movie_matrix.shape)

movie_similarity = cosine_similarity(user_movie_matrix.T)

print("\nMovie Similarity Matrix Shape:")
print(movie_similarity.shape)