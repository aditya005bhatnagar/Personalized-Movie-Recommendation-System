import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

ratings = pd.read_csv("data/ratings.csv")
movies = pd.read_csv("data/movies.csv")

user_movie_matrix = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
).fillna(0)

movie_similarity = cosine_similarity(user_movie_matrix.T)

movie_ids = user_movie_matrix.columns

def recommend_movies(movie_id, n=10):

    movie_index = list(movie_ids).index(movie_id)

    similarity_scores = list(
        enumerate(movie_similarity[movie_index])
    )

    similarity_scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []

    for index, score in similarity_scores[1:n+1]:

        recommended_movie_id = movie_ids[index]

        movie_title = movies[
            movies["movieId"] == recommended_movie_id
        ]["title"].values[0]

        recommendations.append(movie_title)

    return recommendations


recommendations = recommend_movies(1)

print("\nMovies similar to Toy Story:")

for movie in recommendations:
    print(movie)