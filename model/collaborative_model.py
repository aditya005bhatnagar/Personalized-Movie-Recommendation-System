import pandas as pd
from sklearn.neighbors import NearestNeighbors


def create_user_movie_matrix(ratings):
    user_movie_matrix = ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating"
    ).fillna(0)

    return user_movie_matrix


def train_user_model(user_movie_matrix):
    user_model = NearestNeighbors(
        metric="cosine",
        algorithm="brute"
    )

    user_model.fit(user_movie_matrix)

    return user_model


def personalized_recommendations(
    user_id,
    user_model,
    user_movie_matrix,
    movies,
    n=10
):
    user_ids = user_movie_matrix.index.tolist()

    if user_id not in user_ids:
        return []

    user_index = user_ids.index(user_id)

    distances, indices = user_model.kneighbors(
        [user_movie_matrix.iloc[user_index].values],
        n_neighbors=6
    )

    similar_users = indices[0][1:]
    similar_distances = distances[0][1:]

    watched_movies = set(
        user_movie_matrix.iloc[user_index]
        .loc[lambda x: x > 0]
        .index
    )

    scores = {}

    for i in range(len(similar_users)):

        similar_user_index = similar_users[i]

        similarity = 1 - similar_distances[i]

        ratings = user_movie_matrix.iloc[
            similar_user_index
        ]

        for movie_id, rating in ratings.items():

            if (
                rating > 0
                and movie_id not in watched_movies
            ):

                if movie_id not in scores:
                    scores[movie_id] = 0

                scores[movie_id] += (
                    similarity * rating
                )

    recommended_movies = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    result = []

    for movie_id, score in recommended_movies:

        movie = movies[
            movies["movieId"] == movie_id
        ]

        if len(movie) > 0:

            result.append({
                "title": movie.iloc[0]["title"],
                "genres": movie.iloc[0]["genres"],
                "score": score
            })

        if len(result) == n:
            break

    return result