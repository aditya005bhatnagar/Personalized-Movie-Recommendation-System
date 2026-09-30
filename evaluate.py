import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors


K = 10

ratings = pd.read_csv("data/ratings.csv")

user_movie_matrix = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
).fillna(0)


model = NearestNeighbors(
    metric="cosine",
    algorithm="brute"
)

model.fit(user_movie_matrix)


precision_scores = []
recall_scores = []


users = ratings["userId"].unique()


for count, user_id in enumerate(users, 1):

    user_ratings = ratings[
        ratings["userId"] == user_id
    ]

    if len(user_ratings) < 5:
        continue

    test_rating = user_ratings.sample(
        n=1,
        random_state=42
    )

    test_movie = test_rating.iloc[0]["movieId"]

    user_index = user_movie_matrix.index.get_loc(
        user_id
    )

    user_vector = user_movie_matrix.iloc[
        user_index
    ].values.copy()

    movie_index = user_movie_matrix.columns.get_loc(
        test_movie
    )

    user_vector[movie_index] = 0

    distances, indices = model.kneighbors(
        [user_vector],
        n_neighbors=6
    )

    similar_users = indices[0][1:]
    similar_distances = distances[0][1:]

    watched_movies = set(
        np.where(user_vector > 0)[0]
    )

    scores = {}

    for i in range(len(similar_users)):

        similarity = 1 - similar_distances[i]

        similar_ratings = user_movie_matrix.iloc[
            similar_users[i]
        ]

        for movie_index, rating in similar_ratings.items():

            if (
                rating > 0
                and user_movie_matrix.columns.get_loc(movie_index)
                not in watched_movies
            ):

                if movie_index not in scores:
                    scores[movie_index] = 0

                scores[movie_index] += (
                    similarity * rating
                )

    recommendations = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:K]

    recommended_movies = {
        movie_id
        for movie_id, score in recommendations
    }

    hit = test_movie in recommended_movies

    precision_scores.append(
        1 / K if hit else 0
    )

    recall_scores.append(
        1 if hit else 0
    )

    if count % 500 == 0:
        print(
            f"Processed {count} users..."
        )


print("\nEvaluation completed.")

print(
    f"Precision@{K}: "
    f"{np.mean(precision_scores):.4f}"
)

print(
    f"Recall@{K}: "
    f"{np.mean(recall_scores):.4f}"
)