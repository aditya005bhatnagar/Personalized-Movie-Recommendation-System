import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")

movies["genres"] = movies["genres"].str.replace("|", " ", regex=False)

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(movies["genres"])

content_similarity = cosine_similarity(tfidf_matrix)

user_movie_matrix = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
).fillna(0)

collaborative_similarity = cosine_similarity(user_movie_matrix.T)

movie_ids = user_movie_matrix.columns

def recommend_movies(movie_id, n=10):

    movie_index = movies[movies["movieId"] == movie_id].index[0]

    content_scores = content_similarity[movie_index]

    if movie_id in movie_ids:
        collaborative_index = list(movie_ids).index(movie_id)
        collaborative_scores = collaborative_similarity[collaborative_index]
    else:
        collaborative_scores = [0] * len(movies)

    hybrid_scores = []

    for i in range(len(movies)):

        if movies.iloc[i]["movieId"] in movie_ids:
            coll_index = list(movie_ids).index(movies.iloc[i]["movieId"])
            collaborative_score = collaborative_scores[coll_index]
        else:
            collaborative_score = 0

        score = (
            0.5 * content_scores[i] +
            0.5 * collaborative_score
        )

        hybrid_scores.append(score)

    movie_scores = list(enumerate(hybrid_scores))

    movie_scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []

    for index, score in movie_scores:

        if movies.iloc[index]["movieId"] != movie_id:

            recommendations.append(
                movies.iloc[index]["title"]
            )

        if len(recommendations) == n:
            break

    return recommendations


recommendations = recommend_movies(1)

print("\nHybrid Recommendations for Toy Story:")

for movie in recommendations:
    print(movie)