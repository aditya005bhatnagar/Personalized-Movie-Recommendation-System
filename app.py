import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Movie Recommendation System")
st.write("Get personalized movie recommendations using a hybrid recommendation system.")

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

    collaborative_scores = [0] * len(movies)

    if movie_id in movie_ids:

        collaborative_index = list(movie_ids).index(movie_id)

        for i, mid in enumerate(movie_ids):
            movie_index_from_movies = movies[
                movies["movieId"] == mid
            ].index

            if len(movie_index_from_movies) > 0:
                movie_position = movie_index_from_movies[0]
                collaborative_scores[movie_position] = (
                    collaborative_similarity[collaborative_index][i]
                )

    hybrid_scores = []

    for i in range(len(movies)):

        score = (
            0.5 * content_scores[i]
            + 0.5 * collaborative_scores[i]
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


movie_list = movies["title"].sort_values().tolist()

selected_movie = st.selectbox(
    "Select a movie:",
    movie_list
)

if st.button("Recommend Movies"):

    movie_id = movies[
        movies["title"] == selected_movie
    ]["movieId"].values[0]

    recommendations = recommend_movies(movie_id)

    st.subheader("Recommended Movies")

    for i, movie in enumerate(recommendations, 1):

        st.write(f"{i}. {movie}")