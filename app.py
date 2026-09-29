import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)


@st.cache_data
def load_data():

    movies = pd.read_csv("data/movies.csv")
    ratings = pd.read_csv("data/ratings.csv")

    movies["genres"] = movies["genres"].str.replace(
        "|", " ", regex=False
    )

    return movies, ratings


@st.cache_resource
def build_models(movies, ratings):

    tfidf = TfidfVectorizer()

    tfidf_matrix = tfidf.fit_transform(
        movies["genres"]
    )

    content_similarity = cosine_similarity(
        tfidf_matrix
    )

    user_movie_matrix = ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating"
    ).fillna(0)

    collaborative_similarity = cosine_similarity(
        user_movie_matrix.T
    )

    movie_ids = user_movie_matrix.columns

    return (
        content_similarity,
        collaborative_similarity,
        movie_ids
    )


movies, ratings = load_data()

content_similarity, collaborative_similarity, movie_ids = build_models(
    movies,
    ratings
)


def recommend_movies(movie_id, n=10):

    movie_index = movies[
        movies["movieId"] == movie_id
    ].index[0]

    content_scores = content_similarity[movie_index]

    hybrid_scores = []

    collaborative_indexes = {
        movie_id: index
        for index, movie_id in enumerate(movie_ids)
    }

    if movie_id in collaborative_indexes:

        collaborative_index = collaborative_indexes[movie_id]

        collaborative_scores = collaborative_similarity[
            collaborative_index
        ]

    else:

        collaborative_scores = []


    for i in range(len(movies)):

        current_movie_id = movies.iloc[i]["movieId"]

        if current_movie_id in collaborative_indexes:

            coll_index = collaborative_indexes[current_movie_id]

            collaborative_score = collaborative_scores[coll_index]

        else:

            collaborative_score = 0

        score = (
            0.5 * content_scores[i]
            + 0.5 * collaborative_score
        )

        hybrid_scores.append(score)


    movie_scores = list(
        enumerate(hybrid_scores)
    )

    movie_scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []

    for index, score in movie_scores:

        if movies.iloc[index]["movieId"] != movie_id:

            recommendations.append(
                (
                    movies.iloc[index]["title"],
                    movies.iloc[index]["genres"],
                    score
                )
            )

        if len(recommendations) == n:
            break

    return recommendations


st.title("🎬 Movie Recommendation System")

st.markdown(
    "### Discover movies you may enjoy"
)

st.divider()

st.subheader("🎥 Choose a Movie")

movie_list = movies["title"].sort_values().tolist()

selected_movie = st.selectbox(
    "Select a movie you like:",
    movie_list
)

selected_movie_id = movies[
    movies["title"] == selected_movie
]["movieId"].values[0]

selected_genres = movies[
    movies["title"] == selected_movie
]["genres"].values[0]

col1, col2 = st.columns(2)

with col1:

    st.markdown("#### Selected Movie")

    st.info(selected_movie)

with col2:

    st.markdown("#### Genres")

    st.info(selected_genres)


if st.button(
    "🎯 Recommend Movies",
    use_container_width=True
):

    recommendations = recommend_movies(
        selected_movie_id
    )

    st.divider()

    st.subheader(
        f"🎬 Recommendations for {selected_movie}"
    )

    for i, (title, genres, score) in enumerate(
        recommendations, 1
    ):

        col1, col2 = st.columns(
            [1, 5]
        )

        with col1:

            st.markdown(
                f"## {i}"
            )

        with col2:

            st.markdown(
                f"### {title}"
            )

            st.write(
                f"🎭 **Genres:** {genres}"
            )

            st.progress(
                min(float(score), 1.0)
            )

            st.caption(
                f"Hybrid similarity score: {score:.3f}"
            )

        st.divider()


st.caption(
    "Built with Python, Pandas, Scikit-learn and Streamlit"
)

st.write(
    "Get personalized movie recommendations using "
    "a hybrid recommendation system."
)

st.divider()

movie_list = movies["title"].sort_values().tolist()

selected_movie = st.selectbox(
    "Select a movie:",
    movie_list
)

if st.button("Recommend Movies"):

    movie_id = movies[
        movies["title"] == selected_movie
    ]["movieId"].values[0]

    recommendations = recommend_movies(
        movie_id
    )

    st.subheader(
        f"Movies Recommended for {selected_movie}"
    )

    for i, (title, genres, score) in enumerate(
        recommendations, 1
    ):

        st.write(
            f"### {i}. {title}"
        )

        st.write(
            f"Genres: {genres}"
        )

        st.write(
            f"Hybrid Score: {score:.3f}"
        )

        st.divider()