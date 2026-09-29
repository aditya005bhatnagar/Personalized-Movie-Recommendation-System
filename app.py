import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)


@st.cache_data
def load_movies():
    return pd.read_pickle("saved_models/movies.pkl")


@st.cache_resource
def load_models():

    tfidf = joblib.load(
        "saved_models/tfidf.pkl"
    )

    content_model = joblib.load(
        "saved_models/content_model.pkl"
    )

    collaborative_model = joblib.load(
        "saved_models/collaborative_model.pkl"
    )

    movie_ids = joblib.load(
        "saved_models/movie_ids.pkl"
    )

    return (
        tfidf,
        content_model,
        collaborative_model,
        movie_ids
    )


movies = load_movies()

tfidf, content_model, collaborative_model, movie_ids = load_models()


def recommend_movies(movie_id, n=10):

    movie_index = movies[
        movies["movieId"] == movie_id
    ].index[0]

    # Content-based recommendations

    distances, indices = content_model.kneighbors(
        tfidf.transform(
            [movies.iloc[movie_index]["genres"]]
        ),
        n_neighbors=n + 1
    )

    content_movies = []

    for index in indices[0][1:]:

        content_movies.append(
            movies.iloc[index]["title"]
        )


    # Collaborative recommendations

    if movie_id in movie_ids:

        collaborative_index = movie_ids.index(
            movie_id
        )

        distances, indices = collaborative_model.kneighbors(
            [collaborative_model._fit_X[
                collaborative_index
            ]],
            n_neighbors=n + 1
        )

        collaborative_movies = []

        for index in indices[0][1:]:

            recommended_movie_id = movie_ids[index]

            result = movies[
                movies["movieId"] == recommended_movie_id
            ]

            if len(result) > 0:

                collaborative_movies.append(
                    result.iloc[0]["title"]
                )

    else:

        collaborative_movies = []


    # Combine recommendations

    recommendations = []

    for movie in content_movies:

        if movie not in recommendations:

            recommendations.append(movie)


    for movie in collaborative_movies:

        if movie not in recommendations:

            recommendations.append(movie)


    return recommendations[:n]


st.title("🎬 Movie Recommendation System")

st.markdown(
    "### Discover movies you may enjoy"
)

st.write(
    "This application combines content-based filtering "
    "and collaborative filtering."
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

    for i, movie in enumerate(
        recommendations, 1
    ):

        st.markdown(
            f"### {i}. {movie}"
        )

        st.divider()


st.caption(
    "Built with Python, Pandas, Scikit-learn and Streamlit"
)