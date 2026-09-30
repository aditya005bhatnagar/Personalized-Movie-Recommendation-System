import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Personalized Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)


@st.cache_data
def load_movies():
    return pd.read_pickle("saved_models/movies.pkl")


@st.cache_resource
def load_models():

    user_model = joblib.load(
        "saved_models/user_model.pkl"
    )

    user_movie_matrix = joblib.load(
        "saved_models/user_movie_matrix.pkl"
    )

    user_ids = joblib.load(
        "saved_models/user_ids.pkl"
    )

    return (
        user_model,
        user_movie_matrix,
        user_ids
    )


movies = load_movies()

user_model, user_movie_matrix, user_ids = load_models()



def personalized_recommendations(user_id, n=10):

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


st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .movie-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.3);
        margin-bottom: 12px;
    }

    .movie-number {
        font-size: 20px;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    '<div class="main-title">'
    '🎬 Personalized Movie Recommendation System'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Get movie recommendations based on users with similar rating patterns'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


st.subheader("👤 Select User")

selected_user = st.selectbox(
    "Choose a user:",
    user_ids
)


user_index = user_ids.index(selected_user)

user_ratings = user_movie_matrix.iloc[user_index]

rated_movies = user_ratings[
    user_ratings > 0
].sort_values(
    ascending=False
)


col1, col2 = st.columns(2)

with col1:

    st.markdown("### 👤 User ID")

    st.info(
        f"User {selected_user}"
    )


with col2:

    st.markdown("### 🎥 Movies Rated")

    st.info(
        f"{len(rated_movies)} movies"
    )


st.divider()


st.subheader("⭐ Movies You Rated Highly")

top_rated = rated_movies.head(5)

for movie_id, rating in top_rated.items():

    movie = movies[
        movies["movieId"] == movie_id
    ]

    if len(movie) > 0:

        title = movie.iloc[0]["title"]

        st.write(
            f"⭐ **{title}** — Rating: {rating}"
        )


st.divider()


if st.button(
    "🎯 Get Personalized Recommendations",
    use_container_width=True
):

    with st.spinner(
        "Finding users with similar movie preferences..."
    ):

        recommendations = personalized_recommendations(
            selected_user,
            10
        )

    if len(recommendations) == 0:

        st.warning(
            "No recommendations found for this user."
        )

    else:

        st.success(
            f"Found {len(recommendations)} personalized recommendations!"
        )

        st.subheader(
            f"🍿 Recommended Movies for User {selected_user}"
        )

        for i, movie in enumerate(
            recommendations,
            1
        ):

            st.markdown(
                f"""
                <div class="movie-card">

                <span class="movie-number">
                {i}. 🎬 {movie['title']}
                </span>

                <br><br>

                <b>Genres:</b>
                {movie['genres']}

                <br>

                <b>Recommendation Score:</b>
                {movie['score']:.2f}

                </div>
                """,
                unsafe_allow_html=True
            )


st.divider()

st.caption(
    "Built with Python • Pandas • Scikit-learn • Streamlit"
)