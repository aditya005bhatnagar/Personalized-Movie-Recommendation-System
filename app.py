import streamlit as st
import pandas as pd
import joblib
import requests

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

    return user_model, user_movie_matrix, user_ids


movies = load_movies()

user_model, user_movie_matrix, user_ids = load_models()


# ---------------- HEADER ----------------

st.markdown(
    "<h1 style='text-align: center;'>🎬 Personalized Movie Recommendation System</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center; font-size: 18px;'>"
    "Discover movies based on users with similar rating patterns"
    "</p>",
    unsafe_allow_html=True
)

st.divider()


# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.header("🎯 Recommendation Settings")

    selected_user = st.selectbox(
        "Select User",
        user_ids
    )

    st.divider()

    st.markdown("### 🤖 Recommendation Method")

    st.info(
        "User-Based Collaborative Filtering"
    )

    st.markdown("### 🔍 Similarity")

    st.info(
        "Cosine Distance"
    )

    st.divider()

    st.caption(
        "Recommendations are generated from the rating patterns of similar users."
    )


# ---------------- USER DATA ----------------

user_index = user_ids.index(selected_user)

user_ratings = user_movie_matrix.iloc[user_index]

rated_movies = user_ratings[
    user_ratings > 0
].sort_values(
    ascending=False
)


# ---------------- USER SUMMARY ----------------

st.markdown(
    '<div class="section-title">👤 User Profile</div>',
    unsafe_allow_html=True
)


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "User ID",
        selected_user
    )


with col2:

    st.metric(
        "Movies Rated",
        len(rated_movies)
    )


with col3:

    if len(rated_movies) > 0:
        average_rating = rated_movies.mean()
    else:
        average_rating = 0

    st.metric(
        "Average Rating",
        f"{average_rating:.2f} ⭐"
    )


st.divider()


# ---------------- HIGHLY RATED MOVIES ----------------

st.markdown(
    '<div class="section-title">⭐ Your Highly Rated Movies</div>',
    unsafe_allow_html=True
)


top_rated = rated_movies.head(5)


if len(top_rated) == 0:

    st.info(
        "This user has not rated any movies."
    )

else:

    cols = st.columns(5)

    for i, (movie_id, rating) in enumerate(
        top_rated.items()
    ):

        movie = movies[
            movies["movieId"] == movie_id
        ]

        if len(movie) > 0:

            title = movie.iloc[0]["title"]

            with cols[i]:

                st.markdown(
                    f"""
                    <div class="movie-card">

                    <div class="movie-title">
                    🎬
                    </div>

                    <br>

                    <b>{title}</b>

                    <br><br>

                    ⭐ Rating: {rating}

                    </div>
                    """,
                    unsafe_allow_html=True
                )


st.divider()

#---------------- RATE A MOVIE ----------------
st.divider()

st.subheader("⭐ Rate a Movie")

search_movie = st.text_input(
    "🔎 Search for a movie",
    placeholder="Enter movie name..."
)

if search_movie:
    filtered_movies = movies[
        movies["title"].str.contains(
            search_movie,
            case=False,
            na=False
        )
    ].head(20)
else:
    filtered_movies = movies.head(20)

movie_options = filtered_movies[
    ["movieId", "title"]
].values.tolist()

if len(movie_options) > 0:

    selected_movie = st.selectbox(
        "Choose a movie:",
        movie_options,
        format_func=lambda x: x[1]
    )

else:

    selected_movie = None
    st.warning("No movies found.")

rating = st.slider(
    "Give your rating:",
    min_value=1.0,
    max_value=5.0,
    step=0.5,
    value=5.0
)

if st.button("⭐ Submit Rating", width="stretch"):

    if selected_movie is None:
        st.warning("Please select a movie first.")

    else:

        try:
            response = requests.post(
                "http://127.0.0.1:8000/ratings",
                json={
                    "user_id": selected_user,
                    "movie_id": int(selected_movie[0]),
                    "rating": rating
                }
            )

            if response.status_code == 200:

                data = response.json()

                if "error" in data:
                    st.error(data["error"])

                else:
                    st.success(
                        "Rating submitted successfully!"
                    )

            else:
                st.error("Failed to submit rating.")

        except requests.exceptions.ConnectionError:

            st.error(
                "Could not connect to FastAPI. "
                "Make sure the backend is running."
            )
st.divider()

st.subheader("📋 My Ratings")

try:
    response = requests.get(
        f"http://127.0.0.1:8000/ratings/{selected_user}"
    )

    if response.status_code == 200:

        data = response.json()
        ratings = data["ratings"]

        if len(ratings) == 0:

            st.info("You have not submitted any ratings yet.")

        else:

            rating_data = []

            for item in ratings:

                movie = movies[
                    movies["movieId"] == item["movie_id"]
                ]

                if len(movie) > 0:

                    rating_data.append({
                        "Movie": movie.iloc[0]["title"],
                        "Genres": movie.iloc[0]["genres"],
                        "Rating": item["rating"]
                    })

            st.dataframe(
                rating_data,
                width="stretch",
                hide_index=True
            )

    else:

        st.error("Could not load your ratings.")

except requests.exceptions.ConnectionError:

    st.error(
        "Could not connect to FastAPI. "
        "Make sure the backend is running."
    )

# ---------------- RECOMMENDATION BUTTON ----------------

st.subheader("🍿 Personalized Recommendations")

if st.button(
    "🎯 Generate My Recommendations",
    width="stretch"
):

    with st.spinner(
        "Finding users with similar movie preferences..."
    ):

        try:

            response = requests.get(
                f"http://127.0.0.1:8000/recommendations/{selected_user}"
            )

            if response.status_code == 200:

                data = response.json()
                recommendations = data["recommendations"]

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
                        recommendations, 1
                    ):

                        st.markdown(
                            f"### #{i} 🎬 {movie['title']}"
                        )

                        st.write(
                            f"🎭 **Genres:** {movie['genres']}"
                        )

                        st.write(
                            f"⭐ **Recommendation Score:** "
                            f"{movie['score']:.2f}"
                        )

                        st.divider()

            else:

                st.error(
                    "Backend returned an error."
                )

        except requests.exceptions.ConnectionError:

            st.error(
                "Could not connect to the FastAPI backend. "
                "Make sure the backend is running."
            )
# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "Built with Python • Pandas • Scikit-learn • Streamlit • User-Based Collaborative Filtering"
)