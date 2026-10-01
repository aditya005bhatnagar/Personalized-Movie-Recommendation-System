import streamlit as st
import pandas as pd
import joblib
import requests

from api.database import SessionLocal, Rating

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "movie_user_id" not in st.session_state:
    st.session_state.movie_user_id = None

if "username" not in st.session_state:
    st.session_state.username = None

st.set_page_config(
    page_title="CineMatch - Personalized Movie Recommendation System",
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



if not st.session_state.logged_in:

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        st.subheader("Login")

        email = st.text_input(
            "Email",
            key="login_email"
        )

        password = st.text_input(
            "Password",
            type="password",
            key="login_password"
        )

        if st.button("Login", width="stretch"):
            response = requests.post(
                "http://127.0.0.1:8000/login",
                json={
                    "email": email,
                    "password": password
                }
            )

            if response.status_code == 200:
                data = response.json()

                if data.get("message") == "Login successful":
                    st.session_state.logged_in = True
                    st.session_state.user_id = data["user_id"]
                    st.session_state.username = data["username"]
                    st.session_state.movie_user_id = data.get("movie_user_id")
                    st.rerun()
                else:
                    st.error(data.get("message"))
            else:
                st.error("Unable to connect to the backend.")

    with tab2:
        st.subheader("Create Account")

        username = st.text_input(
            "Username",
            key="register_username"
        )

        email = st.text_input(
            "Email",
            key="register_email"
        )

        password = st.text_input(
            "Password",
            type="password",
            key="register_password"
        )

        if st.button("Register", width="stretch"):
            response = requests.post(
                "http://127.0.0.1:8000/register",
                json={
                    "username": username,
                    "email": email,
                    "password": password
                }
            )

            if response.status_code == 200:
                data = response.json()

                if data.get("message") == "Registration successful":
                    st.success(
                        "Registration successful! You can now login."
                    )
                else:
                    st.error(data.get("message"))
            else:
                st.error("Unable to connect to the backend.")

    st.stop()

movies = load_movies()

user_model, user_movie_matrix, user_ids = load_models()


# ---------------- HEADER ----------------

st.markdown(
    "<h1 style='text-align: center;'>🎬 CineMatch</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<h3 style='text-align: center;'>Personalized Movie Recommendation System</h3>",
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
    
    if st.button("Logout", width="stretch"):
       st.session_state.logged_in = False
       st.session_state.user_id = None
       st.session_state.username = None
       st.session_state.movie_user_id = None
       st.rerun()
    
    st.header("🎯 Recommendation Settings")

    
    selected_user = st.session_state.user_id

    st.write(
        f"👤 User: {st.session_state.username}"
    )

    st.write(
        f"User ID: {selected_user}"
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

db = SessionLocal()

user_ratings_db = db.query(Rating).filter(
    Rating.user_id == selected_user
).all()

db.close()

rated_movies = pd.Series(
    {
        rating.movie_id: rating.rating
        for rating in user_ratings_db
    }
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
        average_rating = rated_movies.mean() if len(rated_movies) > 0 else 0
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