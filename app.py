import os
import re
from difflib import SequenceMatcher



import joblib

import pandas as pd

import requests

import streamlit as st

from dotenv import load_dotenv



from api.database import SessionLocal, Rating





# ---------------- CONFIGURATION ----------------



st.set_page_config(

    page_title="CineMatch - Personalized Movie Recommendation System",

    page_icon="🎬",

    layout="wide"

)



load_dotenv()



TMDB_TOKEN = os.getenv("TMDB_TOKEN")



TMDB_HEADERS = {

    "Authorization": f"Bearer {TMDB_TOKEN}",

    "accept": "application/json"

}



API_URL = "http://127.0.0.1:8000"





# ---------------- MOVIE POSTERS ----------------



@st.cache_data(ttl=86400)

def get_movie_poster(title):

    if not TMDB_TOKEN:

        return ""



    # Extract the release year from MovieLens titles

    title = str(title).strip()

    year_match = re.search(r"\((\d{4})\)\s*$", title)

    year = year_match.group(1) if year_match else None



    clean_title = re.sub(r"\s*\(\d{4}\)\s*$", "", title).strip()



    try:

        # Search using the movie title and release year

        params = {

            "query": clean_title,

            "language": "en-US",

            "include_adult": "false"

        }



        if year:

            params["year"] = year



        response = requests.get(

            "https://api.themoviedb.org/3/search/movie",

            headers=TMDB_HEADERS,

            params=params,

            timeout=15

        )



        response.raise_for_status()

        results = response.json().get("results", [])



        # If year-specific search finds nothing, try without the year

        if not results and year:

            response = requests.get(

                "https://api.themoviedb.org/3/search/movie",

                headers=TMDB_HEADERS,

                params={

                    "query": clean_title,

                    "language": "en-US",

                    "include_adult": "false"

                },

                timeout=15

            )



            response.raise_for_status()

            results = response.json().get("results", [])



        if not results:

            print(f"No TMDB result for: {title}")

            return ""



        # Choose the closest title and release year match

        target = clean_title.casefold()

        best_result = None

        best_score = -1



        for result in results:

            poster_path = result.get("poster_path")

            if not poster_path:

                continue



            candidate = (

                result.get("title")

                or result.get("original_title")

                or ""

            ).casefold()



            similarity = SequenceMatcher(

                None, target, candidate

            ).ratio()



            release_date = result.get("release_date", "")

            candidate_year = (

                release_date[:4] if len(release_date) >= 4 else None

            )



            score = similarity



            if year and candidate_year:

                if year == candidate_year:

                    score += 0.5

                else:

                    score -= 0.5



            if score > best_score:

                best_score = score

                best_result = poster_path



        if best_result:

            return "https://image.tmdb.org/t/p/w500" + best_result



        print(f"No poster available for: {title}")

        return ""



    except requests.exceptions.RequestException as error:

        print(f"TMDB request failed for {title}: {error}")

        return ""





# ---------------- SESSION STATE ----------------



if "logged_in" not in st.session_state:

    st.session_state.logged_in = False



if "user_id" not in st.session_state:

    st.session_state.user_id = None



if "username" not in st.session_state:

    st.session_state.username = None





# ---------------- LOAD DATA AND MODELS ----------------



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





# ---------------- LOGIN AND REGISTRATION ----------------



if not st.session_state.logged_in:



    st.markdown(

        "<h1 style='text-align;'>🎬 CineMatch</h1>",

        unsafe_allow_html=True

    )



    st.markdown(

        "<p style='text-align;'>"

        "Your personalized movie recommendation system"

        "</p>",

        unsafe_allow_html=True

    )



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



            if not email.strip() or not password:

                st.warning("Please enter your email and password.")



            else:

                try:

                    response = requests.post(

                        f"{API_URL}/login",

                        json={

                            "email": email.strip(),

                            "password": password

                        },

                        timeout=15

                    )



                    try:

                        data = response.json()

                    except ValueError:

                        data = {}



                    if response.status_code == 200:

                        if data.get("message") == "Login successful":

                            st.session_state.logged_in = True

                            st.session_state.user_id = data["user_id"]

                            st.session_state.username = data["username"]

                            st.rerun()

                        else:

                            st.error(

                                data.get("message", "Login failed.")

                            )



                    else:

                        st.error(

                            data.get(

                                "detail",

                                data.get(

                                    "message",

                                    "Login failed. Check your credentials."

                                )

                            )

                        )



                except requests.exceptions.RequestException:

                    st.error(

                        "Unable to connect to the backend. "

                        "Make sure FastAPI is running."

                    )



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



            if not username.strip() or not email.strip() or not password:

                st.warning("Please complete all fields.")



            else:

                try:

                    response = requests.post(

                        f"{API_URL}/register",

                        json={

                            "username": username.strip(),

                            "email": email.strip(),

                            "password": password

                        },

                        timeout=15

                    )



                    try:

                        data = response.json()

                    except ValueError:

                        data = {}



                    if response.status_code == 200:

                        if data.get("message") == "Registration successful":

                            st.success(

                                "Registration successful! You can now log in."

                            )

                        else:

                            st.error(

                                data.get(

                                    "message",

                                    "Registration could not be completed."

                                )

                            )

                    else:

                        st.error(

                            data.get(

                                "detail",

                                data.get(

                                    "message",

                                    "Registration failed."

                                )

                            )

                        )



                except requests.exceptions.RequestException:

                    st.error(

                        "Unable to connect to the backend. "

                        "Make sure FastAPI is running."

                    )



    st.stop()





# ---------------- LOAD APP DATA ----------------



try:

    movies = load_movies()

    user_model, user_movie_matrix, user_ids = load_models()



except Exception as error:

    st.error(f"Could not load movie data or models: {error}")

    st.stop()





selected_user = st.session_state.user_id





# ---------------- HEADER ----------------



st.markdown(

    "<h1 style='text-align;'>🎬 CineMatch</h1>",

    unsafe_allow_html=True

)



st.markdown(

    "<h3 style='text-align;'>"

    "Personalized Movie Recommendation System"

    "</h3>",

    unsafe_allow_html=True

)



st.markdown(

    "<p style='text-align;font-size:18px;'>"

    "Discover movies based on users with similar rating patterns"

    "</p>",

    unsafe_allow_html=True

)



st.divider()





# ---------------- SIDEBAR ----------------



with st.sidebar:



    st.header("🎯 Recommendation Settings")



    st.write(f"👤 User: {st.session_state.username}")

    st.write(f"User ID: {selected_user}")



    st.divider()



    st.markdown("### 🤖 Recommendation Method")

    st.info("User-Based Collaborative Filtering")



    st.markdown("### 🔍 Similarity")

    st.info("Cosine Distance")



    st.divider()



    st.caption(

        "Recommendations are generated from the rating "

        "patterns of similar users."

    )



    if st.button("Logout", width="stretch"):

        st.session_state.logged_in = False

        st.session_state.user_id = None

        st.session_state.username = None

        st.rerun()





# ---------------- USER RATINGS ----------------



db = SessionLocal()



try:

    user_ratings_db = (

        db.query(Rating)

        .filter(Rating.user_id == selected_user)

        .all()

    )

finally:

    db.close()





rated_movies = pd.Series(

    {

        rating.movie_id: rating.rating

        for rating in user_ratings_db

    },

    dtype=float

)





# ---------------- USER PROFILE ----------------



st.markdown(

    '<div class="section-title">👤 User Profile</div>',

    unsafe_allow_html=True

)



col1, col2, col3 = st.columns(3)



with col1:

    st.metric("User ID", selected_user)



with col2:

    st.metric("Movies Rated", len(rated_movies))



with col3:

    average_rating = (

        rated_movies.mean()

        if not rated_movies.empty

        else 0

    )



    st.metric(

        "Average Rating",

        f"{average_rating:.2f} ⭐"

    )



st.divider()





# ---------------- HIGHLY RATED MOVIES ----------------





# ---------------- HIGHLY RATED MOVIES ----------------



st.markdown(

    '<div class="section-title">⭐ Your Highly Rated Movies</div>',

    unsafe_allow_html=True

)



top_rated = rated_movies.sort_values(

    ascending=False

).head(5)



if top_rated.empty:

    st.info("This user has not rated any movies.")



else:

    cols = st.columns(5)



    for i, (movie_id, rating) in enumerate(top_rated.items()):

        movie = movies[

            movies["movieId"] == movie_id

        ]



        if not movie.empty:

            title = movie.iloc[0]["title"]



            with cols[i]:

                poster_url = get_movie_poster(title)



                if poster_url:

                    st.image(

                        poster_url,

                        width=150

                    )

                else:

                    st.write("🎬 Poster unavailable")



                st.markdown(f"**{title}**")

                st.write(f"⭐ Rating: {rating}/5")





# ---------------- RATE A MOVIE ----------------



st.subheader("⭐ Rate a Movie")



search_movie = st.text_input(

    "🔎 Search for a movie",

    placeholder="Enter movie name..."

)



if search_movie.strip():

    filtered_movies = movies[

        movies["title"].astype(str).str.contains(

            search_movie.strip(),

            case=False,

            na=False,

            regex=False

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

        format_func=lambda item: item[1]

    )



    selected_movie_id = int(selected_movie[0])



    existing_rating = rated_movies.get(

        selected_movie_id,

        None

    )



    if existing_rating is not None:

        st.info(

            f"You have already rated this movie: "

            f"⭐ {existing_rating}/5"

        )



        default_rating = float(existing_rating)

        button_text = "🔄 Update Rating"



    else:

        default_rating = 5.0

        button_text = "⭐ Submit Rating"



    rating = st.slider(

        "Your Rating",

        min_value=1.0,

        max_value=5.0,

        step=0.5,

        value=default_rating

    )



    if st.button(button_text, width="stretch"):



        try:

            response = requests.post(

                f"{API_URL}/ratings",

                json={

                    "user_id": selected_user,

                    "movie_id": selected_movie_id,

                    "rating": rating

                },

                timeout=15

            )



            try:

                data = response.json()

            except ValueError:

                data = {}



            if response.status_code == 200:

                if "error" in data:

                    st.error(data["error"])

                else:

                    st.success(

                        data.get(

                            "message",

                            "Rating saved successfully."

                        )

                    )

                    st.rerun()



            else:

                st.error(

                    data.get(

                        "detail",

                        data.get("error", "Failed to submit rating.")

                    )

                )



        except requests.exceptions.RequestException:

            st.error(

                "Could not connect to FastAPI. "

                "Make sure the backend is running."

            )



else:

    st.warning("No movies found.")



st.divider()





# ---------------- PERSONALIZED RECOMMENDATIONS ----------------



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

                f"{API_URL}/recommendations/{selected_user}",

                timeout=30

            )



            try:

                data = response.json()

            except ValueError:

                data = {}



            if response.status_code == 200:



                recommendations = data.get(

                    "recommendations",

                    []

                )



                if not recommendations:

                    st.warning(

                        "No recommendations found for this user. "

                        "Try rating more movies first."

                    )



                else:

                    st.success(

                        f"Found {len(recommendations)} "

                        "personalized recommendations!"

                    )



                    st.subheader(

                        f"🍿 Recommended Movies for User {selected_user}"

                    )



                    for i, movie in enumerate(recommendations, 1):



                        col1, col2 = st.columns([1, 3])



                        with col1:

                            poster_url = get_movie_poster(

                                movie["title"]

                            )



                            if poster_url:

                                st.image(

                                    poster_url,

                                    width=150

                                )

                            else:

                                st.write("🎬 Poster unavailable")



                        with col2:

                            st.markdown(

                                f"### #{i} 🎬 {movie['title']}"

                            )



                            st.write(

                                f"🎭 **Genres:** "

                                f"{movie.get('genres', 'Unknown')}"

                            )



                            score = movie.get("score")



                            if score is not None:

                                st.write(

                                    f"⭐ **Recommendation Score:** "

                                    f"{float(score):.2f}"

                                )



                        st.divider()



            else:

                st.error(

                    data.get(

                        "detail",

                        data.get(

                            "error",

                            "Backend returned an error."

                        )

                    )

                )



        except requests.exceptions.RequestException:

            st.error(

                "Could not connect to the FastAPI backend. "

                "Make sure the backend is running."

            )





# ---------------- FOOTER ----------------



st.divider()



st.caption(
    
    "Built with Python • Pandas • Scikit-learn • Streamlit • "

    "User-Based Collaborative Filtering"

)



st.caption(
    
    "This product uses the TMDB API but is not endorsed "
    "or certified by TMDB."

)