from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

from model.collaborative_model import personalized_recommendations
from api.database import SessionLocal, Rating


app = FastAPI(
    title="Personalized Movie Recommendation API",
    description="User-Based Collaborative Filtering Recommendation System",
    version="1.0"
)


class RatingRequest(BaseModel):
    user_id: int
    movie_id: int
    rating: float


movies = pd.read_pickle(
    "saved_models/movies.pkl"
)

user_model = joblib.load(
    "saved_models/user_model.pkl"
)

user_movie_matrix = joblib.load(
    "saved_models/user_movie_matrix.pkl"
)

user_ids = joblib.load(
    "saved_models/user_ids.pkl"
)


@app.get("/")
def home():
    return {
        "message": "Movie Recommendation API is running"
    }


@app.get("/users")
def get_users():
    return {
        "users": user_ids
    }


@app.get("/movies")
def get_movies():
    movie_list = movies[
        ["movieId", "title", "genres"]
    ].to_dict(orient="records")

    return {
        "movies": movie_list
    }


@app.get("/movies/{movie_id}")
def get_movie(movie_id: int):

    movie = movies[
        movies["movieId"] == movie_id
    ]

    if len(movie) == 0:
        return {
            "error": "Movie not found"
        }

    movie = movie.iloc[0]

    return {
        "movieId": int(movie["movieId"]),
        "title": movie["title"],
        "genres": movie["genres"]
    }


@app.get("/recommendations/{user_id}")
def get_recommendations(user_id: int):

    if user_id not in user_ids:
        return {
            "error": "User not found"
        }

    user_index = user_ids.index(user_id)

    user_profile = user_movie_matrix.iloc[
        user_index
    ].copy()

    db = SessionLocal()

    new_ratings = db.query(Rating).filter(
        Rating.user_id == user_id
    ).all()

    for rating in new_ratings:

        movie_id = rating.movie_id

        if movie_id in user_profile.index:
            user_profile[movie_id] = rating.rating

    db.close()

    updated_matrix = user_movie_matrix.copy()

    updated_matrix.loc[user_id] = user_profile

    recommendations = personalized_recommendations(
        user_id,
        user_model,
        updated_matrix,
        movies,
        10
    )

    return {
        "user_id": user_id,
        "recommendations": recommendations
    }


@app.post("/ratings")
def add_rating(data: RatingRequest):

    if data.rating < 1 or data.rating > 5:
        return {
            "error": "Rating must be between 1 and 5"
        }

    db = SessionLocal()

    existing_rating = db.query(Rating).filter(
        Rating.user_id == data.user_id,
        Rating.movie_id == data.movie_id
    ).first()

    if existing_rating:

        existing_rating.rating = data.rating

        db.commit()

        db.close()

        return {
            "message": "Rating updated successfully",
            "user_id": data.user_id,
            "movie_id": data.movie_id,
            "rating": data.rating
        }

    new_rating = Rating(
        user_id=data.user_id,
        movie_id=data.movie_id,
        rating=data.rating
    )

    db.add(new_rating)

    db.commit()

    db.refresh(new_rating)

    db.close()

    return {
        "message": "Rating added successfully",
        "user_id": data.user_id,
        "movie_id": data.movie_id,
        "rating": data.rating
    }


@app.get("/ratings/{user_id}")
def get_user_ratings(user_id: int):

    db = SessionLocal()

    ratings = db.query(Rating).filter(
        Rating.user_id == user_id
    ).all()

    result = []

    for rating in ratings:

        result.append({
            "movie_id": rating.movie_id,
            "rating": rating.rating
        })

    db.close()

    return {
        "user_id": user_id,
        "ratings": result
    }