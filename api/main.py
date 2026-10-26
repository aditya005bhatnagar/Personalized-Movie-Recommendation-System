from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from pydantic import BaseModel
from api.database import SessionLocal, User
from sklearn.metrics.pairwise import cosine_similarity
from model.collaborative_model import personalized_recommendations
from api.database import SessionLocal, Rating
import hashlib


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

class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

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

    db = SessionLocal()

    cine_user = db.query(User).filter(
        User.id == user_id
    ).first()

    if not cine_user:
        db.close()
        return {
            "error": "User not found"
        }

    new_ratings = db.query(Rating).filter(
        Rating.user_id == user_id
    ).all()

    db.close()

    if user_id in user_ids:
        user_index = user_ids.index(user_id)

        user_profile = user_movie_matrix.iloc[
            user_index
        ].copy()

        for rating in new_ratings:
            if rating.movie_id in user_profile.index:
                user_profile[rating.movie_id] = rating.rating

        updated_matrix = user_movie_matrix.copy()

        updated_matrix.loc[user_id] = user_profile

        recommendations = personalized_recommendations(
            user_id,
            user_model,
            updated_matrix,
            movies,
            10
        )

    else:

        if len(new_ratings) == 0:
            return {
                "user_id": user_id,
                "recommendations": [],
                "message": "Please rate some movies first."
            }

        user_profile = pd.Series(
            0.0,
            index=user_movie_matrix.columns
        )

        for rating in new_ratings:
            if rating.movie_id in user_profile.index:
                user_profile[rating.movie_id] = rating.rating

        similarities = cosine_similarity(
            [user_profile.values],
            user_movie_matrix.values
        )[0]

        similar_indices = similarities.argsort()[::-1]

        rated_movies = set(
            rating.movie_id for rating in new_ratings
        )

        scores = {}

        for idx in similar_indices[:20]:

            similarity = similarities[idx]

            if similarity <= 0:
                continue

            similar_user = user_movie_matrix.iloc[idx]

            for movie_id, rating in similar_user.items():

                if rating > 0 and movie_id not in rated_movies:

                    if movie_id not in scores:
                        scores[movie_id] = 0

                    scores[movie_id] += (
                        similarity * rating
                    )

        top_movies = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        recommendations = []

        for movie_id, score in top_movies:

            movie_row = movies[
                movies["movieId"] == movie_id
            ]

            if len(movie_row) == 0:
                continue

            recommendations.append({
                "title": movie_row.iloc[0]["title"],
                "genres": movie_row.iloc[0]["genres"],
                "score": round(float(score), 2)
            })

    return {
        "user_id": user_id,
        "username": cine_user.username,
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


@app.post("/register")
def register_user(data: RegisterRequest):

    db = SessionLocal()

    existing_user = db.query(User).filter(
        User.username == data.username
    ).first()

    if existing_user:
        db.close()
        return {
            "message": "Username already exists"
        }

    existing_email = db.query(User).filter(
        User.email == data.email
    ).first()

    if existing_email:
        db.close()
        return {
            "message": "Email already exists"
        }

    hashed_password = hashlib.sha256(
        data.password.encode()
    ).hexdigest()

    new_user = User(
        username=data.username,
        email=data.email,
        password=hashed_password
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    user_id = new_user.id

    db.close()

    return {
        "message": "Registration successful",
        "user_id": user_id,
        "username": new_user.username,
        "email": new_user.email
    }
    
@app.post("/login")
def login_user(data: LoginRequest):

    db = SessionLocal()

    user = db.query(User).filter(
        User.email == data.email
    ).first()

    if not user:
        db.close()
        return {
            "message": "Invalid email or password"
        }

    hashed_password = hashlib.sha256(
        data.password.encode()
    ).hexdigest()

    if hashed_password != user.password:
        db.close()
        return {
            "message": "Invalid email or password"
        }

    db.close()

    return {
        "message": "Login successful",
        "user_id": user.id,
        "username": user.username,
        "email": user.email
    }