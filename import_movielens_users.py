import pandas as pd
import hashlib

from api.database import SessionLocal, User

users = pd.read_csv("data/users.csv")

db = SessionLocal()

count = 0

for _, row in users.iterrows():

    movie_user_id = int(row["userId"])

    existing_user = db.query(User).filter(
        User.movie_user_id == movie_user_id
    ).first()

    if existing_user:
        continue

    username = f"ml_user_{movie_user_id}"
    email = f"ml_user_{movie_user_id}@cinematch.local"

    temporary_password = f"MovieLens@{movie_user_id}"

    hashed_password = hashlib.sha256(
        temporary_password.encode()
    ).hexdigest()

    new_user = User(
        username=username,
        email=email,
        password=hashed_password,
        movie_user_id=movie_user_id
    )

    db.add(new_user)
    count += 1

db.commit()
db.close()

print(f"Imported {count} MovieLens users.")